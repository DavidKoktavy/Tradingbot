"""
Execution mode gate.

The spec requires a strict progression:

    BACKTEST -> SIMULATION -> PAPER -> LIVE

Design decisions:

- **The gate is checked at the submission boundary, not only at startup.**
  Config validation (Phase 1) prevents the process from *starting* in an
  inconsistent LIVE state. This module prevents an order from *reaching a
  live broker* even if something later in the process believes it should.
  Two independent checks at two different layers, because a single check
  is a single bug away from being bypassed.

- **`assert_can_submit_live()` raises rather than returning a bool.** A
  boolean can be ignored by a caller that forgets to check it; an
  exception cannot. The failure mode of forgetting to call it at all is
  handled by having exactly one submission path, which calls it.

- **Promotion is one step at a time and requires explicit confirmation.**
  There is no `promote_to(LIVE)` that skips stages. Good paper results do
  not authorise live trading — the spec is explicit about this, and the
  code refuses to encode "performance was good" as an input to promotion
  at all.

- The AI layer has no reference to this module. Nothing it can return can
  reach these functions.
"""

from __future__ import annotations

from datetime import datetime, timezone

import structlog
from pydantic import BaseModel, Field

from app.config import TradingMode

log = structlog.get_logger(__name__)

LIVE_CONFIRMATION_PHRASE = "I_UNDERSTAND_THIS_ENABLES_REAL_ORDERS"

# The only legal promotions. Note the absence of any edge that skips a
# stage, and the absence of any edge INTO LIVE that is not from PAPER.
_PROMOTION_PATH: dict[TradingMode, TradingMode | None] = {
    TradingMode.BACKTEST: TradingMode.SIMULATION,
    TradingMode.SIMULATION: TradingMode.PAPER,
    TradingMode.PAPER: TradingMode.LIVE,
    TradingMode.LIVE: None,
}

# Modes in which orders may reach an external broker at all.
_BROKER_MODES = frozenset({TradingMode.PAPER, TradingMode.LIVE})


class LiveTradingNotAuthorised(Exception):
    """Raised when live submission is attempted without full authorisation.
    Never caught and downgraded to a warning."""


class IllegalModePromotion(Exception):
    """Raised on an attempt to skip a stage in the progression."""


class ModeAuthorisation(BaseModel):
    """Evidence that live trading was explicitly authorised. All three
    conditions are required; none has a default that satisfies it."""

    enable_live_trading: bool = False
    confirmation_phrase: str = ""
    operator_acknowledged: bool = False
    authorised_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_complete(self) -> bool:
        return (
            self.enable_live_trading
            and self.confirmation_phrase == LIVE_CONFIRMATION_PHRASE
            and self.operator_acknowledged
        )

    def missing(self) -> list[str]:
        gaps = []
        if not self.enable_live_trading:
            gaps.append("ENABLE_LIVE_TRADING is not true")
        if self.confirmation_phrase != LIVE_CONFIRMATION_PHRASE:
            gaps.append("LIVE_TRADING_CONFIRMATION does not match the required phrase")
        if not self.operator_acknowledged:
            gaps.append("operator acknowledgement was not given")
        return gaps


class ModeGate:
    """Owns the current execution mode and guards the live boundary."""

    def __init__(
        self,
        mode: TradingMode = TradingMode.PAPER,
        *,
        authorisation: ModeAuthorisation | None = None,
    ) -> None:
        self._mode = mode
        self._auth = authorisation or ModeAuthorisation()
        if mode is TradingMode.LIVE and not self._auth.is_complete:
            raise LiveTradingNotAuthorised(
                "Cannot construct a ModeGate in LIVE mode without complete "
                f"authorisation. Missing: {'; '.join(self._auth.missing())}"
            )

    @property
    def mode(self) -> TradingMode:
        return self._mode

    @property
    def is_live(self) -> bool:
        return self._mode is TradingMode.LIVE

    @property
    def uses_real_broker(self) -> bool:
        """PAPER also talks to a real broker connection (IBKR paper
        account), so connection handling must be identical to LIVE."""
        return self._mode in _BROKER_MODES

    @property
    def authorisation(self) -> ModeAuthorisation:
        return self._auth

    def assert_can_submit(self) -> None:
        """Called on every submission path. Raises if the current mode is
        not permitted to send orders to a broker."""
        if self._mode not in _BROKER_MODES:
            raise LiveTradingNotAuthorised(
                f"Mode {self._mode} does not submit orders to a broker. "
                "BACKTEST and SIMULATION use internal fill simulation."
            )
        if self._mode is TradingMode.LIVE:
            self.assert_can_submit_live()

    def assert_can_submit_live(self) -> None:
        """The final gate before a real order. Raises unless every
        authorisation condition is satisfied."""
        if self._mode is not TradingMode.LIVE:
            raise LiveTradingNotAuthorised(
                f"assert_can_submit_live called in mode {self._mode}"
            )
        if not self._auth.is_complete:
            log.critical(
                "live_submission_blocked", missing=self._auth.missing()
            )
            raise LiveTradingNotAuthorised(
                "Live order submission refused. Missing: "
                f"{'; '.join(self._auth.missing())}"
            )

    def promote(
        self, target: TradingMode, *, authorisation: ModeAuthorisation | None = None
    ) -> None:
        """Advance exactly one stage. Refuses to skip stages, and refuses
        to enter LIVE without complete authorisation.

        Deliberately takes no performance argument: backtest or paper
        results are not an input to this decision.
        """
        expected = _PROMOTION_PATH[self._mode]
        if expected is None:
            raise IllegalModePromotion(f"{self._mode} is terminal; cannot promote further")
        if target is not expected:
            raise IllegalModePromotion(
                f"Illegal promotion {self._mode} -> {target}. "
                f"The only legal next stage is {expected}."
            )

        if target is TradingMode.LIVE:
            auth = authorisation or self._auth
            if not auth.is_complete:
                raise LiveTradingNotAuthorised(
                    "Promotion to LIVE refused. Missing: " + "; ".join(auth.missing())
                )
            self._auth = auth
            log.critical(
                "mode.promoted_to_live",
                authorised_at=auth.authorised_at.isoformat(),
            )
        else:
            log.warning("mode.promoted", from_mode=self._mode, to_mode=target)

        self._mode = target

    def demote_to_safe(self, reason: str) -> None:
        """Drop out of LIVE immediately. Always permitted — reducing
        exposure to risk never requires authorisation."""
        if self._mode is TradingMode.LIVE:
            log.critical("mode.demoted_from_live", reason=reason)
        self._mode = TradingMode.PAPER


def gate_from_settings(settings: object) -> ModeGate:
    """Build a ModeGate from application settings.

    `operator_acknowledged` is set from the same explicit env-var evidence
    the config layer already validated. It is a separate field so that a
    future interactive confirmation can populate it without weakening the
    other two conditions.
    """
    mode = getattr(settings, "trading_mode", TradingMode.PAPER)
    enable = bool(getattr(settings, "enable_live_trading", False))
    phrase = str(getattr(settings, "live_trading_confirmation", ""))
    auth = ModeAuthorisation(
        enable_live_trading=enable,
        confirmation_phrase=phrase,
        operator_acknowledged=enable and phrase == LIVE_CONFIRMATION_PHRASE,
    )
    return ModeGate(mode, authorisation=auth)
