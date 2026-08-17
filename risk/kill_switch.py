"""
Global kill switch and trading halt state.

Design decisions:

- The kill switch is manual-reset-only by default. Anything that trips it
  represents a condition the system could not resolve on its own; letting
  it auto-clear on a timer would mean the system resumes trading without
  anyone having understood what happened. `auto_resettable=False` is
  enforced in `deactivate()`.

- There are two distinct concepts, deliberately not merged:
    * KILL SWITCH — hard stop, operator-level, survives reconnects.
    * TRADING HALT — soft stop with a cause (stale data, reconciliation
      pending, disconnect). Clears automatically when the cause clears.
  Merging them would mean a transient data gap requires a human to reset,
  or worse, that a genuine emergency clears itself.

- Emergency flattening is *not* automatic. `EmergencyPolicy` defaults to
  CANCEL_ONLY. Liquidating a book into whatever conditions triggered the
  kill switch (a crash, a data outage, a fat-finger) can be worse than
  holding. Flattening happens only if an operator explicitly configured
  FLATTEN_ALL, per the spec's "optionally flatten ... according to an
  explicitly configured emergency policy".

- The AI layer has no reference to this object and no method to clear it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

import structlog
from pydantic import BaseModel, Field

log = structlog.get_logger(__name__)


class EmergencyPolicy(StrEnum):
    """What happens to existing exposure when the kill switch trips."""

    CANCEL_ONLY = "CANCEL_ONLY"  # cancel working orders, hold positions
    FLATTEN_ALL = "FLATTEN_ALL"  # cancel + close positions at market
    HOLD = "HOLD"  # do nothing; stop new orders only


class KillSwitchTrigger(StrEnum):
    MANUAL = "MANUAL"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    MAX_DRAWDOWN = "MAX_DRAWDOWN"
    BROKER_DISCONNECT = "BROKER_DISCONNECT"
    RECONCILIATION_FAILURE = "RECONCILIATION_FAILURE"
    ABNORMAL_MARKET = "ABNORMAL_MARKET"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    ORDER_REJECTION_STORM = "ORDER_REJECTION_STORM"


class KillSwitchEvent(BaseModel):
    trigger: KillSwitchTrigger
    detail: str
    activated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    deactivated_at: datetime | None = None


class HaltReason(StrEnum):
    BROKER_DISCONNECTED = "BROKER_DISCONNECTED"
    RECONCILIATION_PENDING = "RECONCILIATION_PENDING"
    STALE_MARKET_DATA = "STALE_MARKET_DATA"
    DATABASE_UNAVAILABLE = "DATABASE_UNAVAILABLE"
    OUTSIDE_TRADING_HOURS = "OUTSIDE_TRADING_HOURS"
    STARTUP = "STARTUP"


class KillSwitch:
    def __init__(
        self,
        *,
        emergency_policy: EmergencyPolicy = EmergencyPolicy.CANCEL_ONLY,
        auto_resettable: bool = False,
    ) -> None:
        self._active = False
        self._current: KillSwitchEvent | None = None
        self._history: list[KillSwitchEvent] = []
        self._policy = emergency_policy
        self._auto_resettable = auto_resettable

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def emergency_policy(self) -> EmergencyPolicy:
        return self._policy

    @property
    def current_event(self) -> KillSwitchEvent | None:
        return self._current

    @property
    def history(self) -> list[KillSwitchEvent]:
        return list(self._history)

    def activate(self, trigger: KillSwitchTrigger, detail: str = "") -> KillSwitchEvent:
        """Trip the kill switch. Idempotent — re-activating while already
        active keeps the ORIGINAL event, because the first cause is the
        one worth investigating."""
        if self._active and self._current is not None:
            log.warning(
                "kill_switch.already_active",
                original_trigger=self._current.trigger,
                new_trigger=trigger,
            )
            return self._current

        event = KillSwitchEvent(trigger=trigger, detail=detail)
        self._active = True
        self._current = event
        self._history.append(event)
        log.critical(
            "kill_switch.activated",
            trigger=trigger,
            detail=detail,
            policy=self._policy,
        )
        return event

    def deactivate(self, *, operator_confirmed: bool = False) -> None:
        """Clear the kill switch. Requires explicit operator confirmation
        unless the switch was constructed as auto-resettable. We do not
        let the system talk itself back into trading."""
        if not self._active:
            return
        if not self._auto_resettable and not operator_confirmed:
            raise PermissionError(
                "Kill switch reset requires operator_confirmed=True. "
                "Automated components must not clear the kill switch."
            )
        if self._current is not None:
            self._current.deactivated_at = datetime.now(timezone.utc)
        log.warning(
            "kill_switch.deactivated",
            trigger=self._current.trigger if self._current else None,
            operator_confirmed=operator_confirmed,
        )
        self._active = False
        self._current = None


class TradingHalt:
    """Soft, self-clearing halts keyed by cause. Trading is permitted only
    when no causes are active. Multiple causes can overlap; clearing one
    does not resume trading while others remain."""

    def __init__(self) -> None:
        self._reasons: dict[HaltReason, str] = {}

    @property
    def is_halted(self) -> bool:
        return bool(self._reasons)

    @property
    def reasons(self) -> dict[HaltReason, str]:
        return dict(self._reasons)

    def set(self, reason: HaltReason, detail: str = "") -> None:
        if reason not in self._reasons:
            log.warning("trading_halt.set", reason=reason, detail=detail)
        self._reasons[reason] = detail

    def clear(self, reason: HaltReason) -> None:
        if self._reasons.pop(reason, None) is not None:
            log.info("trading_halt.cleared", reason=reason, remaining=list(self._reasons))

    def clear_all(self) -> None:
        self._reasons.clear()
        log.info("trading_halt.cleared_all")
