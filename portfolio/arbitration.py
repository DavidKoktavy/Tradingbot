"""
Signal arbitration.

Closes the Phase 5 limitation: if two strategies propose opposing trades in
the same instrument, both intents previously reached the risk engine
independently, which could net out to whipsawing in and out of a position
within one cycle.

Design decisions:

- **Arbitration happens before the risk engine, and reduces only.** It can
  drop or shrink a proposal; it can never create one, enlarge one, or
  approve one. The risk engine remains the sole authority on whether a
  trade is permissible — this layer only decides *which* of several
  competing proposals is worth asking about.

- **Direct conflicts resolve to no trade, not to the stronger signal.**
  When strategies disagree about direction, the honest reading is that the
  evidence is mixed. Picking the higher-confidence side dresses up a
  coin-flip as a decision, and confidence scores across different
  strategies are not calibrated against each other, so comparing them is
  comparing different units.

- **Exits always beat entries.** A FLAT proposal (close the position) wins
  over any entry in the same instrument. Blocking an exit to take an entry
  is the asymmetry that turns a manageable loss into an unmanageable one.

- **Deduplication is by instrument and direction**, so two strategies
  independently reaching the same conclusion produce one order at one
  size, not two. Agreement is not a reason to double the position.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum

import structlog

from data.models import Instrument
from execution.execution_models import OrderIntent, OrderSide
from portfolio.positions import Position
from strategies.base import Signal, SignalDirection

log = structlog.get_logger(__name__)


class ArbitrationOutcome(StrEnum):
    ACCEPTED = "ACCEPTED"
    DROPPED_CONFLICT = "DROPPED_CONFLICT"
    DROPPED_DUPLICATE = "DROPPED_DUPLICATE"
    DROPPED_SUPERSEDED_BY_EXIT = "DROPPED_SUPERSEDED_BY_EXIT"
    REDUCED_TO_EXISTING = "REDUCED_TO_EXISTING"


@dataclass
class ArbitrationDecision:
    intent: OrderIntent | None
    outcome: ArbitrationOutcome
    detail: str = ""
    contributing_strategies: list[str] = field(default_factory=list)


@dataclass
class ArbitrationResult:
    accepted: list[OrderIntent] = field(default_factory=list)
    decisions: list[ArbitrationDecision] = field(default_factory=list)

    @property
    def dropped(self) -> list[ArbitrationDecision]:
        return [d for d in self.decisions if d.outcome is not ArbitrationOutcome.ACCEPTED]

    def summary(self) -> str:
        counts: dict[str, int] = defaultdict(int)
        for d in self.decisions:
            counts[str(d.outcome)] += 1
        return ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))


class SignalArbitrator:
    """Resolves competing proposals for the same instrument."""

    def __init__(self, *, allow_position_reversal: bool = False) -> None:
        # Reversing a position in a single order (long 100 -> short 50 via
        # a sell of 150) doubles the effective trade size and is rarely
        # what a strategy actually intended. Off by default; when off, a
        # reversal is trimmed to a flatten.
        self._allow_reversal = allow_position_reversal

    def arbitrate(
        self,
        proposals: list[tuple[Signal | None, OrderIntent]],
        *,
        positions: dict[str, Position] | None = None,
    ) -> ArbitrationResult:
        result = ArbitrationResult()
        positions = positions or {}

        by_instrument: dict[str, list[tuple[Signal | None, OrderIntent]]] = defaultdict(list)
        for signal, intent in proposals:
            by_instrument[str(intent.instrument)].append((signal, intent))

        for key, group in by_instrument.items():
            position = positions.get(key)
            for decision in self._resolve_group(key, group, position):
                result.decisions.append(decision)
                if decision.intent is not None:
                    result.accepted.append(decision.intent)

        if result.dropped:
            log.info("arbitration.resolved", summary=result.summary())
        return result

    def _resolve_group(
        self,
        key: str,
        group: list[tuple[Signal | None, OrderIntent]],
        position: Position | None,
    ) -> list[ArbitrationDecision]:
        if len(group) == 1:
            signal, intent = group[0]
            return [self._finalise(intent, signal, position)]

        # 1. Exits take absolute priority.
        exits = [
            (s, i)
            for s, i in group
            if s is not None and s.direction is SignalDirection.FLAT
        ]
        if exits:
            signal, intent = exits[0]
            decisions = [
                ArbitrationDecision(
                    intent=intent,
                    outcome=ArbitrationOutcome.ACCEPTED,
                    detail="exit takes priority over entries",
                    contributing_strategies=[signal.strategy] if signal else [],
                )
            ]
            for other_signal, other in group:
                if other is intent:
                    continue
                decisions.append(
                    ArbitrationDecision(
                        intent=None,
                        outcome=ArbitrationOutcome.DROPPED_SUPERSEDED_BY_EXIT,
                        detail=f"{key}: superseded by an exit proposal",
                        contributing_strategies=(
                            [other_signal.strategy] if other_signal else [other.source]
                        ),
                    )
                )
            return decisions

        # 2. Direct conflict -> no trade.
        sides = {i.side for _, i in group}
        if len(sides) > 1:
            strategies = [s.strategy if s else i.source for s, i in group]
            log.info(
                "arbitration.conflict",
                instrument=key,
                strategies=strategies,
                detail="opposing directions; taking no action",
            )
            return [
                ArbitrationDecision(
                    intent=None,
                    outcome=ArbitrationOutcome.DROPPED_CONFLICT,
                    detail=(
                        f"{key}: strategies disagree on direction ({strategies}); "
                        "mixed evidence resolves to no trade"
                    ),
                    contributing_strategies=strategies,
                )
                for _ in group
            ]

        # 3. Agreement -> one order, at the smallest proposed size.
        # Agreement is not a reason to double the position; the smallest
        # size is the one all contributors would accept.
        chosen_signal, chosen = min(group, key=lambda pair: pair[1].quantity)
        contributors = [s.strategy if s else i.source for s, i in group]
        decisions = [
            self._finalise(
                chosen,
                chosen_signal,
                position,
                detail=f"{len(group)} strategies agree; using smallest proposed size",
                contributors=contributors,
            )
        ]
        for other_signal, other in group:
            if other is chosen:
                continue
            decisions.append(
                ArbitrationDecision(
                    intent=None,
                    outcome=ArbitrationOutcome.DROPPED_DUPLICATE,
                    detail=f"{key}: same direction as an accepted proposal",
                    contributing_strategies=(
                        [other_signal.strategy] if other_signal else [other.source]
                    ),
                )
            )
        return decisions

    def _finalise(
        self,
        intent: OrderIntent,
        signal: Signal | None,
        position: Position | None,
        *,
        detail: str = "",
        contributors: list[str] | None = None,
    ) -> ArbitrationDecision:
        """Apply the reversal guard to a single accepted proposal."""
        contributors = contributors or ([signal.strategy] if signal else [intent.source])

        if (
            not self._allow_reversal
            and position is not None
            and not position.is_flat
        ):
            opposing = (intent.side is OrderSide.SELL and position.is_long) or (
                intent.side is OrderSide.BUY and position.is_short
            )
            existing = abs(position.quantity)
            if opposing and intent.quantity > existing:
                trimmed = intent.model_copy(update={"quantity": existing})
                return ArbitrationDecision(
                    intent=trimmed,
                    outcome=ArbitrationOutcome.REDUCED_TO_EXISTING,
                    detail=(
                        f"trimmed {intent.quantity} -> {existing} to flatten rather than "
                        "reverse the position"
                    ),
                    contributing_strategies=contributors,
                )

        return ArbitrationDecision(
            intent=intent,
            outcome=ArbitrationOutcome.ACCEPTED,
            detail=detail,
            contributing_strategies=contributors,
        )
