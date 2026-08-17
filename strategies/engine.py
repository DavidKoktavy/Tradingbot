"""
Strategy engine: runs a set of strategies over a context and collects
their signals and proposed intents.

Design decisions:

- A failing strategy is isolated. If one strategy raises, it is logged,
  disabled for the cycle, and the others continue. A bug in an
  experimental strategy must not take down the trading process — but the
  failure is surfaced, not swallowed silently, and repeated failures
  disable the strategy entirely rather than retrying forever.

- The engine produces intents; it does not evaluate or submit them. The
  caller passes each intent to the risk engine. This keeps the engine
  free of any downstream dependency and makes it trivially testable.
"""

from __future__ import annotations

from collections import defaultdict

import structlog

from execution.execution_models import OrderIntent
from strategies.base import Signal, Strategy, StrategyContext

log = structlog.get_logger(__name__)


class StrategyEngine:
    def __init__(
        self, strategies: list[Strategy], *, max_consecutive_failures: int = 3
    ) -> None:
        self._strategies = list(strategies)
        self._failures: dict[str, int] = defaultdict(int)
        self._disabled: set[str] = set()
        self._max_failures = max_consecutive_failures

    @property
    def active_strategies(self) -> list[Strategy]:
        return [s for s in self._strategies if s.name not in self._disabled]

    @property
    def disabled_strategies(self) -> set[str]:
        return set(self._disabled)

    def evaluate(self, context: StrategyContext) -> list[tuple[Signal, OrderIntent | None]]:
        """Run every active strategy. Returns (signal, intent) pairs;
        intent is None when the signal isn't actionable."""
        results: list[tuple[Signal, OrderIntent | None]] = []

        for strategy in self.active_strategies:
            try:
                signal = strategy.generate_signal(context)
                intent = (
                    strategy.generate_order_intent(signal, context)
                    if signal.is_actionable
                    else None
                )
                self._failures[strategy.name] = 0
                results.append((signal, intent))

                if signal.is_actionable:
                    log.info(
                        "strategy.signal",
                        strategy=strategy.name,
                        instrument=str(context.instrument),
                        direction=signal.direction,
                        strength=round(signal.strength, 3),
                        rationale=signal.rationale,
                        has_intent=intent is not None,
                    )
            except Exception as exc:  # noqa: BLE001 — isolate strategy faults
                self._failures[strategy.name] += 1
                log.error(
                    "strategy.failed",
                    strategy=strategy.name,
                    error=str(exc),
                    consecutive_failures=self._failures[strategy.name],
                    exc_info=True,
                )
                if self._failures[strategy.name] >= self._max_failures:
                    self._disabled.add(strategy.name)
                    log.error(
                        "strategy.disabled",
                        strategy=strategy.name,
                        reason=f"{self._max_failures} consecutive failures",
                    )

        return results

    def reset_strategy(self, name: str) -> None:
        """Re-enable a disabled strategy. Explicit operator action."""
        self._disabled.discard(name)
        self._failures[name] = 0
