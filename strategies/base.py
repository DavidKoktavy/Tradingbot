"""
Strategy framework.

Design decisions:

- A `Strategy` receives a `StrategyContext` (bars, snapshot, and its own
  current position) and returns a `Signal`. It is handed **no broker, no
  order store, no risk engine, and no portfolio mutation methods**. The
  only object it can construct that touches trading is an `OrderIntent`,
  which is inert until the risk engine approves it. A strategy physically
  cannot submit an order — there is no method to call.

- Strategies are given a read-only view of their *own* position only, not
  the whole portfolio. Cross-position decisions are a portfolio-level
  concern; letting each strategy see and reason about the whole book makes
  their behaviour interdependent and impossible to backtest in isolation.

- `Signal` is separate from `OrderIntent` deliberately. A signal is an
  opinion ("this looks bullish, conviction 0.7"); an intent is a concrete
  proposed trade. Keeping them apart means the AI layer and the sizing
  logic can consume signals without every strategy having to guess at
  position sizes — the risk engine owns sizing.

- Strategies must be deterministic and stateless across calls where
  possible. Any internal state must be reconstructible from the bars
  supplied, otherwise backtests won't reproduce and a restart mid-session
  changes behaviour.

- **No strategy here is claimed to be profitable.** These are framework
  demonstrations using well-known textbook constructions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from data.models import Bar, Instrument, MarketSnapshot
from execution.execution_models import OrderIntent, OrderSide, OrderType
from portfolio.positions import Position


class SignalDirection(StrEnum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"  # close any existing position
    NONE = "NONE"  # no opinion; do nothing


class Signal(BaseModel):
    """A strategy's opinion. Not a trade."""

    model_config = {"frozen": True}

    signal_id: str = Field(default_factory=lambda: __import__("uuid").uuid4().hex)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    instrument: Instrument
    direction: SignalDirection
    strength: float = Field(default=0.0, ge=0.0, le=1.0)
    strategy: str = ""
    features: dict[str, float] = Field(default_factory=dict)
    rationale: str = ""

    @property
    def is_actionable(self) -> bool:
        return self.direction is not SignalDirection.NONE


class StrategyContext(BaseModel):
    """Everything a strategy is allowed to see. Note what is absent:
    the broker, the order store, other strategies' positions, and any
    method that could mutate state."""

    model_config = {"arbitrary_types_allowed": True}

    instrument: Instrument
    bars: list[Bar]
    snapshot: MarketSnapshot | None = None
    position: Position | None = None
    equity: Decimal = Decimal("0")

    @property
    def closes(self) -> list[float]:
        return [b.close for b in self.bars]

    @property
    def last_close(self) -> float | None:
        return self.bars[-1].close if self.bars else None

    @property
    def current_quantity(self) -> Decimal:
        return self.position.quantity if self.position else Decimal("0")

    @property
    def is_long(self) -> bool:
        return self.current_quantity > 0

    @property
    def is_short(self) -> bool:
        return self.current_quantity < 0

    @property
    def is_flat(self) -> bool:
        return self.current_quantity == 0


class StrategyParams(BaseModel):
    """Base for strategy parameters. Subclasses add typed fields, which
    means a bad parameter set fails at construction rather than producing
    silently wrong signals."""

    model_config = {"extra": "forbid"}


class Strategy(ABC):
    """Base strategy. Subclasses implement `calculate_features` and
    `generate_signal`; `generate_order_intent` has a sensible default."""

    name: str = "unnamed"
    version: str = "0.1.0"

    def __init__(self, params: StrategyParams | None = None) -> None:
        self.params = params or StrategyParams()

    @property
    @abstractmethod
    def min_bars(self) -> int:
        """Minimum bars required before this strategy can emit a signal.
        The engine checks this rather than letting strategies emit
        garbage from insufficient history."""

    @abstractmethod
    def calculate_features(self, context: StrategyContext) -> dict[str, float]:
        """Compute the numeric features this strategy reasons over.
        Exposed separately so they can be logged for the audit trail and
        fed to the AI layer for analysis."""

    @abstractmethod
    def generate_signal(self, context: StrategyContext) -> Signal:
        """Produce a signal from the context. Must be deterministic."""

    def generate_order_intent(
        self, signal: Signal, context: StrategyContext
    ) -> OrderIntent | None:
        """Translate a signal into a proposed trade.

        Default behaviour: propose a market order with an ATR-derived
        protective stop, requesting a nominal quantity. The requested
        quantity is intentionally a *ceiling request*, not a decision —
        the risk engine's position sizer determines the actual size and
        will only ever reduce it.
        """
        if not signal.is_actionable:
            return None
        if context.snapshot is None or context.snapshot.mid is None:
            return None

        price = Decimal(str(context.snapshot.mid))
        atr_value = signal.features.get("atr")

        if signal.direction is SignalDirection.FLAT:
            if context.is_flat:
                return None
            side = OrderSide.SELL if context.is_long else OrderSide.BUY
            return OrderIntent(
                instrument=context.instrument,
                side=side,
                quantity=abs(context.current_quantity),
                order_type=OrderType.MARKET,
                source=self.name,
                strategy=self.name,
                signal_id=signal.signal_id,
            )

        side = OrderSide.BUY if signal.direction is SignalDirection.LONG else OrderSide.SELL

        # Don't stack a new entry on top of an existing same-side position.
        if (side is OrderSide.BUY and context.is_long) or (
            side is OrderSide.SELL and context.is_short
        ):
            return None

        stop_loss = self._derive_stop(price, side, atr_value)
        take_profit = self._derive_target(price, side, atr_value)

        return OrderIntent(
            instrument=context.instrument,
            side=side,
            quantity=self._requested_quantity(context, price),
            order_type=OrderType.MARKET,
            stop_loss=stop_loss,
            take_profit=take_profit,
            source=self.name,
            strategy=self.name,
            signal_id=signal.signal_id,
        )

    # ---- helpers ----------------------------------------------------------

    def _derive_stop(
        self, price: Decimal, side: OrderSide, atr_value: float | None, multiple: float = 2.0
    ) -> Decimal | None:
        if not atr_value or atr_value <= 0:
            return None
        distance = Decimal(str(atr_value * multiple))
        stop = price - distance if side is OrderSide.BUY else price + distance
        return stop.quantize(Decimal("0.01")) if stop > 0 else None

    def _derive_target(
        self, price: Decimal, side: OrderSide, atr_value: float | None, multiple: float = 4.0
    ) -> Decimal | None:
        if not atr_value or atr_value <= 0:
            return None
        distance = Decimal(str(atr_value * multiple))
        target = price + distance if side is OrderSide.BUY else price - distance
        return target.quantize(Decimal("0.01")) if target > 0 else None

    def _requested_quantity(self, context: StrategyContext, price: Decimal) -> Decimal:
        """A nominal request. The risk engine sizes the trade; this is
        only an upper bound expressing 'I'd take up to this much'."""
        if context.equity <= 0 or price <= 0:
            return Decimal("1")
        nominal = (context.equity * Decimal("0.10")) / price
        return max(Decimal("1"), nominal.quantize(Decimal("1")))

    def has_enough_history(self, context: StrategyContext) -> bool:
        return len(context.bars) >= self.min_bars

    def no_signal(self, context: StrategyContext, reason: str) -> Signal:
        return Signal(
            instrument=context.instrument,
            direction=SignalDirection.NONE,
            strategy=self.name,
            rationale=reason,
        )
