"""
Simulated broker gateway.

Used in SIMULATION mode and throughout the test suite. Implements the
same `BrokerOrderGateway` interface as the IBKR gateway, so order
management, reconciliation, the control loop, and the kill switch all
exercise identical code paths regardless of mode.

Design decisions:

- Fills are **not** instantaneous or guaranteed. The simulator supports
  configurable rejection, partial fills, and latency, because the failure
  paths are the ones worth rehearsing before going anywhere near real
  money. A simulator that always fills perfectly trains the system (and
  the operator) on a world that doesn't exist.

- It reuses the Phase 6 `CostModel`, so simulated fills carry the same
  spread and slippage assumptions as backtests. Divergence between
  simulation and backtest cost assumptions would make paper results
  incomparable to backtest results.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import structlog

from backtesting.costs import CostModel
from broker.order_manager import BrokerOrderGateway, OrderSubmissionError
from data.models import MarketSnapshot
from execution.execution_models import Fill, Order, OrderState
from execution.reconciliation import BrokerOrder, BrokerPosition

log = structlog.get_logger(__name__)


@dataclass
class SimulationConfig:
    """Knobs for rehearsing failure. Defaults are benign; tests and
    operators turn on the nasty ones deliberately."""

    reject_probability: float = 0.0
    partial_fill_probability: float = 0.0
    partial_fill_fraction: Decimal = Decimal("0.5")
    fill_latency_seconds: float = 0.0
    reject_next_n: int = 0
    fail_submission_next_n: int = 0


@dataclass
class SimulatedBrokerGateway(BrokerOrderGateway):
    """In-memory broker. Deterministic: 'probabilities' are driven by a
    counter, not an RNG, so simulation runs are reproducible."""

    cost_model: CostModel = field(default_factory=CostModel)
    config: SimulationConfig = field(default_factory=SimulationConfig)

    _orders: dict[str, Order] = field(default_factory=dict)
    _snapshots: dict[str, MarketSnapshot] = field(default_factory=dict)
    _positions: dict[str, BrokerPosition] = field(default_factory=dict)
    _counter: int = 0
    _next_id: int = 1000

    def set_snapshot(self, snapshot: MarketSnapshot) -> None:
        self._snapshots[str(snapshot.instrument)] = snapshot

    async def place(self, order: Order) -> str:
        if self.config.fail_submission_next_n > 0:
            self.config.fail_submission_next_n -= 1
            raise OrderSubmissionError("Simulated submission failure")

        self._next_id += 1
        broker_id = f"SIM-{self._next_id}"
        self._orders[broker_id] = order
        log.debug("sim.order_placed", broker_order_id=broker_id)
        return broker_id

    async def cancel(self, broker_order_id: str) -> None:
        order = self._orders.get(broker_order_id)
        if order is None:
            raise OrderSubmissionError(f"Unknown simulated order {broker_order_id}")
        if not order.is_terminal:
            order.transition_to(OrderState.CANCELLED)

    async def open_orders(self) -> list[BrokerOrder]:
        return [
            BrokerOrder(
                broker_order_id=bid,
                instrument=o.intent.instrument,
                quantity=o.intent.quantity,
                filled_quantity=o.filled_quantity,
                state=o.state,
            )
            for bid, o in self._orders.items()
            if o.is_active
        ]

    async def positions(self) -> list[BrokerPosition]:
        return list(self._positions.values())

    def set_position(self, position: BrokerPosition) -> None:
        self._positions[str(position.instrument)] = position

    # ---- fill simulation ----------------------------------------------------

    async def try_fill(self, broker_order_id: str) -> Fill | None:
        """Attempt to fill a resting order against the current snapshot.

        Returns a Fill, or None if the order was rejected or cannot be
        priced. Called by the execution listener in SIMULATION mode.
        """
        order = self._orders.get(broker_order_id)
        if order is None or order.is_terminal:
            return None

        snapshot = self._snapshots.get(str(order.intent.instrument))
        if snapshot is None or snapshot.mid is None:
            return None  # cannot price: no fill, no guess

        self._counter += 1

        if self.config.reject_next_n > 0:
            self.config.reject_next_n -= 1
            order.transition_to(OrderState.REJECTED, error_message="Simulated rejection")
            log.info("sim.order_rejected", broker_order_id=broker_order_id)
            return None

        if self.config.reject_probability > 0:
            period = max(1, int(1 / self.config.reject_probability))
            if self._counter % period == 0:
                order.transition_to(OrderState.REJECTED, error_message="Simulated rejection")
                return None

        if self.config.fill_latency_seconds > 0:
            await asyncio.sleep(self.config.fill_latency_seconds)

        quantity = order.remaining_quantity
        if self.config.partial_fill_probability > 0:
            period = max(1, int(1 / self.config.partial_fill_probability))
            if self._counter % period == 0:
                quantity = (quantity * self.config.partial_fill_fraction).quantize(
                    Decimal("1")
                )
                quantity = max(quantity, Decimal("1"))

        price = self.cost_model.fill_price(
            reference_price=Decimal(str(snapshot.mid)),
            side=order.intent.side,
            quantity=quantity,
            bid=Decimal(str(snapshot.bid)) if snapshot.bid else None,
            ask=Decimal(str(snapshot.ask)) if snapshot.ask else None,
        )
        commission = self.cost_model.commission(quantity, price)

        fill = Fill(
            fill_id=f"simfill-{self._counter}",
            order_id=order.order_id,
            timestamp=snapshot.timestamp,
            quantity=quantity,
            price=price,
            commission=commission,
        )
        if order.state is OrderState.SUBMITTED:
            order.transition_to(OrderState.ACKNOWLEDGED)
        order.apply_fill(fill)

        # Keep simulated broker positions consistent so reconciliation
        # against this gateway behaves like it would against IBKR.
        key = str(order.intent.instrument)
        existing = self._positions.get(key)
        signed = quantity if order.intent.side.value == "BUY" else -quantity
        if existing is None:
            self._positions[key] = BrokerPosition(
                instrument=order.intent.instrument, quantity=signed, average_cost=price
            )
        else:
            new_qty = existing.quantity + signed
            self._positions[key] = BrokerPosition(
                instrument=order.intent.instrument,
                quantity=new_qty,
                average_cost=price if new_qty != 0 else Decimal("0"),
            )
        return fill

    async def fill_all_pending(self) -> list[Fill]:
        fills = []
        for broker_id in list(self._orders):
            fill = await self.try_fill(broker_id)
            if fill is not None:
                fills.append(fill)
        return fills
