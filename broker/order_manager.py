"""
Order manager: the single path from an approved `Order` to a broker.

Design decisions:

- **There is exactly one submission method in the system**, and it calls
  `ModeGate.assert_can_submit()` before doing anything else. Having one
  path is what makes the gate reliable: a gate that must be remembered in
  five places will eventually be forgotten in one.

- **`submit()` accepts only an `Order` in APPROVED state.** An order in
  any other state is a programming error — it means something built an
  order without going through `OrderValidator.build_order()`, which is the
  only constructor that can produce APPROVED. Raises rather than
  submitting.

- **State transitions follow the broker, not our hopes.** We move to
  SUBMITTED when the API accepts the request, and only to ACKNOWLEDGED /
  FILLED when the broker says so. A submission that raises leaves the
  order in ERROR with the exception recorded, never silently retried:
  retrying an order whose fate is unknown is how duplicate positions
  happen.

- **Cancellation is requested, not asserted.** `cancel()` moves to
  CANCEL_REQUESTED; only a broker confirmation moves it to CANCELLED. A
  cancel can lose the race with a fill.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any, Protocol

import structlog

from app.mode_gate import ModeGate
from execution.execution_models import (
    Order,
    OrderSide,
    OrderState,
    OrderType,
    TimeInForce,
)
from execution.order_store import OrderStore

log = structlog.get_logger(__name__)


class OrderSubmissionError(Exception):
    """Broker rejected or failed to accept the order."""


class BrokerOrderGateway(ABC):
    """What the order manager needs from a broker. Implemented by the
    IBKR gateway and by the simulated broker, so PAPER and SIMULATION
    exercise identical order-management code."""

    @abstractmethod
    async def place(self, order: Order) -> str:
        """Submit and return the broker's order id."""

    @abstractmethod
    async def cancel(self, broker_order_id: str) -> None: ...

    @abstractmethod
    async def open_orders(self) -> list[Any]: ...

    @abstractmethod
    async def positions(self) -> list[Any]: ...


class IBLike(Protocol):
    def placeOrder(self, contract: Any, order: Any) -> Any: ...
    def cancelOrder(self, order: Any) -> None: ...
    async def reqOpenOrdersAsync(self) -> list[Any]: ...
    async def reqPositionsAsync(self) -> list[Any]: ...
    async def qualifyContractsAsync(self, *contracts: Any) -> list[Any]: ...


def _to_ib_order(order: Order) -> Any:
    """Translate our normalised order into an ib_async order object."""
    from ib_async import LimitOrder, MarketOrder, Order as IBOrder, StopOrder

    intent = order.intent
    action = "BUY" if intent.side is OrderSide.BUY else "SELL"
    quantity = float(intent.quantity)
    tif = intent.time_in_force.value

    if intent.order_type is OrderType.MARKET:
        ib_order = MarketOrder(action, quantity)
    elif intent.order_type is OrderType.LIMIT:
        ib_order = LimitOrder(action, quantity, float(intent.limit_price))
    elif intent.order_type is OrderType.STOP:
        ib_order = StopOrder(action, quantity, float(intent.stop_price))
    elif intent.order_type is OrderType.STOP_LIMIT:
        ib_order = IBOrder(
            action=action,
            totalQuantity=quantity,
            orderType="STP LMT",
            lmtPrice=float(intent.limit_price),
            auxPrice=float(intent.stop_price),
        )
    else:  # pragma: no cover - enum is exhaustive
        raise OrderSubmissionError(f"Unsupported order type {intent.order_type}")

    ib_order.tif = tif
    # Our own id travels with the order so broker callbacks can be matched
    # back to local state even across a reconnect.
    ib_order.orderRef = order.order_id
    return ib_order


class IBKROrderGateway(BrokerOrderGateway):
    """Real IBKR gateway. The ib_async calls are isolated here so every
    layer above is testable against the simulated gateway."""

    def __init__(self, ib: IBLike) -> None:
        self._ib = ib
        self._trades: dict[str, Any] = {}

    async def place(self, order: Order) -> str:
        from ib_async import Stock

        instrument = order.intent.instrument
        contract = Stock(instrument.symbol, instrument.exchange, instrument.currency)
        qualified = await self._ib.qualifyContractsAsync(contract)
        if not qualified:
            raise OrderSubmissionError(f"Could not qualify contract for {instrument}")

        trade = self._ib.placeOrder(qualified[0], _to_ib_order(order))
        broker_id = str(getattr(trade.order, "orderId", "") or "")
        if not broker_id:
            raise OrderSubmissionError("Broker returned no order id")
        self._trades[broker_id] = trade
        return broker_id

    async def cancel(self, broker_order_id: str) -> None:
        trade = self._trades.get(broker_order_id)
        if trade is None:
            raise OrderSubmissionError(f"No known broker order {broker_order_id}")
        self._ib.cancelOrder(trade.order)

    async def open_orders(self) -> list[Any]:
        return await self._ib.reqOpenOrdersAsync()

    async def positions(self) -> list[Any]:
        return await self._ib.reqPositionsAsync()


class OrderManager:
    def __init__(
        self,
        gateway: BrokerOrderGateway,
        store: OrderStore,
        mode_gate: ModeGate,
    ) -> None:
        self._gateway = gateway
        self._store = store
        self._mode = mode_gate

    async def submit(self, order: Order) -> Order:
        """The one and only submission path.

        Raises `LiveTradingNotAuthorised` before touching the broker if
        the mode gate does not permit submission.
        """
        # 1. Mode gate FIRST, before any broker interaction.
        self._mode.assert_can_submit()

        # 2. Only orders that came through OrderValidator.build_order()
        #    can be in APPROVED state.
        if order.state is not OrderState.APPROVED:
            raise OrderSubmissionError(
                f"Refusing to submit order {order.order_id} in state {order.state}; "
                "only APPROVED orders may be submitted"
            )

        try:
            broker_id = await self._gateway.place(order)
        except Exception as exc:  # noqa: BLE001
            order.transition_to(OrderState.ERROR, error_message=str(exc))
            log.error(
                "order.submission_failed",
                order_id=order.order_id,
                error=str(exc),
                mode=self._mode.mode,
            )
            # Deliberately not retried: the order's fate at the broker is
            # unknown, and a blind retry is how duplicate positions occur.
            # Reconciliation resolves it.
            raise OrderSubmissionError(str(exc)) from exc

        self._store.link_broker_id(order.order_id, broker_id)
        order.transition_to(OrderState.SUBMITTED)
        log.info(
            "order.submitted",
            order_id=order.order_id,
            broker_order_id=broker_id,
            mode=self._mode.mode,
            instrument=str(order.intent.instrument),
            side=order.intent.side,
            quantity=str(order.intent.quantity),
            source=order.intent.source,
        )
        return order

    async def cancel(self, order: Order) -> Order:
        """Request cancellation. The order is not CANCELLED until the
        broker confirms — a cancel can lose the race with a fill."""
        if order.is_terminal:
            return order
        if not order.broker_order_id:
            order.transition_to(OrderState.CANCELLED)
            return order

        order.transition_to(OrderState.CANCEL_REQUESTED)
        try:
            await self._gateway.cancel(order.broker_order_id)
        except Exception as exc:  # noqa: BLE001
            log.error("order.cancel_failed", order_id=order.order_id, error=str(exc))
            # Stay in CANCEL_REQUESTED: we genuinely do not know its state.
            return order
        log.info("order.cancel_requested", order_id=order.order_id)
        return order

    async def cancel_all(self) -> int:
        """Cancel every working order. Used by the kill switch."""
        cancelled = 0
        for order in self._store.active_orders():
            await self.cancel(order)
            cancelled += 1
        log.warning("order.cancel_all", count=cancelled)
        return cancelled
