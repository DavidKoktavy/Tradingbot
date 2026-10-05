"""
Tests for bracket order submission.

The bug these cover: `stop_loss` and `take_profit` were carried on every
OrderIntent and used by the position sizer to compute risk-per-trade, but
they were never translated into orders. The risk engine therefore sized
positions on the assumption of a bounded loss that nothing at the broker
would enforce. If the process died — crash, reboot, closed laptop — the
position sat unprotected while every record in the system said otherwise.

These assert the IBKR bracket contract specifically, because getting it
subtly wrong is worse than not sending it: a child without `parentId` is a
naked order, and children without a shared OCA group leave a working stop
behind after the take-profit fills, which would open a position in the
opposite direction with nothing to close it.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from broker.order_manager import (
    IBKROrderGateway,
    OrderSubmissionError,
    _to_ib_bracket,
)
from data.models import Instrument
from execution.execution_models import (
    Order,
    OrderIntent,
    OrderSide,
    OrderState,
    OrderType,
    TimeInForce,
)

AAPL = Instrument(symbol="AAPL")


def _intent(**kw: Any) -> OrderIntent:
    defaults: dict[str, Any] = dict(
        instrument=AAPL,
        side=OrderSide.BUY,
        quantity=Decimal("100"),
        order_type=OrderType.MARKET,
        source="momentum",
    )
    defaults.update(kw)
    return OrderIntent(**defaults)


def _order(**kw: Any) -> Order:
    order = Order(intent=_intent(**kw))
    order.transition_to(OrderState.VALIDATING)
    order.transition_to(OrderState.APPROVED)
    return order


class _Counter:
    """Monotonic id allocator, standing in for ib_async's getReqId()."""

    def __init__(self, start: int = 900) -> None:
        self._next = start

    def __call__(self) -> int:
        self._next += 1
        return self._next


class _FakeTrade:
    def __init__(self, order: Any) -> None:
        self.order = order


class _FakeClient:
    def __init__(self, start: int = 900) -> None:
        self._counter = _Counter(start)

    def getReqId(self) -> int:
        return self._counter()


class _FakeIB:
    """Minimal IBLike with order placement and id allocation."""

    def __init__(self, *, with_client: bool = True, start: int = 900) -> None:
        self.placed: list[Any] = []
        self.cancelled: list[Any] = []
        self.client = _FakeClient(start) if with_client else None
        self._next_auto = 500

    async def qualifyContractsAsync(self, *contracts: Any) -> list[Any]:
        return list(contracts)

    def placeOrder(self, contract: Any, order: Any) -> Any:
        # Mirror IBKR: an order submitted without an id is assigned one.
        if not getattr(order, "orderId", 0):
            self._next_auto += 1
            order.orderId = self._next_auto
        self.placed.append(order)
        return _FakeTrade(order)

    def cancelOrder(self, order: Any) -> None:
        self.cancelled.append(order)

    async def reqOpenOrdersAsync(self) -> list[Any]:
        return []

    async def reqPositionsAsync(self) -> list[Any]:
        return []


# ---- _to_ib_bracket ------------------------------------------------------


def test_bracket_has_parent_and_both_children():
    order = _order(stop_loss=Decimal("95"), take_profit=Decimal("110"))
    orders = _to_ib_bracket(order, _Counter())
    assert len(orders) == 3


def test_children_reference_parent_order_id():
    order = _order(stop_loss=Decimal("95"), take_profit=Decimal("110"))
    parent, *children = _to_ib_bracket(order, _Counter())
    assert parent.orderId
    for child in children:
        assert child.parentId == parent.orderId


def test_only_the_last_order_transmits():
    """The whole point of transmit=False: no window in which the entry is
    live at the broker without its protective exits attached."""
    order = _order(stop_loss=Decimal("95"), take_profit=Decimal("110"))
    orders = _to_ib_bracket(order, _Counter())
    assert [o.transmit for o in orders] == [False, False, True]


def test_children_share_one_oca_group():
    order = _order(stop_loss=Decimal("95"), take_profit=Decimal("110"))
    _parent, *children = _to_ib_bracket(order, _Counter())
    groups = {c.ocaGroup for c in children}
    assert len(groups) == 1
    assert all(c.ocaType == 1 for c in children)


def test_oca_group_is_unique_per_order():
    a = _order(stop_loss=Decimal("95"), take_profit=Decimal("110"))
    b = _order(stop_loss=Decimal("95"), take_profit=Decimal("110"))
    _, *ca = _to_ib_bracket(a, _Counter())
    _, *cb = _to_ib_bracket(b, _Counter())
    assert ca[0].ocaGroup != cb[0].ocaGroup


def test_buy_entry_produces_sell_exits():
    order = _order(side=OrderSide.BUY, stop_loss=Decimal("95"), take_profit=Decimal("110"))
    _parent, *children = _to_ib_bracket(order, _Counter())
    assert all(c.action == "SELL" for c in children)


def test_sell_entry_produces_buy_exits():
    order = _order(
        side=OrderSide.SELL, stop_loss=Decimal("110"), take_profit=Decimal("95")
    )
    _parent, *children = _to_ib_bracket(order, _Counter())
    assert all(c.action == "BUY" for c in children)


def test_exit_quantity_matches_entry_quantity():
    """A partial exit would leave a residual position the risk model does
    not know about."""
    order = _order(
        quantity=Decimal("37"), stop_loss=Decimal("95"), take_profit=Decimal("110")
    )
    orders = _to_ib_bracket(order, _Counter())
    assert all(float(o.totalQuantity) == 37.0 for o in orders)


def test_stop_child_uses_stop_price():
    order = _order(stop_loss=Decimal("95.5"), take_profit=Decimal("110"))
    _parent, *children = _to_ib_bracket(order, _Counter())
    stop = next(c for c in children if c.orderType == "STP")
    assert stop.auxPrice == pytest.approx(95.5)


def test_take_profit_child_uses_limit_price():
    order = _order(stop_loss=Decimal("95"), take_profit=Decimal("110.25"))
    _parent, *children = _to_ib_bracket(order, _Counter())
    limit = next(c for c in children if c.orderType == "LMT")
    assert limit.lmtPrice == pytest.approx(110.25)


def test_stop_only_produces_two_orders():
    order = _order(stop_loss=Decimal("95"))
    orders = _to_ib_bracket(order, _Counter())
    assert len(orders) == 2
    assert orders[-1].transmit is True
    assert orders[1].orderType == "STP"


def test_take_profit_only_produces_two_orders():
    order = _order(take_profit=Decimal("110"))
    orders = _to_ib_bracket(order, _Counter())
    assert len(orders) == 2
    assert orders[1].orderType == "LMT"


def test_no_exits_produces_a_single_transmitting_parent():
    order = _order()
    orders = _to_ib_bracket(order, _Counter())
    assert len(orders) == 1
    assert orders[0].transmit is True


def test_children_carry_our_order_ref_for_reconciliation():
    """Broker callbacks must be matchable to local state across a
    reconnect, including for the children."""
    order = _order(stop_loss=Decimal("95"), take_profit=Decimal("110"))
    for ib_order in _to_ib_bracket(order, _Counter()):
        assert ib_order.orderRef == order.order_id


def test_children_inherit_time_in_force():
    order = _order(
        stop_loss=Decimal("95"),
        take_profit=Decimal("110"),
        time_in_force=TimeInForce.GTC,
    )
    for ib_order in _to_ib_bracket(order, _Counter()):
        assert ib_order.tif == "GTC"


def test_order_ids_are_distinct():
    order = _order(stop_loss=Decimal("95"), take_profit=Decimal("110"))
    orders = _to_ib_bracket(order, _Counter())
    ids = [o.orderId for o in orders]
    assert len(set(ids)) == len(ids)


# ---- IBKROrderGateway ----------------------------------------------------


async def test_gateway_places_three_orders_for_a_bracket():
    ib = _FakeIB()
    gateway = IBKROrderGateway(ib)
    order = _order(stop_loss=Decimal("95"), take_profit=Decimal("110"))
    await gateway.place(order)
    assert len(ib.placed) == 3


async def test_gateway_returns_the_parent_broker_id():
    ib = _FakeIB()
    gateway = IBKROrderGateway(ib)
    order = _order(stop_loss=Decimal("95"), take_profit=Decimal("110"))
    broker_id = await gateway.place(order)
    assert broker_id == str(ib.placed[0].orderId)


async def test_gateway_places_one_order_when_there_are_no_exits():
    ib = _FakeIB()
    gateway = IBKROrderGateway(ib)
    await gateway.place(_order())
    assert len(ib.placed) == 1


async def test_brackets_can_be_disabled():
    """Kept as an escape hatch: an account or venue that rejects brackets
    must still be able to trade, explicitly and visibly."""
    ib = _FakeIB()
    gateway = IBKROrderGateway(ib, use_brackets=False)
    order = _order(stop_loss=Decimal("95"), take_profit=Decimal("110"))
    await gateway.place(order)
    assert len(ib.placed) == 1


async def test_cancelling_the_parent_cancels_the_children():
    """A surviving stop after its parent is gone would, on trigger, open a
    position in the opposite direction with nothing to close it."""
    ib = _FakeIB()
    gateway = IBKROrderGateway(ib)
    order = _order(stop_loss=Decimal("95"), take_profit=Decimal("110"))
    broker_id = await gateway.place(order)
    await gateway.cancel(broker_id)
    assert len(ib.cancelled) == 3


async def test_cancelling_an_unknown_order_raises():
    gateway = IBKROrderGateway(_FakeIB())
    with pytest.raises(OrderSubmissionError):
        await gateway.cancel("does-not-exist")


async def test_bracket_without_an_id_source_fails_loudly():
    """Fail closed: inventing our own ids would collide with the client's
    sequence and get orders silently rejected by IBKR."""
    ib = _FakeIB(with_client=False)
    gateway = IBKROrderGateway(ib)
    order = _order(stop_loss=Decimal("95"), take_profit=Decimal("110"))
    with pytest.raises(OrderSubmissionError):
        await gateway.place(order)


async def test_a_failed_child_cancel_does_not_hide_the_parent_cancel():
    class _BadCancelIB(_FakeIB):
        def cancelOrder(self, order: Any) -> None:
            self.cancelled.append(order)
            if getattr(order, "parentId", 0):
                raise RuntimeError("child cancel rejected")

    ib = _BadCancelIB()
    gateway = IBKROrderGateway(ib)
    order = _order(stop_loss=Decimal("95"), take_profit=Decimal("110"))
    broker_id = await gateway.place(order)
    await gateway.cancel(broker_id)  # must not raise
    assert len(ib.cancelled) == 3
