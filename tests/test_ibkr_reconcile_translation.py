"""The IBKR gateway must hand the reconciler the same shapes the simulator does.

Found in a live paper run, not by the suite: `IBKROrderGateway.open_orders()`
and `.positions()` returned raw ib_async `Trade` / `Position` objects, while
the reconciler (and the smoke test) read `broker_order_id`, `instrument`,
`quantity`, ... — fields only the simulator's `BrokerOrder`/`BrokerPosition`
have. Every existing test used the simulator, so none of them could see it.

These tests deliberately build *real* ib_async objects.

They also cover the bracket children introduced with protective exits: a stop
and a take-profit are separate live orders at IBKR with ids we never stored,
and must not be mistaken for unknown orders (which halts trading).
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest
from ib_async import Contract, LimitOrder, OrderStatus, Stock, StopOrder, Trade
from ib_async.objects import Position

from broker.order_manager import IBKROrderGateway
from data.models import AssetClass, Instrument
from execution.execution_models import Fill, Order, OrderIntent, OrderSide, OrderState
from execution.order_store import OrderStore
from execution.reconciliation import BrokerOrder, BrokerPosition, Reconciler
from portfolio.portfolio_manager import AccountState, PortfolioManager

AAPL = Instrument(symbol="AAPL")


class _FakeIB:
    def __init__(self, trades: list[Any] | None = None, positions: list[Any] | None = None):
        self._trades = trades or []
        self._positions = positions or []

    async def reqOpenOrdersAsync(self) -> list[Any]:
        return self._trades

    async def reqPositionsAsync(self) -> list[Any]:
        return self._positions


def _trade(order: Any, status: str = "Submitted", filled: float = 0.0, contract=None) -> Trade:
    return Trade(
        contract=contract or Stock("AAPL", "SMART", "USD"),
        order=order,
        orderStatus=OrderStatus(orderId=order.orderId, status=status, filled=filled),
    )


def _ib_position(qty: float, avg: float, contract=None) -> Position:
    # IBKR reports the listing exchange on positions, not SMART.
    contract = contract or Stock("AAPL", "", "USD", primaryExchange="NASDAQ")
    return Position(account="DU0000000", contract=contract, position=qty, avgCost=avg)


@pytest.fixture
def setup():
    store = OrderStore()
    portfolio = PortfolioManager()
    portfolio.update_account(AccountState(equity=Decimal(100000)))
    return store, portfolio, Reconciler(store, portfolio)


def _seed_position(portfolio, qty: str, price: str) -> None:
    order = Order(
        intent=OrderIntent(instrument=AAPL, side=OrderSide.BUY, quantity=Decimal(qty), source="t")
    )
    portfolio.apply_fill(
        order,
        Fill(
            fill_id="f",
            order_id=order.order_id,
            timestamp=datetime.now(UTC),
            quantity=Decimal(qty),
            price=Decimal(price),
        ),
    )


def _local_live_order(store: OrderStore, broker_id: str) -> Order:
    order = Order(
        intent=OrderIntent(
            instrument=AAPL, side=OrderSide.BUY, quantity=Decimal(1), source="ma_crossover"
        )
    )
    for state in (
        OrderState.VALIDATING,
        OrderState.APPROVED,
        OrderState.SUBMITTED,
        OrderState.ACKNOWLEDGED,
    ):
        order.transition_to(state)
    store.add(order)
    store.link_broker_id(order.order_id, broker_id)
    return order


# --- translation -----------------------------------------------------------


async def test_open_orders_returns_broker_orders_not_raw_trades():
    order = LimitOrder("BUY", 1, 272.02)
    order.orderId = 30
    gateway = IBKROrderGateway(_FakeIB(trades=[_trade(order)]))

    [bo] = await gateway.open_orders()

    assert isinstance(bo, BrokerOrder)
    assert bo.broker_order_id == "30"
    assert str(bo.instrument) == "AAPL:SMART:USD"
    assert bo.quantity == Decimal(1)
    assert bo.state is OrderState.ACKNOWLEDGED
    assert bo.parent_broker_order_id is None


async def test_positions_return_broker_positions_keyed_smart():
    gateway = IBKROrderGateway(_FakeIB(positions=[_ib_position(2, 340.5)]))

    [bp] = await gateway.positions()

    assert isinstance(bp, BrokerPosition)
    # Keyed SMART even though IBKR reported NASDAQ, so it matches what the
    # bot itself records for a position it opened.
    assert str(bp.instrument) == str(AAPL)
    assert bp.quantity == Decimal(2)
    assert bp.average_cost == Decimal("340.5")


@pytest.mark.parametrize(
    ("status", "filled", "expected"),
    [
        ("PendingSubmit", 0, OrderState.SUBMITTED),
        ("PreSubmitted", 0, OrderState.ACKNOWLEDGED),
        ("Submitted", 0, OrderState.ACKNOWLEDGED),
        ("Submitted", 1, OrderState.PARTIALLY_FILLED),
        ("PendingCancel", 0, OrderState.CANCEL_REQUESTED),
        ("Cancelled", 0, OrderState.CANCELLED),
        ("Filled", 2, OrderState.FILLED),
        ("Inactive", 0, OrderState.REJECTED),
        ("SomethingNew", 0, OrderState.ERROR),
    ],
)
async def test_status_mapping(status, filled, expected):
    order = LimitOrder("BUY", 2, 100.0)
    order.orderId = 1
    gateway = IBKROrderGateway(_FakeIB(trades=[_trade(order, status, filled)]))
    [bo] = await gateway.open_orders()
    assert bo.state is expected


async def test_non_stock_position_cannot_collide_with_a_stock():
    fx = Contract(secType="CASH", symbol="EUR", currency="USD", localSymbol="EUR.USD",
                  exchange="IDEALPRO")
    fund = Contract(secType="FUND", symbol="AAPL", currency="USD")
    gateway = IBKROrderGateway(_FakeIB(positions=[_ib_position(1, 1, fx),
                                                  _ib_position(1, 1, fund)]))
    fx_bp, fund_bp = await gateway.positions()
    assert fx_bp.instrument.asset_class is AssetClass.FOREX
    assert str(fund_bp.instrument) != str(AAPL)


# --- through the reconciler ------------------------------------------------


async def test_real_ibkr_position_reconciles_clean_against_own_record(setup):
    _store, portfolio, reconciler = setup
    _seed_position(portfolio, "2", "340.5")
    gateway = IBKROrderGateway(_FakeIB(positions=[_ib_position(2, 340.5)]))

    report = reconciler.reconcile(
        broker_positions=await gateway.positions(), broker_orders=await gateway.open_orders()
    )

    assert report.is_clean, [d.detail for d in report.discrepancies]


async def test_position_opened_outside_the_bot_still_halts(setup):
    _store, _portfolio, reconciler = setup
    gateway = IBKROrderGateway(_FakeIB(positions=[_ib_position(5, 200)]))

    report = reconciler.reconcile(broker_positions=await gateway.positions(), broker_orders=[])

    assert report.requires_halt
    assert report.discrepancies[0].kind == "UNKNOWN_BROKER_POSITION"


def _bracket_children(parent_id: int) -> list[Trade]:
    tp = LimitOrder("SELL", 1, 380.0)
    tp.orderId, tp.parentId = parent_id + 1, parent_id
    sl = StopOrder("SELL", 1, 320.0)
    sl.orderId, sl.parentId = parent_id + 2, parent_id
    return [_trade(tp, "PreSubmitted"), _trade(sl, "PreSubmitted")]


async def test_bracket_children_of_our_own_entry_are_not_discrepancies(setup):
    store, _portfolio, reconciler = setup
    _local_live_order(store, "900")
    parent = LimitOrder("BUY", 1, 340.0)
    parent.orderId = 900
    gateway = IBKROrderGateway(_FakeIB(trades=[_trade(parent), *_bracket_children(900)]))

    report = reconciler.reconcile(broker_positions=[], broker_orders=await gateway.open_orders())

    assert not report.requires_halt, [d.detail for d in report.discrepancies]


async def test_bracket_children_after_entry_filled_are_not_discrepancies(setup):
    store, portfolio, reconciler = setup
    entry = _local_live_order(store, "900")
    entry.transition_to(OrderState.FILLED)
    _seed_position(portfolio, "1", "340")
    gateway = IBKROrderGateway(
        _FakeIB(trades=_bracket_children(900), positions=[_ib_position(1, 340)])
    )

    report = reconciler.reconcile(
        broker_positions=await gateway.positions(), broker_orders=await gateway.open_orders()
    )

    assert report.is_clean, [d.detail for d in report.discrepancies]


async def test_child_whose_parent_we_do_not_know_still_halts(setup):
    _store, _portfolio, reconciler = setup
    gateway = IBKROrderGateway(_FakeIB(trades=_bracket_children(555)))

    report = reconciler.reconcile(broker_positions=[], broker_orders=await gateway.open_orders())

    assert report.requires_halt
    assert {d.kind for d in report.discrepancies} == {"UNKNOWN_BROKER_ORDER"}
