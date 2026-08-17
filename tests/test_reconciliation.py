from datetime import datetime, timezone
from decimal import Decimal

import pytest

from data.models import Instrument
from execution.execution_models import Fill, Order, OrderIntent, OrderSide, OrderState
from execution.order_store import OrderStore
from execution.reconciliation import BrokerOrder, BrokerPosition, Reconciler
from portfolio.portfolio_manager import AccountState, PortfolioManager

AAPL = Instrument(symbol="AAPL")
MSFT = Instrument(symbol="MSFT")


@pytest.fixture
def setup():
    store = OrderStore()
    portfolio = PortfolioManager()
    portfolio.update_account(AccountState(equity=Decimal("100000")))
    return store, portfolio, Reconciler(store, portfolio)


def _seed_position(portfolio, instrument, qty: str, price: str, side=OrderSide.BUY):
    order = Order(
        intent=OrderIntent(instrument=instrument, side=side, quantity=Decimal(qty), source="test")
    )
    fill = Fill(
        fill_id="f",
        order_id=order.order_id,
        timestamp=datetime.now(timezone.utc),
        quantity=Decimal(qty),
        price=Decimal(price),
    )
    portfolio.apply_fill(order, fill)
    return order


def test_clean_reconciliation(setup):
    store, portfolio, reconciler = setup
    _seed_position(portfolio, AAPL, "100", "50")
    report = reconciler.reconcile(
        broker_positions=[
            BrokerPosition(instrument=AAPL, quantity=Decimal("100"), average_cost=Decimal("50"))
        ],
        broker_orders=[],
    )
    assert report.is_clean
    assert not report.requires_halt


def test_unknown_broker_position_is_adopted_and_halts(setup):
    store, portfolio, reconciler = setup
    report = reconciler.reconcile(
        broker_positions=[
            BrokerPosition(instrument=MSFT, quantity=Decimal("75"), average_cost=Decimal("300"))
        ],
        broker_orders=[],
    )
    assert not report.is_clean
    assert report.requires_halt
    assert report.discrepancies[0].kind == "UNKNOWN_BROKER_POSITION"
    # Adopted so risk calcs include it.
    assert portfolio.get_position(MSFT).quantity == Decimal("75")


def test_position_quantity_mismatch_adopts_broker_value(setup):
    store, portfolio, reconciler = setup
    _seed_position(portfolio, AAPL, "100", "50")
    report = reconciler.reconcile(
        broker_positions=[
            BrokerPosition(instrument=AAPL, quantity=Decimal("140"), average_cost=Decimal("51"))
        ],
        broker_orders=[],
    )
    assert report.requires_halt
    assert portfolio.get_position(AAPL).quantity == Decimal("140")
    assert portfolio.get_position(AAPL).average_cost == Decimal("51")


def test_local_position_broker_has_none_is_flattened(setup):
    store, portfolio, reconciler = setup
    _seed_position(portfolio, AAPL, "100", "50")
    report = reconciler.reconcile(broker_positions=[], broker_orders=[])
    assert report.requires_halt
    assert portfolio.get_position(AAPL).is_flat


def test_unknown_broker_order_flags_halt(setup):
    store, portfolio, reconciler = setup
    report = reconciler.reconcile(
        broker_positions=[],
        broker_orders=[
            BrokerOrder(
                broker_order_id="IB-999",
                instrument=AAPL,
                quantity=Decimal("50"),
                filled_quantity=Decimal("0"),
                state=OrderState.ACKNOWLEDGED,
            )
        ],
    )
    assert report.requires_halt
    assert report.orders_adopted == 1
    assert report.discrepancies[0].kind == "UNKNOWN_BROKER_ORDER"


def test_locally_active_order_missing_at_broker_flags_halt(setup):
    store, portfolio, reconciler = setup
    order = Order(
        intent=OrderIntent(
            instrument=AAPL, side=OrderSide.BUY, quantity=Decimal("10"), source="test"
        )
    )
    order.transition_to(OrderState.VALIDATING)
    order.transition_to(OrderState.APPROVED)
    order.transition_to(OrderState.SUBMITTED)
    store.add(order)
    store.link_broker_id(order.order_id, "IB-111")

    report = reconciler.reconcile(broker_positions=[], broker_orders=[])
    kinds = {d.kind for d in report.discrepancies}
    assert "MISSING_AT_BROKER" in kinds
    assert report.requires_halt


def test_order_state_mismatch_adopts_legal_broker_state(setup):
    store, portfolio, reconciler = setup
    order = Order(
        intent=OrderIntent(
            instrument=AAPL, side=OrderSide.BUY, quantity=Decimal("10"), source="test"
        )
    )
    order.transition_to(OrderState.VALIDATING)
    order.transition_to(OrderState.APPROVED)
    order.transition_to(OrderState.SUBMITTED)
    store.add(order)
    store.link_broker_id(order.order_id, "IB-222")

    report = reconciler.reconcile(
        broker_positions=[],
        broker_orders=[
            BrokerOrder(
                broker_order_id="IB-222",
                instrument=AAPL,
                quantity=Decimal("10"),
                filled_quantity=Decimal("0"),
                state=OrderState.ACKNOWLEDGED,
            )
        ],
    )
    assert order.state is OrderState.ACKNOWLEDGED
    # State mismatch alone is informational, not a halt condition.
    assert not report.requires_halt


def test_reconciler_never_places_compensating_trades(setup):
    """Reconciliation must only *report* and adopt state — it must never
    generate orders. Verified structurally: the store gains no orders."""
    store, portfolio, reconciler = setup
    _seed_position(portfolio, AAPL, "100", "50")
    reconciler.reconcile(
        broker_positions=[
            BrokerPosition(instrument=AAPL, quantity=Decimal("40"), average_cost=Decimal("50"))
        ],
        broker_orders=[],
    )
    assert store.all_orders() == []
