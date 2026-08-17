from datetime import datetime, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from data.models import Instrument
from execution.execution_models import (
    Fill,
    IllegalStateTransition,
    Order,
    OrderIntent,
    OrderSide,
    OrderState,
    OrderType,
    can_transition,
)


@pytest.fixture
def instrument():
    return Instrument(symbol="AAPL")


@pytest.fixture
def intent(instrument):
    return OrderIntent(
        instrument=instrument,
        side=OrderSide.BUY,
        quantity=Decimal("100"),
        order_type=OrderType.MARKET,
        source="momentum",
        strategy="momentum",
    )


def _fill(order_id: str, qty: str, price: str, n: int = 1) -> Fill:
    return Fill(
        fill_id=f"f{n}",
        order_id=order_id,
        timestamp=datetime.now(timezone.utc),
        quantity=Decimal(qty),
        price=Decimal(price),
    )


# ---- OrderIntent validation --------------------------------------------


def test_limit_order_requires_limit_price(instrument):
    with pytest.raises(ValidationError):
        OrderIntent(
            instrument=instrument,
            side=OrderSide.BUY,
            quantity=Decimal("10"),
            order_type=OrderType.LIMIT,
        )


def test_market_order_rejects_limit_price(instrument):
    with pytest.raises(ValidationError):
        OrderIntent(
            instrument=instrument,
            side=OrderSide.BUY,
            quantity=Decimal("10"),
            order_type=OrderType.MARKET,
            limit_price=Decimal("100"),
        )


def test_stop_limit_requires_both_prices(instrument):
    with pytest.raises(ValidationError):
        OrderIntent(
            instrument=instrument,
            side=OrderSide.BUY,
            quantity=Decimal("10"),
            order_type=OrderType.STOP_LIMIT,
            limit_price=Decimal("100"),
        )
    ok = OrderIntent(
        instrument=instrument,
        side=OrderSide.BUY,
        quantity=Decimal("10"),
        order_type=OrderType.STOP_LIMIT,
        limit_price=Decimal("100"),
        stop_price=Decimal("99"),
    )
    assert ok.stop_price == Decimal("99")


def test_negative_quantity_rejected(instrument):
    with pytest.raises(ValidationError):
        OrderIntent(instrument=instrument, side=OrderSide.BUY, quantity=Decimal("-5"))


def test_intent_is_immutable(intent):
    with pytest.raises(ValidationError):
        intent.quantity = Decimal("999")


# ---- state machine ------------------------------------------------------


def test_happy_path_transitions(intent):
    order = Order(intent=intent)
    for state in (
        OrderState.VALIDATING,
        OrderState.APPROVED,
        OrderState.SUBMITTED,
        OrderState.ACKNOWLEDGED,
    ):
        order.transition_to(state)
    assert order.state is OrderState.ACKNOWLEDGED
    assert len(order.state_history) == 4


def test_illegal_transition_raises(intent):
    order = Order(intent=intent)
    with pytest.raises(IllegalStateTransition):
        order.transition_to(OrderState.FILLED)  # CREATED -> FILLED is not legal


def test_terminal_states_are_terminal(intent):
    order = Order(intent=intent)
    order.transition_to(OrderState.VALIDATING)
    order.transition_to(OrderState.REJECTED)
    assert order.is_terminal
    with pytest.raises(IllegalStateTransition):
        order.transition_to(OrderState.SUBMITTED)


def test_cancel_can_lose_race_with_fill(intent):
    # A cancel request can be beaten by a fill; the broker is authoritative.
    assert can_transition(OrderState.CANCEL_REQUESTED, OrderState.FILLED)


# ---- fills --------------------------------------------------------------


def test_partial_then_complete_fill(intent):
    order = Order(intent=intent)
    order.transition_to(OrderState.VALIDATING)
    order.transition_to(OrderState.APPROVED)
    order.transition_to(OrderState.SUBMITTED)

    order.apply_fill(_fill(order.order_id, "40", "100.00", 1))
    assert order.state is OrderState.PARTIALLY_FILLED
    assert order.filled_quantity == Decimal("40")
    assert order.remaining_quantity == Decimal("60")

    order.apply_fill(_fill(order.order_id, "60", "101.00", 2))
    assert order.state is OrderState.FILLED
    assert order.remaining_quantity == Decimal("0")
    # VWAP: (40*100 + 60*101) / 100 = 100.6
    assert order.average_fill_price == Decimal("100.6")


def test_overfill_is_rejected(intent):
    order = Order(intent=intent)
    order.transition_to(OrderState.VALIDATING)
    order.transition_to(OrderState.APPROVED)
    order.transition_to(OrderState.SUBMITTED)
    with pytest.raises(ValueError, match="overfill"):
        order.apply_fill(_fill(order.order_id, "150", "100.00"))


def test_cannot_fill_terminal_order(intent):
    order = Order(intent=intent)
    order.transition_to(OrderState.VALIDATING)
    order.transition_to(OrderState.REJECTED)
    with pytest.raises(IllegalStateTransition):
        order.apply_fill(_fill(order.order_id, "10", "100.00"))


def test_fill_is_only_path_to_filled(intent):
    # There must be no way to reach FILLED from SUBMITTED except via a fill
    # object; transition_to alone is legal but apply_fill is what the
    # execution listener uses. Verify direct CREATED->FILLED is blocked.
    order = Order(intent=intent)
    assert not can_transition(OrderState.CREATED, OrderState.FILLED)
    assert not can_transition(OrderState.APPROVED, OrderState.FILLED)
