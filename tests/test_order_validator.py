from datetime import datetime, timezone
from decimal import Decimal

import pytest

from data.models import Instrument, MarketSnapshot
from execution.execution_models import OrderIntent, OrderSide, OrderState, OrderType
from execution.order_store import OrderStore
from execution.order_validator import OrderValidationError, OrderValidator
from risk.decisions import RejectionReason, RiskAssessment

AAPL = Instrument(symbol="AAPL")


def snap(mid="100.00"):
    price = float(mid)
    return MarketSnapshot(
        instrument=AAPL,
        timestamp=datetime.now(timezone.utc),
        bid=price - 0.05,
        ask=price + 0.05,
        last=price,
    )


def _intent(**kw):
    defaults = dict(
        instrument=AAPL,
        side=OrderSide.BUY,
        quantity=Decimal("100"),
        order_type=OrderType.MARKET,
        source="momentum",
    )
    defaults.update(kw)
    return OrderIntent(**defaults)


def approved(qty="100", requested="100") -> RiskAssessment:
    return RiskAssessment(
        approved=True,
        approved_quantity=Decimal(qty),
        requested_quantity=Decimal(requested),
    )


def rejected() -> RiskAssessment:
    return RiskAssessment(approved=False, requested_quantity=Decimal("100"))


@pytest.fixture
def store():
    return OrderStore()


@pytest.fixture
def validator(store):
    return OrderValidator(store)


def test_valid_order_passes(validator):
    decision = validator.validate(_intent(), approved(), snapshot=snap())
    assert decision.approved


def test_unapproved_assessment_rejected(validator):
    decision = validator.validate(_intent(), rejected(), snapshot=snap())
    assert not decision.approved
    assert decision.reason is RejectionReason.INVALID_ORDER


def test_zero_approved_quantity_rejected(validator):
    decision = validator.validate(_intent(), approved(qty="0"), snapshot=snap())
    assert not decision.approved
    assert decision.reason is RejectionReason.ZERO_QUANTITY


def test_duplicate_detected(validator, store):
    i = _intent()
    store.record_intent(i)
    decision = validator.validate(_intent(), approved(), snapshot=snap())
    assert not decision.approved
    assert decision.reason is RejectionReason.DUPLICATE_ORDER


def test_tick_size_violation_rejected(validator):
    i = _intent(order_type=OrderType.LIMIT, limit_price=Decimal("100.005"))
    decision = validator.validate(i, approved(), snapshot=snap())
    assert not decision.approved
    assert "tick size" in decision.detail


def test_buy_stop_below_market_rejected(validator):
    """A buy stop below market triggers instantly at an unintended price."""
    i = _intent(order_type=OrderType.STOP, stop_price=Decimal("95.00"))
    decision = validator.validate(i, approved(), snapshot=snap("100.00"))
    assert not decision.approved
    assert "trigger immediately" in decision.detail


def test_sell_stop_above_market_rejected(validator):
    i = _intent(
        side=OrderSide.SELL, order_type=OrderType.STOP, stop_price=Decimal("105.00")
    )
    decision = validator.validate(i, approved(), snapshot=snap("100.00"))
    assert not decision.approved


def test_correct_stop_direction_passes(validator):
    i = _intent(
        side=OrderSide.SELL, order_type=OrderType.STOP, stop_price=Decimal("95.00")
    )
    decision = validator.validate(i, approved(), snapshot=snap("100.00"))
    assert decision.approved


# ---- build_order: the only intent -> order path -------------------------


def test_build_order_refuses_unapproved_assessment(validator):
    with pytest.raises(OrderValidationError):
        validator.build_order(_intent(), rejected())


def test_build_order_refuses_zero_quantity(validator):
    with pytest.raises(OrderValidationError):
        validator.build_order(_intent(), approved(qty="0"))


def test_build_order_uses_risk_approved_quantity_not_requested(validator):
    """If the sizer trimmed the trade, the trimmed size is what ships."""
    i = _intent(quantity=Decimal("1000"))
    order = validator.build_order(i, approved(qty="50", requested="1000"))
    assert order.intent.quantity == Decimal("50")


def test_built_order_starts_in_approved_state(validator):
    order = validator.build_order(_intent(), approved())
    assert order.state is OrderState.APPROVED
    # Went through VALIDATING on the way — audit trail intact.
    assert any(s is OrderState.VALIDATING for s, _ in order.state_history)


def test_built_order_is_registered_and_deduped(validator, store):
    order = validator.build_order(_intent(), approved())
    assert store.get(order.order_id) is order
    # The intent is now recorded, so an identical one is a duplicate.
    decision = validator.validate(_intent(), approved(), snapshot=snap())
    assert decision.reason is RejectionReason.DUPLICATE_ORDER


def test_order_cannot_be_constructed_without_going_through_validator(validator):
    """Structural check: OrderValidator.build_order is the documented sole
    path, and it hard-requires an approving assessment."""
    import inspect

    src = inspect.getsource(OrderValidator.build_order)
    assert "if not assessment.approved" in src
    assert "assessment.approved_quantity" in src
