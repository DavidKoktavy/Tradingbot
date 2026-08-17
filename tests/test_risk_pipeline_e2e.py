"""
End-to-end test of the deterministic gate chain:

    OrderIntent -> RiskEngine -> OrderValidator -> Order

Including explicit adversarial cases where an "AI" intent tries to get
around the controls. These are the tests that would fail loudly if a
future refactor accidentally opened a bypass.
"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from data.models import Instrument, MarketSnapshot
from execution.execution_models import OrderIntent, OrderSide, OrderState, OrderType
from execution.order_store import OrderStore
from execution.order_validator import OrderValidationError, OrderValidator
from portfolio.portfolio_manager import AccountState, PortfolioManager
from risk.decisions import RejectionReason
from risk.kill_switch import KillSwitch, KillSwitchTrigger, TradingHalt
from risk.risk_engine import RiskEngine, RiskEngineLimits

AAPL = Instrument(symbol="AAPL")


@pytest.fixture
def pipeline():
    portfolio = PortfolioManager()
    portfolio.update_account(
        AccountState(
            equity=Decimal("100000"),
            cash=Decimal("100000"),
            buying_power=Decimal("200000"),
        )
    )
    kill_switch = KillSwitch()
    halt = TradingHalt()
    engine = RiskEngine(
        limits=RiskEngineLimits(),
        portfolio=portfolio,
        kill_switch=kill_switch,
        trading_halt=halt,
    )
    store = OrderStore()
    validator = OrderValidator(store)
    return engine, validator, store, kill_switch, halt, portfolio


def snap():
    return MarketSnapshot(
        instrument=AAPL,
        timestamp=datetime.now(timezone.utc),
        bid=99.95,
        ask=100.05,
        last=100.0,
    )


def ai_intent(**kw):
    defaults = dict(
        instrument=AAPL,
        side=OrderSide.BUY,
        quantity=Decimal("100"),
        order_type=OrderType.MARKET,
        stop_loss=Decimal("95"),
        source="ai",
        strategy="momentum",
    )
    defaults.update(kw)
    return OrderIntent(**defaults)


def run(engine, validator, intent, prices=None):
    assessment = engine.evaluate(intent, snapshot=snap(), prices=prices or {})
    if not assessment.approved:
        return assessment, None, None
    decision = validator.validate(intent, assessment, snapshot=snap())
    if not decision.approved:
        return assessment, decision, None
    return assessment, decision, validator.build_order(intent, assessment)


def test_full_happy_path(pipeline):
    engine, validator, store, *_ = pipeline
    assessment, decision, order = run(engine, validator, ai_intent())
    assert assessment.approved
    assert decision.approved
    assert order is not None
    assert order.state is OrderState.APPROVED
    assert store.get(order.order_id) is order


def test_kill_switch_stops_pipeline_before_order_exists(pipeline):
    engine, validator, store, kill_switch, *_ = pipeline
    kill_switch.activate(KillSwitchTrigger.MANUAL, "operator halt")
    assessment, decision, order = run(engine, validator, ai_intent())
    assert not assessment.approved
    assert order is None
    assert store.all_orders() == []  # nothing was created


def test_ai_oversized_request_is_silently_trimmed(pipeline):
    """The AI asks for 10,000 shares. It gets the risk-permitted size,
    with no error and no negotiation."""
    engine, validator, store, *_ = pipeline
    assessment, decision, order = run(engine, validator, ai_intent(quantity=Decimal("10000")))
    assert assessment.approved
    assert assessment.was_reduced
    assert order.intent.quantity < Decimal("10000")
    assert order.intent.quantity == assessment.approved_quantity


def test_ai_cannot_bypass_by_omitting_stop(pipeline):
    """No stop and no volatility estimate means unknown risk, which means
    no trade — not a default position size."""
    engine, validator, store, *_ = pipeline
    assessment, decision, order = run(engine, validator, ai_intent(stop_loss=None))
    assert not assessment.approved
    assert assessment.reason is RejectionReason.MAX_RISK_PER_TRADE_EXCEEDED
    assert order is None


def test_ai_cannot_construct_order_directly_from_rejected_assessment(pipeline):
    """Even holding a rejected assessment, there is no path to an Order."""
    engine, validator, store, kill_switch, *_ = pipeline
    kill_switch.activate(KillSwitchTrigger.MANUAL)
    intent = ai_intent()
    assessment = engine.evaluate(intent, snapshot=snap(), prices={})
    with pytest.raises(OrderValidationError):
        validator.build_order(intent, assessment)


def test_ai_cannot_forge_approval_by_mutating_intent(pipeline):
    """OrderIntent is frozen — an AI-driven code path cannot enlarge a
    quantity after it was approved."""
    engine, validator, store, *_ = pipeline
    intent = ai_intent(quantity=Decimal("10"))
    with pytest.raises(ValidationError):
        intent.quantity = Decimal("100000")


def test_repeated_identical_ai_signal_is_deduped(pipeline):
    """A strategy re-firing every tick must not stack up 3x the position."""
    engine, validator, store, *_ = pipeline
    _, _, first = run(engine, validator, ai_intent())
    assert first is not None

    assessment, decision, second = run(engine, validator, ai_intent())
    assert assessment.approved  # risk is fine
    assert not decision.approved  # but it's a duplicate
    assert decision.reason is RejectionReason.DUPLICATE_ORDER
    assert second is None
    assert len(store.all_orders()) == 1


def test_every_decision_is_recorded_for_audit(pipeline):
    engine, validator, *_ = pipeline
    assessment = engine.evaluate(ai_intent(), snapshot=snap(), prices={})
    checks_run = {d.check_name for d in assessment.decisions}
    for expected in ("kill_switch", "trading_halt", "market_data", "daily_loss",
                     "drawdown", "position_sizing", "position_size", "gross_exposure"):
        assert expected in checks_run, f"{expected} not recorded in audit trail"


def test_rejection_short_circuits_but_records_what_ran(pipeline):
    engine, validator, store, kill_switch, *_ = pipeline
    kill_switch.activate(KillSwitchTrigger.MANUAL)
    assessment = engine.evaluate(ai_intent(), snapshot=snap(), prices={})
    assert len(assessment.decisions) == 1  # short-circuited immediately
    assert assessment.summary().startswith("REJECTED")


# ---- protective level invariants ----------------------------------------


def test_market_order_may_carry_protective_stop():
    """A MARKET entry with an attached protective stop is a normal
    bracket, and must not be confused with a STOP order."""
    intent = OrderIntent(
        instrument=AAPL,
        side=OrderSide.BUY,
        quantity=Decimal("100"),
        order_type=OrderType.MARKET,
        stop_loss=Decimal("95"),
        take_profit=Decimal("110"),
        source="test",
    )
    assert intent.order_type is OrderType.MARKET
    assert intent.stop_price is None


def test_inverted_protective_levels_rejected_long():
    with pytest.raises(ValidationError, match="stop_loss must be below take_profit"):
        OrderIntent(
            instrument=AAPL,
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            stop_loss=Decimal("110"),
            take_profit=Decimal("95"),
            source="test",
        )


def test_inverted_protective_levels_rejected_short():
    with pytest.raises(ValidationError, match="stop_loss must be above take_profit"):
        OrderIntent(
            instrument=AAPL,
            side=OrderSide.SELL,
            quantity=Decimal("100"),
            stop_loss=Decimal("95"),
            take_profit=Decimal("110"),
            source="test",
        )
