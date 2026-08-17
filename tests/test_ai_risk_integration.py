"""
AI -> OrderIntent -> RiskEngine -> OrderValidator -> Order.

The point of this file is to demonstrate that an AI-originated intent is
treated *identically* to a strategy-originated one, and that every risk
control still binds.
"""

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from ai.decision_engine import AIDecisionEngine
from ai.providers import FailingProvider, ScriptedProvider
from data.models import Instrument, MarketSnapshot
from execution.order_store import OrderStore
from execution.order_validator import OrderValidator
from portfolio.portfolio_manager import AccountState, PortfolioManager
from risk.decisions import RejectionReason
from risk.kill_switch import KillSwitch, KillSwitchTrigger, TradingHalt
from risk.risk_engine import RiskEngine, RiskEngineLimits

AAPL = Instrument(symbol="AAPL")


def snap(mid: float = 100.0, age: float = 0.0) -> MarketSnapshot:
    return MarketSnapshot(
        instrument=AAPL,
        timestamp=datetime.now(timezone.utc) - timedelta(seconds=age),
        bid=mid - 0.05,
        ask=mid + 0.05,
        last=mid,
    )


def ai_response(**overrides) -> str:
    payload = {
        "action": "BUY",
        "symbol": "AAPL",
        "confidence": 0.85,
        "strategy": "momentum",
        "entry": 100.0,
        "stop_loss": 96.0,
        "take_profit": 108.0,
        "reasoning": "Uptrend intact.",
        "time_horizon": "intraday",
        "risk_score": 0.3,
    }
    payload.update(overrides)
    return json.dumps(payload)


@pytest.fixture
def stack():
    portfolio = PortfolioManager()
    portfolio.update_account(
        AccountState(
            equity=Decimal("100000"), cash=Decimal("100000"), buying_power=Decimal("200000")
        )
    )
    kill_switch = KillSwitch()
    halt = TradingHalt()
    risk = RiskEngine(
        limits=RiskEngineLimits(),
        portfolio=portfolio,
        kill_switch=kill_switch,
        trading_halt=halt,
    )
    store = OrderStore()
    validator = OrderValidator(store)
    return risk, validator, store, kill_switch, halt, portfolio


async def run_ai(stack, response: str, *, snapshot=None):
    risk, validator, store, *_ = stack
    snapshot = snapshot or snap()
    engine = AIDecisionEngine(ScriptedProvider([response]), allowed_symbols={"AAPL"})
    result = await engine.decide(instrument=AAPL, snapshot=snapshot)
    if not result.accepted:
        return result, None, None
    intent = AIDecisionEngine.to_order_intent(
        result.decision, instrument=AAPL, snapshot=snapshot, equity=Decimal("100000")
    )
    if intent is None:
        return result, None, None
    assessment = risk.evaluate(intent, snapshot=snapshot, prices={})
    return result, intent, assessment


async def test_ai_decision_reaches_an_order(stack):
    risk, validator, store, *_ = stack
    result, intent, assessment = await run_ai(stack, ai_response())
    assert result.accepted
    assert assessment.approved, assessment.summary()

    decision = validator.validate(intent, assessment, snapshot=snap())
    assert decision.approved
    order = validator.build_order(intent, assessment)
    assert order.intent.source == "ai"
    assert order.intent.quantity == assessment.approved_quantity


async def test_kill_switch_blocks_ai_decision(stack):
    risk, validator, store, kill_switch, *_ = stack
    kill_switch.activate(KillSwitchTrigger.MANUAL, "operator halt")
    result, intent, assessment = await run_ai(stack, ai_response())
    assert result.accepted  # the AI produced a valid opinion
    assert not assessment.approved  # but it changes nothing
    assert assessment.reason is RejectionReason.KILL_SWITCH_ACTIVE
    assert store.all_orders() == []


async def test_maximum_confidence_does_not_bypass_kill_switch(stack):
    risk, validator, store, kill_switch, *_ = stack
    kill_switch.activate(KillSwitchTrigger.DAILY_LOSS_LIMIT, "limit hit")
    _, _, assessment = await run_ai(stack, ai_response(confidence=1.0, risk_score=0.0))
    assert not assessment.approved
    assert assessment.reason is RejectionReason.KILL_SWITCH_ACTIVE


async def test_ai_position_is_sized_by_risk_engine_not_by_ai(stack):
    """The AI expresses no size at all; the sizer decides independently.

    The nominal request is deliberately made much larger than the
    risk-permitted size, so the assertion can distinguish 'sized by the
    risk engine' from 'passed through unchanged'.
    """
    risk, validator, store, *_ = stack
    snapshot = snap()
    engine = AIDecisionEngine(ScriptedProvider([ai_response()]), allowed_symbols={"AAPL"})
    result = await engine.decide(instrument=AAPL, snapshot=snapshot)
    assert result.accepted

    # Nominal request of 1000 shares (10% of a notional $1m).
    intent = AIDecisionEngine.to_order_intent(
        result.decision, instrument=AAPL, snapshot=snapshot, equity=Decimal("1000000")
    )
    assert intent.quantity == Decimal("1000")

    # Actual account equity is $100k: 0.5% risk = $500, stop distance 4
    # -> 125 shares, capped by max_position_size to 100.
    assessment = risk.evaluate(intent, snapshot=snapshot, prices={})
    assert assessment.approved
    assert assessment.approved_quantity == Decimal("100")
    assert assessment.approved_quantity < intent.quantity
    assert assessment.was_reduced


async def test_stale_data_blocks_ai_decision(stack):
    stale = snap(age=300)
    result, intent, assessment = await run_ai(stack, ai_response(), snapshot=stale)
    assert not assessment.approved
    assert assessment.reason is RejectionReason.STALE_MARKET_DATA


async def test_daily_loss_limit_blocks_ai_decision(stack):
    risk, validator, store, kill_switch, halt, portfolio = stack
    portfolio.update_account(
        AccountState(equity=Decimal("97000"), buying_power=Decimal("100000"))
    )
    _, _, assessment = await run_ai(stack, ai_response())
    assert not assessment.approved
    assert assessment.reason is RejectionReason.MAX_DAILY_LOSS_BREACHED


async def test_ai_failure_produces_no_order(stack):
    risk, validator, store, *_ = stack
    engine = AIDecisionEngine(FailingProvider(), allowed_symbols={"AAPL"})
    result = await engine.decide(instrument=AAPL, snapshot=snap())
    assert not result.accepted
    assert result.decision is None
    assert store.all_orders() == []


async def test_ai_intent_is_indistinguishable_from_strategy_intent(stack):
    """Apart from provenance, the risk engine sees an ordinary intent."""
    from strategies.base import Strategy

    _, intent, _ = await run_ai(stack, ai_response())
    fields = set(intent.model_dump())
    # No AI-specific privileged fields exist on the intent at all.
    assert "confidence" not in fields
    assert "risk_score" not in fields
    assert "priority" not in fields
    assert intent.source == "ai"


async def test_duplicate_ai_decisions_are_deduped(stack):
    risk, validator, store, *_ = stack
    _, intent, assessment = await run_ai(stack, ai_response())
    validator.build_order(intent, assessment)

    _, intent2, assessment2 = await run_ai(stack, ai_response())
    decision = validator.validate(intent2, assessment2, snapshot=snap())
    assert not decision.approved
    assert decision.reason is RejectionReason.DUPLICATE_ORDER
    assert len(store.all_orders()) == 1


async def test_ai_cannot_exceed_gross_exposure(stack):
    risk, validator, store, kill_switch, halt, portfolio = stack
    tight = RiskEngine(
        limits=RiskEngineLimits(
            max_gross_exposure=Decimal("0.01"),
            max_position_size=Decimal("1.0"),
            max_risk_per_trade=Decimal("0.5"),
        ),
        portfolio=portfolio,
        kill_switch=kill_switch,
        trading_halt=halt,
    )
    engine = AIDecisionEngine(ScriptedProvider([ai_response()]), allowed_symbols={"AAPL"})
    result = await engine.decide(instrument=AAPL, snapshot=snap())
    intent = AIDecisionEngine.to_order_intent(
        result.decision, instrument=AAPL, snapshot=snap(), equity=Decimal("100000")
    )
    assessment = tight.evaluate(intent, snapshot=snap(), prices={})
    assert not assessment.approved
    assert assessment.reason is RejectionReason.MAX_GROSS_EXPOSURE_EXCEEDED
