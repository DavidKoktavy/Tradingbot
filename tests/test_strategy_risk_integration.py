"""
Strategy -> Risk -> Validator -> Order.

Confirms the Phase 5 layer plugs into the Phase 4 gate without any
shortcut: a strategy's intent is treated exactly like any other.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from data.models import Bar, Instrument, MarketSnapshot
from execution.order_store import OrderStore
from execution.order_validator import OrderValidator
from portfolio.portfolio_manager import AccountState, PortfolioManager
from risk.decisions import RejectionReason
from risk.kill_switch import KillSwitch, KillSwitchTrigger, TradingHalt
from risk.risk_engine import RiskEngine, RiskEngineLimits
from strategies.base import StrategyContext
from strategies.engine import StrategyEngine
from strategies.momentum import MomentumParams, MomentumStrategy

AAPL = Instrument(symbol="AAPL")


def rising_with_pullbacks(n: int = 40) -> list[float]:
    out, price = [], 100.0
    for i in range(n):
        price += 1.2 if i % 3 != 2 else -0.5
        out.append(price)
    return out


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
    engine = RiskEngine(
        limits=RiskEngineLimits(),
        portfolio=portfolio,
        kill_switch=kill_switch,
        trading_halt=halt,
    )
    store = OrderStore()
    validator = OrderValidator(store)
    strategy = MomentumStrategy(
        MomentumParams(
            lookback=10, entry_threshold=0.02, rsi_period=5, atr_period=5, rsi_overbought=90
        )
    )
    return engine, validator, store, kill_switch, halt, portfolio, strategy


def build_context(closes: list[float], equity=Decimal("100000")) -> StrategyContext:
    base = datetime(2026, 1, 5, tzinfo=timezone.utc)
    bars = [
        Bar(
            timestamp=base + timedelta(minutes=i),
            open=c,
            high=c + 1,
            low=c - 1,
            close=c,
            volume=1000,
        )
        for i, c in enumerate(closes)
    ]
    price = closes[-1]
    return StrategyContext(
        instrument=AAPL,
        bars=bars,
        snapshot=MarketSnapshot(
            instrument=AAPL,
            timestamp=datetime.now(timezone.utc),
            bid=price - 0.05,
            ask=price + 0.05,
            last=price,
        ),
        equity=equity,
    )


def test_strategy_signal_becomes_risk_approved_order(stack):
    engine, validator, store, _, _, _, strategy = stack
    ctx = build_context(rising_with_pullbacks())

    signal = strategy.generate_signal(ctx)
    intent = strategy.generate_order_intent(signal, ctx)
    assert intent is not None

    assessment = engine.evaluate(intent, snapshot=ctx.snapshot, prices={})
    assert assessment.approved, assessment.summary()

    decision = validator.validate(intent, assessment, snapshot=ctx.snapshot)
    assert decision.approved

    order = validator.build_order(intent, assessment)
    assert order.intent.quantity == assessment.approved_quantity


def test_strategy_request_is_trimmed_by_risk_engine(stack):
    """The strategy requests up to 10% of equity in notional; the risk
    engine's sizer decides the real number."""
    engine, validator, store, _, _, _, strategy = stack
    ctx = build_context(rising_with_pullbacks())
    signal = strategy.generate_signal(ctx)
    intent = strategy.generate_order_intent(signal, ctx)

    assessment = engine.evaluate(intent, snapshot=ctx.snapshot, prices={})
    assert assessment.approved
    assert assessment.approved_quantity <= intent.quantity


def test_kill_switch_blocks_strategy_output(stack):
    engine, validator, store, kill_switch, _, _, strategy = stack
    kill_switch.activate(KillSwitchTrigger.MANUAL, "operator")
    ctx = build_context(rising_with_pullbacks())
    signal = strategy.generate_signal(ctx)
    intent = strategy.generate_order_intent(signal, ctx)

    assessment = engine.evaluate(intent, snapshot=ctx.snapshot, prices={})
    assert not assessment.approved
    assert assessment.reason is RejectionReason.KILL_SWITCH_ACTIVE
    assert store.all_orders() == []


def test_stale_data_blocks_strategy_output(stack):
    engine, validator, store, _, _, _, strategy = stack
    ctx = build_context(rising_with_pullbacks())
    signal = strategy.generate_signal(ctx)
    intent = strategy.generate_order_intent(signal, ctx)

    stale = ctx.snapshot.model_copy(
        update={"timestamp": datetime.now(timezone.utc) - timedelta(seconds=120)}
    )
    assessment = engine.evaluate(intent, snapshot=stale, prices={})
    assert not assessment.approved
    assert assessment.reason is RejectionReason.STALE_MARKET_DATA


def test_engine_produces_intents_for_multiple_strategies(stack):
    from strategies.mean_reversion import MeanReversionStrategy

    _, _, _, _, _, _, strategy = stack
    strategy_engine = StrategyEngine([strategy, MeanReversionStrategy()])
    results = strategy_engine.evaluate(build_context(rising_with_pullbacks()))
    assert len(results) == 2
    # At least the momentum strategy should have produced an intent.
    assert any(intent is not None for _, intent in results)


def test_strategy_engine_never_touches_order_store(stack):
    """The strategy layer must not create orders on its own."""
    engine, validator, store, _, _, _, strategy = stack
    strategy_engine = StrategyEngine([strategy])
    strategy_engine.evaluate(build_context(rising_with_pullbacks()))
    assert store.all_orders() == []
