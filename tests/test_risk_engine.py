from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from data.models import Instrument, MarketSnapshot
from execution.execution_models import (
    Fill,
    Order,
    OrderIntent,
    OrderSide,
    OrderType,
)
from portfolio.portfolio_manager import AccountState, PortfolioManager
from risk.decisions import RejectionReason
from risk.kill_switch import HaltReason, KillSwitch, KillSwitchTrigger, TradingHalt
from risk.rate_limiter import OrderRateLimiter
from risk.risk_engine import RiskEngine, RiskEngineLimits

AAPL = Instrument(symbol="AAPL")
AAPL_KEY = "AAPL:SMART:USD"


def snapshot(
    *, bid="99.95", ask="100.05", age_seconds: float = 0.0, instrument=AAPL
) -> MarketSnapshot:
    return MarketSnapshot(
        instrument=instrument,
        timestamp=datetime.now(timezone.utc) - timedelta(seconds=age_seconds),
        bid=float(bid),
        ask=float(ask),
        last=100.0,
    )


def intent(
    *,
    side=OrderSide.BUY,
    quantity="100",
    order_type=OrderType.MARKET,
    limit_price=None,
    stop_loss="95",
    source="momentum",
    instrument=AAPL,
) -> OrderIntent:
    return OrderIntent(
        instrument=instrument,
        side=side,
        quantity=Decimal(quantity),
        order_type=order_type,
        limit_price=Decimal(limit_price) if limit_price else None,
        stop_loss=Decimal(stop_loss) if stop_loss else None,
        source=source,
        strategy=source,
    )


@pytest.fixture
def portfolio():
    pm = PortfolioManager()
    pm.update_account(
        AccountState(
            equity=Decimal("100000"),
            cash=Decimal("100000"),
            buying_power=Decimal("200000"),
        )
    )
    return pm


@pytest.fixture
def kill_switch():
    return KillSwitch()


@pytest.fixture
def halt():
    return TradingHalt()


@pytest.fixture
def engine(portfolio, kill_switch, halt):
    return RiskEngine(
        limits=RiskEngineLimits(),
        portfolio=portfolio,
        kill_switch=kill_switch,
        trading_halt=halt,
    )


# ---- happy path ---------------------------------------------------------


def test_valid_intent_is_approved(engine):
    a = engine.evaluate(intent(), snapshot=snapshot(), prices={})
    assert a.approved, a.summary()
    assert a.approved_quantity > 0


def test_approved_quantity_respects_risk_budget(engine):
    # 0.5% of 100k = $500 risk; stop distance 5 -> 100 shares max
    a = engine.evaluate(intent(quantity="100"), snapshot=snapshot(), prices={})
    assert a.approved
    assert a.approved_quantity == Decimal("100")


# ---- hard blocks ---------------------------------------------------------


def test_kill_switch_blocks_everything(engine, kill_switch):
    kill_switch.activate(KillSwitchTrigger.DAILY_LOSS_LIMIT, "test")
    a = engine.evaluate(intent(), snapshot=snapshot(), prices={})
    assert not a.approved
    assert a.reason is RejectionReason.KILL_SWITCH_ACTIVE


def test_trading_halt_blocks(engine, halt):
    halt.set(HaltReason.RECONCILIATION_PENDING, "startup reconcile")
    a = engine.evaluate(intent(), snapshot=snapshot(), prices={})
    assert not a.approved
    assert a.reason is RejectionReason.TRADING_HALTED


def test_kill_switch_checked_before_anything_else(engine, kill_switch):
    """Even a completely malformed situation must report the kill switch,
    proving it short-circuits first."""
    kill_switch.activate(KillSwitchTrigger.MANUAL)
    a = engine.evaluate(intent(), snapshot=None, prices={})
    assert a.reason is RejectionReason.KILL_SWITCH_ACTIVE


# ---- data quality --------------------------------------------------------


def test_missing_market_data_rejected(engine):
    a = engine.evaluate(intent(), snapshot=None, prices={})
    assert not a.approved
    assert a.reason is RejectionReason.MISSING_MARKET_DATA


def test_stale_market_data_rejected(engine):
    a = engine.evaluate(intent(), snapshot=snapshot(age_seconds=60), prices={})
    assert not a.approved
    assert a.reason is RejectionReason.STALE_MARKET_DATA


def test_wide_spread_rejected(engine):
    a = engine.evaluate(intent(), snapshot=snapshot(bid="98", ask="102"), prices={})
    assert not a.approved
    assert a.reason is RejectionReason.SPREAD_TOO_WIDE


def test_crossed_book_rejected(engine):
    a = engine.evaluate(intent(), snapshot=snapshot(bid="101", ask="99"), prices={})
    assert not a.approved
    assert a.reason is RejectionReason.PRICE_SANITY_FAILED


def test_absurd_limit_price_rejected(engine):
    """A hallucinated or fat-fingered price far from market is caught
    before it reaches the broker."""
    a = engine.evaluate(
        intent(order_type=OrderType.LIMIT, limit_price="1.00", stop_loss=None),
        snapshot=snapshot(),
        prices={},
    )
    assert not a.approved
    assert a.reason is RejectionReason.PRICE_SANITY_FAILED


# ---- risk limits ---------------------------------------------------------


def test_daily_loss_limit_blocks_new_trades(engine, portfolio):
    portfolio.update_account(AccountState(equity=Decimal("97000"), buying_power=Decimal("100000")))
    a = engine.evaluate(intent(), snapshot=snapshot(), prices={})
    assert not a.approved
    assert a.reason is RejectionReason.MAX_DAILY_LOSS_BREACHED


def test_drawdown_limit_blocks(engine, portfolio):
    engine.update_peak_equity(Decimal("120000"))
    portfolio.start_new_session(Decimal("100000"))
    portfolio.update_account(AccountState(equity=Decimal("100000"), buying_power=Decimal("100000")))
    a = engine.evaluate(intent(), snapshot=snapshot(), prices={})
    assert not a.approved
    assert a.reason is RejectionReason.MAX_DRAWDOWN_BREACHED


def test_peak_equity_never_decreases(engine):
    engine.update_peak_equity(Decimal("120000"))
    engine.update_peak_equity(Decimal("90000"))
    assert engine.peak_equity == Decimal("120000")


def test_max_open_positions_blocks_new_symbol(portfolio, kill_switch, halt):
    limits = RiskEngineLimits(max_open_positions=1)
    engine = RiskEngine(
        limits=limits, portfolio=portfolio, kill_switch=kill_switch, trading_halt=halt
    )
    # Seed one open position.
    order = Order(
        intent=OrderIntent(
            instrument=Instrument(symbol="MSFT"),
            side=OrderSide.BUY,
            quantity=Decimal("10"),
            source="test",
        )
    )
    portfolio.apply_fill(
        order,
        Fill(
            fill_id="f",
            order_id=order.order_id,
            timestamp=datetime.now(timezone.utc),
            quantity=Decimal("10"),
            price=Decimal("300"),
        ),
    )
    a = engine.evaluate(intent(), snapshot=snapshot(), prices={"MSFT:SMART:USD": Decimal("300")})
    assert not a.approved
    assert a.reason is RejectionReason.MAX_OPEN_POSITIONS_EXCEEDED


def test_adding_to_existing_position_not_blocked_by_position_count(
    portfolio, kill_switch, halt
):
    limits = RiskEngineLimits(max_open_positions=1)
    engine = RiskEngine(
        limits=limits, portfolio=portfolio, kill_switch=kill_switch, trading_halt=halt
    )
    order = Order(intent=intent(quantity="10"))
    portfolio.apply_fill(
        order,
        Fill(
            fill_id="f",
            order_id=order.order_id,
            timestamp=datetime.now(timezone.utc),
            quantity=Decimal("10"),
            price=Decimal("100"),
        ),
    )
    a = engine.evaluate(intent(quantity="10"), snapshot=snapshot(), prices={AAPL_KEY: Decimal("100")})
    # Not blocked on position count (may pass or fail other checks).
    assert a.reason is not RejectionReason.MAX_OPEN_POSITIONS_EXCEEDED


def test_rate_limit_blocks_burst(portfolio, kill_switch, halt):
    limiter = OrderRateLimiter(max_orders_per_minute=2)
    engine = RiskEngine(
        limits=RiskEngineLimits(),
        portfolio=portfolio,
        kill_switch=kill_switch,
        trading_halt=halt,
        rate_limiter=limiter,
    )
    limiter.record()
    limiter.record()
    a = engine.evaluate(intent(), snapshot=snapshot(), prices={})
    assert not a.approved
    assert a.reason is RejectionReason.MAX_ORDER_RATE_EXCEEDED


def test_position_size_limit_enforced(portfolio, kill_switch, halt):
    limits = RiskEngineLimits(max_position_size=Decimal("0.01"), max_risk_per_trade=Decimal("0.5"))
    engine = RiskEngine(
        limits=limits, portfolio=portfolio, kill_switch=kill_switch, trading_halt=halt
    )
    a = engine.evaluate(intent(quantity="1000"), snapshot=snapshot(), prices={})
    # Sizer caps at 1% of equity = $1000 / $100 = 10 shares.
    assert a.approved
    assert a.approved_quantity == Decimal("10")


def test_gross_exposure_limit_enforced(portfolio, kill_switch, halt):
    limits = RiskEngineLimits(
        max_gross_exposure=Decimal("0.05"),
        max_position_size=Decimal("1.0"),
        max_risk_per_trade=Decimal("0.5"),
    )
    engine = RiskEngine(
        limits=limits, portfolio=portfolio, kill_switch=kill_switch, trading_halt=halt
    )
    a = engine.evaluate(intent(quantity="1000"), snapshot=snapshot(), prices={})
    assert not a.approved
    assert a.reason is RejectionReason.MAX_GROSS_EXPOSURE_EXCEEDED


# ---- fail-closed behaviour -----------------------------------------------


def test_missing_price_for_open_position_fails_closed(engine, portfolio):
    """If we can't compute exposure, we can't approve. Never assume fine."""
    order = Order(
        intent=OrderIntent(
            instrument=Instrument(symbol="TSLA"),
            side=OrderSide.BUY,
            quantity=Decimal("10"),
            source="test",
        )
    )
    portfolio.apply_fill(
        order,
        Fill(
            fill_id="f",
            order_id=order.order_id,
            timestamp=datetime.now(timezone.utc),
            quantity=Decimal("10"),
            price=Decimal("200"),
        ),
    )
    a = engine.evaluate(intent(), snapshot=snapshot(), prices={})  # no TSLA mark
    assert not a.approved
    assert a.reason is RejectionReason.MISSING_MARKET_DATA


def test_risk_decision_defaults_to_rejection():
    from risk.decisions import RiskDecision

    assert RiskDecision().approved is False


def test_assessment_defaults_to_rejection():
    from risk.decisions import RiskAssessment

    assert RiskAssessment().approved is False


# ---- AI cannot bypass ----------------------------------------------------


def test_ai_sourced_intent_gets_identical_treatment(engine, kill_switch):
    """An intent claiming to come from the AI layer is evaluated by the
    exact same checks. Source is recorded for audit, never privileged."""
    kill_switch.activate(KillSwitchTrigger.MANUAL)
    ai_intent = intent(source="ai")
    strategy_intent = intent(source="momentum")
    a1 = engine.evaluate(ai_intent, snapshot=snapshot(), prices={})
    a2 = engine.evaluate(strategy_intent, snapshot=snapshot(), prices={})
    assert a1.reason == a2.reason == RejectionReason.KILL_SWITCH_ACTIVE


def test_engine_exposes_no_limit_mutation_api(engine):
    """There must be no setter an AI-driven code path could call to raise
    limits at runtime."""
    forbidden = [
        name
        for name in dir(engine)
        if name.startswith(("set_limit", "update_limit", "override", "disable"))
    ]
    assert forbidden == []


def test_limits_object_has_no_dict_for_mutation():
    limits = RiskEngineLimits()
    # __slots__ means no __dict__, so attributes can't be injected.
    assert not hasattr(limits, "__dict__")


def test_confidence_is_not_an_input_to_evaluate(engine):
    import inspect

    params = inspect.signature(engine.evaluate).parameters
    assert "confidence" not in params
    assert "override" not in params
    assert "force" not in params


# ---- price sanity bands (regression) --------------------------------------


def test_distant_take_profit_is_allowed(engine):
    """A take-profit target is SUPPOSED to be far from the market. The
    narrow executable-price band must not apply to it, or legitimate
    trades — and worse, legitimate exits — get rejected."""
    a = engine.evaluate(
        intent(stop_loss="97", quantity="10"),
        snapshot=snapshot(),
        prices={},
    )
    assert a.approved

    far_target = OrderIntent(
        instrument=AAPL,
        side=OrderSide.BUY,
        quantity=Decimal("10"),
        stop_loss=Decimal("97"),
        take_profit=Decimal("130"),  # 30% away: a real target, not a typo
        source="test",
    )
    b = engine.evaluate(far_target, snapshot=snapshot(), prices={})
    assert b.reason is not RejectionReason.PRICE_SANITY_FAILED


def test_absurd_take_profit_still_rejected(engine):
    """The wider band must still catch genuinely nonsensical targets."""
    absurd = OrderIntent(
        instrument=AAPL,
        side=OrderSide.BUY,
        quantity=Decimal("10"),
        stop_loss=Decimal("97"),
        take_profit=Decimal("5000"),
        source="test",
    )
    a = engine.evaluate(absurd, snapshot=snapshot(), prices={})
    assert not a.approved
    assert a.reason is RejectionReason.PRICE_SANITY_FAILED


def test_distant_stop_loss_still_rejected(engine):
    """A stop loss far from the market is not protecting anything, so the
    narrow band still applies to it."""
    far_stop = OrderIntent(
        instrument=AAPL,
        side=OrderSide.BUY,
        quantity=Decimal("10"),
        stop_loss=Decimal("40"),  # 60% away
        source="test",
    )
    a = engine.evaluate(far_stop, snapshot=snapshot(), prices={})
    assert not a.approved
    assert a.reason is RejectionReason.PRICE_SANITY_FAILED
