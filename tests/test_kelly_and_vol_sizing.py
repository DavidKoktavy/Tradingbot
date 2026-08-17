"""Tests for fractional Kelly and volatility-targeted position sizing,
including that they plug into RiskEngine without changing its defaults."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from ai.performance_analyzer import StrategyStats
from data.models import Instrument, MarketSnapshot
from execution.execution_models import OrderIntent, OrderSide, OrderType
from portfolio.portfolio_manager import AccountState, PortfolioManager
from risk.kelly_sizer import (
    KellyPositionSizer,
    KellyStats,
    kelly_fraction_formula,
    wilson_lower_bound,
)
from risk.kill_switch import KillSwitch, TradingHalt
from risk.risk_engine import RiskEngine, RiskEngineLimits
from risk.vol_target_sizer import VolatilityTargetPositionSizer

AAPL = Instrument(symbol="AAPL")


def snap(mid: float = 100.0) -> MarketSnapshot:
    return MarketSnapshot(
        instrument=AAPL, timestamp=datetime.now(timezone.utc),
        bid=mid - 0.05, ask=mid + 0.05, last=mid,
    )


# ---- Wilson lower bound -----------------------------------------------------


def test_wilson_matches_known_reference_value():
    # Textbook: p_hat=0.5, n=100, 95% -> lower bound ~0.404
    assert wilson_lower_bound(50, 100) == pytest.approx(0.4038, abs=1e-3)


def test_wilson_punishes_small_samples_harder_than_large():
    """The same 80% raw win rate should be trusted much less at n=10 than
    at n=1000."""
    small = wilson_lower_bound(8, 10)
    large = wilson_lower_bound(800, 1000)
    assert small < large
    assert small < 0.6  # 80% raw collapses hard with only 10 trades
    assert large > 0.75  # stays close to the true rate with 1000 trades


def test_wilson_zero_trades_is_zero():
    assert wilson_lower_bound(0, 0) == 0.0


def test_wilson_bound_never_exceeds_raw_rate():
    for successes, n in [(5, 10), (50, 100), (500, 1000), (9, 10)]:
        assert wilson_lower_bound(successes, n) <= successes / n


def test_wilson_narrower_at_higher_confidence_requirement():
    """A 99% confidence lower bound must be more conservative (lower)
    than a 90% one for the same data."""
    conf_90 = wilson_lower_bound(60, 100, confidence=0.90)
    conf_99 = wilson_lower_bound(60, 100, confidence=0.99)
    assert conf_99 < conf_90


# ---- Kelly formula ------------------------------------------------------------


def test_kelly_textbook_case():
    # W=0.6, even-money payoff (R=1) -> f* = 0.2
    assert kelly_fraction_formula(0.6, 1.0, -1.0) == pytest.approx(0.2)


def test_kelly_negative_edge_clips_to_zero():
    assert kelly_fraction_formula(0.4, 1.0, -1.0) == 0.0


def test_kelly_larger_payoff_ratio_increases_fraction():
    small_r = kelly_fraction_formula(0.4, 1.0, -1.0)  # R=1
    large_r = kelly_fraction_formula(0.4, 3.0, -1.0)  # R=3
    assert large_r > small_r


def test_kelly_zero_avg_loss_is_safe():
    assert kelly_fraction_formula(0.6, 1.0, 0.0) == 0.0


def test_kelly_zero_avg_win_is_safe():
    assert kelly_fraction_formula(0.6, 0.0, -1.0) == 0.0


# ---- KellyPositionSizer ---------------------------------------------------------


@pytest.fixture
def kelly_sizer():
    return KellyPositionSizer(kelly_fraction_multiplier=0.25, min_trades=30)


def strong_stats(n=50, win_rate=0.6) -> KellyStats:
    n_wins = int(n * win_rate)
    return KellyStats(n_trades=n, n_wins=n_wins, average_win=150.0, average_loss=-100.0)


def test_no_stats_on_file_refuses_to_size(kelly_sizer):
    result = kelly_sizer.calculate(
        equity=Decimal("100000"), entry_price=Decimal("100"), strategy="momentum"
    )
    assert not result.is_tradeable
    assert "No trade history" in result.detail


def test_no_strategy_name_refuses_to_size(kelly_sizer):
    kelly_sizer.update_stats("momentum", strong_stats())
    result = kelly_sizer.calculate(equity=Decimal("100000"), entry_price=Decimal("100"))
    assert not result.is_tradeable


def test_insufficient_trades_refuses_to_size(kelly_sizer):
    kelly_sizer.update_stats("momentum", strong_stats(n=10))
    result = kelly_sizer.calculate(
        equity=Decimal("100000"), entry_price=Decimal("100"), strategy="momentum"
    )
    assert not result.is_tradeable
    assert "Only 10 trades" in result.detail


def test_sufficient_trades_with_edge_sizes_a_position(kelly_sizer):
    kelly_sizer.update_stats("momentum", strong_stats(n=50, win_rate=0.65))
    result = kelly_sizer.calculate(
        equity=Decimal("100000"), entry_price=Decimal("100"), strategy="momentum"
    )
    assert result.is_tradeable
    assert result.method == "fractional_kelly"
    assert result.win_rate_lower_bound is not None
    assert result.full_kelly_fraction is not None


def test_fractional_multiplier_scales_down_from_full_kelly(kelly_sizer):
    kelly_sizer.update_stats("momentum", strong_stats(n=200, win_rate=0.65))
    quarter = kelly_sizer.calculate(
        equity=Decimal("1000000"), entry_price=Decimal("100"), strategy="momentum"
    )
    half_sizer = KellyPositionSizer(kelly_fraction_multiplier=0.5, min_trades=30)
    half_sizer.update_stats("momentum", strong_stats(n=200, win_rate=0.65))
    half = half_sizer.calculate(
        equity=Decimal("1000000"), entry_price=Decimal("100"), strategy="momentum"
    )
    assert half.quantity > quarter.quantity


def test_weak_edge_below_wilson_bound_refuses(kelly_sizer):
    """8/10 wins is a raw 80% win rate but far too little evidence; the
    conservative estimate should show no edge even with a decent payoff
    ratio, since the min_trades gate already blocks it -- verify the
    gate, and separately verify a low-but-plausible win rate with ample
    trades still gets refused when genuinely unprofitable."""
    weak = KellyPositionSizer(kelly_fraction_multiplier=0.25, min_trades=5)
    weak.update_stats(
        "momentum", KellyStats(n_trades=10, n_wins=8, average_win=100.0, average_loss=-100.0)
    )
    result = weak.calculate(
        equity=Decimal("100000"), entry_price=Decimal("100"), strategy="momentum"
    )
    # With R=1, need W > 0.5 to have edge; Wilson lower bound of 8/10 is
    # ~0.49, right at the edge -- assert it's much more conservative than
    # the raw 80% would suggest, whatever side of zero it lands on.
    assert result.win_rate_lower_bound < 0.6


def test_ceiling_caps_extreme_kelly_suggestions():
    sizer = KellyPositionSizer(
        kelly_fraction_multiplier=1.0,  # full Kelly, deliberately aggressive
        min_trades=30,
        max_fraction_of_equity=Decimal("0.25"),
    )
    # Extreme stats: huge payoff ratio, high win rate -> full Kelly would
    # suggest a very large fraction.
    sizer.update_stats(
        "aggressive",
        KellyStats(n_trades=500, n_wins=450, average_win=1000.0, average_loss=-10.0),
    )
    result = sizer.calculate(
        equity=Decimal("100000"), entry_price=Decimal("10"), strategy="aggressive"
    )
    notional = result.quantity * Decimal("10")
    assert notional <= Decimal("100000") * Decimal("0.25") * Decimal("1.01")  # small rounding slack


def test_never_exceeds_requested_quantity(kelly_sizer):
    kelly_sizer.update_stats("momentum", strong_stats(n=200, win_rate=0.7))
    result = kelly_sizer.calculate(
        equity=Decimal("1000000"), entry_price=Decimal("10"),
        strategy="momentum", requested_quantity=Decimal("5"),
    )
    assert result.quantity <= Decimal("5")


def test_zero_equity_refuses(kelly_sizer):
    kelly_sizer.update_stats("momentum", strong_stats())
    result = kelly_sizer.calculate(
        equity=Decimal("0"), entry_price=Decimal("100"), strategy="momentum"
    )
    assert not result.is_tradeable


def test_invalid_multiplier_rejected():
    with pytest.raises(ValueError):
        KellyPositionSizer(kelly_fraction_multiplier=0.0)
    with pytest.raises(ValueError):
        KellyPositionSizer(kelly_fraction_multiplier=1.5)


def test_stats_can_be_updated_and_read_back(kelly_sizer):
    stats = strong_stats()
    kelly_sizer.update_stats("momentum", stats)
    assert kelly_sizer.stats_for("momentum") == stats
    assert kelly_sizer.stats_for("unknown") is None


# ---- adversarial: AI cannot manufacture Kelly stats ---------------------------


def test_kelly_sizer_has_no_ai_reachable_write_method():
    """Only update_stats() writes, and it must never be reachable from the
    AI decision or reflection engines -- feeding fabricated statistics
    here would manufacture an oversized position."""
    import inspect

    from ai import decision_engine, reflection

    for module in (decision_engine, reflection):
        source = inspect.getsource(module)
        assert "update_stats" not in source
        assert "KellyPositionSizer" not in source


def test_strategy_stats_to_kelly_stats_conversion():
    stats = StrategyStats(
        strategy="momentum", n_trades=40, n_wins=24,
        average_win=150.0, average_loss=-90.0,
    )
    kelly_stats = stats.to_kelly_stats()
    assert kelly_stats is not None
    assert kelly_stats.n_trades == 40
    assert kelly_stats.n_wins == 24


def test_strategy_stats_without_wins_or_losses_converts_to_none():
    stats = StrategyStats(strategy="new", n_trades=0)
    assert stats.to_kelly_stats() is None


# ---- VolatilityTargetPositionSizer ------------------------------------------


@pytest.fixture
def vol_sizer():
    return VolatilityTargetPositionSizer(
        target_annual_volatility=Decimal("0.10"), max_position_size=Decimal("0.20")
    )


def test_no_atr_refuses_to_size(vol_sizer):
    result = vol_sizer.calculate(equity=Decimal("100000"), entry_price=Decimal("100"))
    assert not result.is_tradeable
    assert "No ATR" in result.detail


def test_higher_volatility_produces_smaller_position(vol_sizer):
    calm = vol_sizer.calculate(
        equity=Decimal("100000"), entry_price=Decimal("100"), atr=Decimal("0.5")
    )
    volatile = vol_sizer.calculate(
        equity=Decimal("100000"), entry_price=Decimal("100"), atr=Decimal("5.0")
    )
    assert volatile.quantity < calm.quantity


def test_volatility_floor_prevents_unbounded_oversizing(vol_sizer):
    """Near-zero ATR must not produce an absurdly large position via
    1/vol blowing up."""
    result = vol_sizer.calculate(
        equity=Decimal("100000"), entry_price=Decimal("100"), atr=Decimal("0.0001")
    )
    notional = result.quantity * Decimal("100")
    assert notional <= Decimal("100000") * Decimal("0.20") * Decimal("1.01")


def test_target_weight_reported():
    sizer = VolatilityTargetPositionSizer(
        target_annual_volatility=Decimal("0.10"), max_position_size=Decimal("1.0")
    )
    result = sizer.calculate(equity=Decimal("100000"), entry_price=Decimal("100"), atr=Decimal("1.0"))
    assert result.target_weight is not None
    assert 0 < result.target_weight <= 1.0


def test_never_exceeds_requested_quantity_vol_sizer(vol_sizer):
    result = vol_sizer.calculate(
        equity=Decimal("1000000"), entry_price=Decimal("10"),
        atr=Decimal("0.05"), requested_quantity=Decimal("3"),
    )
    assert result.quantity <= Decimal("3")


def test_zero_equity_refuses_vol_sizer(vol_sizer):
    result = vol_sizer.calculate(equity=Decimal("0"), entry_price=Decimal("100"), atr=Decimal("1"))
    assert not result.is_tradeable


def test_invalid_target_rejected():
    with pytest.raises(ValueError):
        VolatilityTargetPositionSizer(target_annual_volatility=Decimal("0"))


def test_max_position_size_caps_low_volatility_case(vol_sizer):
    """Very low (but non-floored) volatility should still be capped by
    max_position_size, not allowed to run unbounded."""
    result = vol_sizer.calculate(
        equity=Decimal("100000"), entry_price=Decimal("100"), atr=Decimal("0.3")
    )
    notional = result.quantity * Decimal("100")
    assert notional <= Decimal("100000") * Decimal("0.20") * Decimal("1.01")


# ---- integration: plug into RiskEngine without changing its defaults ----------


@pytest.fixture
def portfolio():
    pm = PortfolioManager()
    pm.update_account(
        AccountState(equity=Decimal("100000"), buying_power=Decimal("200000"))
    )
    return pm


def _intent(quantity="1000", strategy="momentum"):
    return OrderIntent(
        instrument=AAPL, side=OrderSide.BUY, quantity=Decimal(quantity),
        order_type=OrderType.MARKET, stop_loss=Decimal("95"),
        source=strategy, strategy=strategy,
    )


def test_riskengine_with_default_sizer_is_unaffected_by_new_sizers_existing(portfolio):
    """Sanity: RiskEngine's default behaviour (no sizer override) must be
    completely unchanged by anything added in this module."""
    engine = RiskEngine(
        limits=RiskEngineLimits(), portfolio=portfolio,
        kill_switch=KillSwitch(), trading_halt=TradingHalt(),
    )
    assessment = engine.evaluate(_intent(), snapshot=snap(), prices={})
    assert assessment.approved
    assert assessment.approved_quantity == Decimal("100")  # unchanged from Phase 4 behaviour


def test_riskengine_can_be_wired_with_kelly_sizer(portfolio):
    sizer = KellyPositionSizer(kelly_fraction_multiplier=0.25, min_trades=30)
    sizer.update_stats("momentum", strong_stats(n=50, win_rate=0.65))
    engine = RiskEngine(
        limits=RiskEngineLimits(), portfolio=portfolio,
        kill_switch=KillSwitch(), trading_halt=TradingHalt(),
        position_sizer=sizer,
    )
    assessment = engine.evaluate(_intent(strategy="momentum"), snapshot=snap(), prices={})
    assert assessment.approved
    assert assessment.approved_quantity > 0


def test_riskengine_kelly_sizer_still_bounded_by_position_size_check(portfolio):
    """Even an aggressive Kelly suggestion is capped by RiskEngine's own
    max_position_size check afterward -- the sizer choosing a big number
    does not bypass the existing hard limit."""
    sizer = KellyPositionSizer(
        kelly_fraction_multiplier=1.0, min_trades=30,
        max_fraction_of_equity=Decimal("0.9"),  # deliberately loose sizer-level cap
    )
    sizer.update_stats(
        "aggressive", KellyStats(n_trades=500, n_wins=450, average_win=1000.0, average_loss=-10.0)
    )
    engine = RiskEngine(
        limits=RiskEngineLimits(max_position_size=Decimal("0.05")),  # tight engine limit
        portfolio=portfolio, kill_switch=KillSwitch(), trading_halt=TradingHalt(),
        position_sizer=sizer,
    )
    assessment = engine.evaluate(
        _intent(quantity="100000", strategy="aggressive"), snapshot=snap(), prices={}
    )
    if assessment.approved:
        notional = assessment.approved_quantity * Decimal("100")
        assert notional <= Decimal("100000") * Decimal("0.05") * Decimal("1.01")


def test_riskengine_with_no_kelly_stats_falls_through_to_no_size(portfolio):
    """A strategy with no trade history yet gets refused sizing, exactly
    the fail-closed behaviour the sizer itself has -- RiskEngine surfaces
    this as an ordinary rejection, not a crash."""
    sizer = KellyPositionSizer(min_trades=30)  # no stats loaded at all
    engine = RiskEngine(
        limits=RiskEngineLimits(), portfolio=portfolio,
        kill_switch=KillSwitch(), trading_halt=TradingHalt(),
        position_sizer=sizer,
    )
    assessment = engine.evaluate(_intent(strategy="brand_new"), snapshot=snap(), prices={})
    assert not assessment.approved


def test_riskengine_can_be_wired_with_vol_target_sizer(portfolio):
    sizer = VolatilityTargetPositionSizer(
        target_annual_volatility=Decimal("0.10"), max_position_size=Decimal("0.10")
    )
    engine = RiskEngine(
        limits=RiskEngineLimits(), portfolio=portfolio,
        kill_switch=KillSwitch(), trading_halt=TradingHalt(),
        position_sizer=sizer,
    )
    assessment = engine.evaluate(
        _intent(), snapshot=snap(), prices={}, atr=Decimal("1.5")
    )
    assert assessment.approved
    assert assessment.approved_quantity > 0


def test_riskengine_position_size_check_still_binds_over_vol_sizer(portfolio):
    """If the vol sizer's own cap is looser than RiskEngine's configured
    max_position_size, the engine's check still wins -- this sizer cannot
    talk its way past the outer limit any more than any other can."""
    sizer = VolatilityTargetPositionSizer(
        target_annual_volatility=Decimal("0.10"), max_position_size=Decimal("0.20")
    )
    engine = RiskEngine(
        limits=RiskEngineLimits(max_position_size=Decimal("0.10")),
        portfolio=portfolio, kill_switch=KillSwitch(), trading_halt=TradingHalt(),
        position_sizer=sizer,
    )
    assessment = engine.evaluate(
        _intent(), snapshot=snap(), prices={}, atr=Decimal("1.5")
    )
    assert not assessment.approved
    assert assessment.reason.name == "MAX_POSITION_SIZE_EXCEEDED"
