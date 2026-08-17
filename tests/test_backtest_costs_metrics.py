from decimal import Decimal

import pytest

from backtesting.costs import ZERO_COST_MODEL, CostModel
from backtesting.metrics import (
    TradeRecord,
    compute_metrics,
    compute_returns,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)
from execution.execution_models import OrderSide


# ---- cost model ----------------------------------------------------------


def test_buy_fills_above_reference():
    costs = CostModel()
    price = costs.fill_price(
        reference_price=Decimal("100"), side=OrderSide.BUY, quantity=Decimal("100")
    )
    assert price > Decimal("100")


def test_sell_fills_below_reference():
    costs = CostModel()
    price = costs.fill_price(
        reference_price=Decimal("100"), side=OrderSide.SELL, quantity=Decimal("100")
    )
    assert price < Decimal("100")


def test_costs_are_adverse_in_both_directions():
    """A round trip at an unchanged price must lose money."""
    costs = CostModel()
    buy = costs.fill_price(
        reference_price=Decimal("100"), side=OrderSide.BUY, quantity=Decimal("100")
    )
    sell = costs.fill_price(
        reference_price=Decimal("100"), side=OrderSide.SELL, quantity=Decimal("100")
    )
    assert sell < buy


def test_larger_orders_pay_more_impact():
    costs = CostModel()
    small = costs.fill_price(
        reference_price=Decimal("100"),
        side=OrderSide.BUY,
        quantity=Decimal("10"),
        bar_volume=Decimal("10000"),
    )
    large = costs.fill_price(
        reference_price=Decimal("100"),
        side=OrderSide.BUY,
        quantity=Decimal("5000"),
        bar_volume=Decimal("10000"),
    )
    assert large > small


def test_commission_respects_minimum():
    costs = CostModel(commission_per_share=Decimal("0.005"), min_commission=Decimal("1.00"))
    assert costs.commission(Decimal("10"), Decimal("100")) == Decimal("1.00")
    assert costs.commission(Decimal("1000"), Decimal("100")) == Decimal("5.00")


def test_participation_cap():
    costs = CostModel(max_participation=Decimal("0.1"))
    assert costs.max_fillable(Decimal("10000")) == Decimal("1000")
    assert costs.max_fillable(None) is None


def test_zero_cost_model_is_frictionless():
    price = ZERO_COST_MODEL.fill_price(
        reference_price=Decimal("100"), side=OrderSide.BUY, quantity=Decimal("100")
    )
    assert price == Decimal("100")


def test_explicit_spread_used_when_available():
    costs = CostModel()
    half = costs.half_spread(Decimal("100"), Decimal("99"), Decimal("101"))
    assert half == Decimal("1")


# ---- metrics -------------------------------------------------------------


def test_returns_computed_pairwise():
    assert compute_returns([100.0, 110.0, 99.0]) == pytest.approx([0.1, -0.1])


def test_max_drawdown_basic():
    dd, duration = max_drawdown([100, 120, 90, 95, 130])
    assert dd == pytest.approx(0.25)  # 120 -> 90
    assert duration > 0


def test_max_drawdown_monotonic_rise_is_zero():
    dd, _ = max_drawdown([100, 110, 120, 130])
    assert dd == 0.0


def test_sharpe_none_for_insufficient_data():
    assert sharpe_ratio([0.01], periods_per_year=252) is None


def test_sharpe_none_for_zero_volatility():
    """Constant returns give zero std. Must be None, not inf."""
    assert sharpe_ratio([0.01] * 20, periods_per_year=252) is None


def test_sortino_none_when_no_downside():
    """No losing periods means the ratio is undefined, not infinite."""
    assert sortino_ratio([0.01, 0.02, 0.03], periods_per_year=252) is None


def test_sortino_penalises_only_downside():
    mixed = [0.02, -0.01, 0.03, -0.005, 0.01, 0.02]
    s = sortino_ratio(mixed, periods_per_year=252)
    assert s is not None


def test_annualisation_uses_supplied_frequency():
    returns = [0.001, -0.0005, 0.002, 0.0015, -0.001] * 10
    daily = sharpe_ratio(returns, periods_per_year=252)
    minute = sharpe_ratio(returns, periods_per_year=252 * 390)
    assert minute > daily  # higher frequency -> larger annualisation factor


def _trade(pnl: str, commission: str = "1") -> TradeRecord:
    return TradeRecord(
        instrument="AAPL",
        strategy="test",
        entry_time="2026-01-01T00:00:00",
        exit_time="2026-01-02T00:00:00",
        side="BUY",
        quantity=Decimal("100"),
        entry_price=Decimal("100"),
        exit_price=Decimal("101"),
        gross_pnl=Decimal(pnl),
        commission=Decimal(commission),
        net_pnl=Decimal(pnl),
        bars_held=5,
    )


def test_trade_statistics():
    trades = [_trade("100"), _trade("-50"), _trade("200"), _trade("-25")]
    m = compute_metrics(
        equity_curve=[100000, 100100, 100050, 100250, 100225],
        trades=trades,
        periods_per_year=252,
    )
    assert m.n_trades == 4
    assert m.win_rate == pytest.approx(0.5)
    assert m.profit_factor == pytest.approx(300 / 75)
    assert m.average_win == pytest.approx(150)
    assert m.average_loss == pytest.approx(-37.5)
    assert m.expectancy == pytest.approx(56.25)


def test_low_trade_count_flagged_as_meaningless():
    m = compute_metrics(
        equity_curve=[100000, 101000], trades=[_trade("100")], periods_per_year=252
    )
    assert not m.is_statistically_meaningful
    assert any("not statistically meaningful" in w for w in m.warnings)


def test_implausible_sharpe_is_flagged():
    # Smooth, strongly positive returns produce an absurd Sharpe.
    curve = [100000 * (1.01**i) for i in range(60)]
    m = compute_metrics(
        equity_curve=curve, trades=[_trade("100") for _ in range(40)], periods_per_year=252
    )
    assert m.sharpe is not None and m.sharpe > 3
    assert any("implausibly high" in w for w in m.warnings)


def test_zero_drawdown_with_trades_is_flagged():
    curve = [100000 + i * 10 for i in range(50)]
    m = compute_metrics(
        equity_curve=curve, trades=[_trade("10") for _ in range(35)], periods_per_year=252
    )
    assert any("Zero drawdown" in w for w in m.warnings)


def test_profit_factor_none_when_no_losses():
    m = compute_metrics(
        equity_curve=[100000, 101000],
        trades=[_trade("100"), _trade("200")],
        periods_per_year=252,
    )
    assert m.profit_factor is None  # division by zero -> None, not inf


def test_empty_equity_curve_handled():
    m = compute_metrics(equity_curve=[], trades=[], periods_per_year=252)
    assert m.total_return is None
    assert m.warnings
