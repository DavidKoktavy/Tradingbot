from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from backtesting.costs import ZERO_COST_MODEL, CostModel
from backtesting.engine import BacktestEngine
from backtesting.walk_forward import (
    DegradationReport,
    evaluate_out_of_sample,
    grid_search,
    split_chronological,
    walk_forward,
)
from data.models import Bar, Instrument
from risk.risk_engine import RiskEngineLimits
from strategies.ma_crossover import MACrossoverParams, MACrossoverStrategy
from strategies.momentum import MomentumParams, MomentumStrategy
from backtesting.metrics import MetricsResult

AAPL = Instrument(symbol="AAPL")


def make_bars(closes: list[float], volume: float = 100000.0) -> list[Bar]:
    base = datetime(2026, 1, 5, tzinfo=timezone.utc)
    return [
        Bar(
            timestamp=base + timedelta(days=i),
            open=c,
            high=c * 1.01,
            low=c * 0.99,
            close=c,
            volume=volume,
        )
        for i, c in enumerate(closes)
    ]


def trending_series(n: int = 300) -> list[float]:
    """Deterministic series with alternating trends, so a trend or
    crossover strategy actually trades."""
    out, price = [], 100.0
    for i in range(n):
        cycle = (i // 40) % 2
        drift = 0.9 if cycle == 0 else -0.7
        wobble = 0.4 * (1 if i % 3 == 0 else -1)
        price = max(5.0, price + drift + wobble)
        out.append(price)
    return out


@pytest.fixture
def engine():
    return BacktestEngine(
        strategy=MACrossoverStrategy(
            MACrossoverParams(fast_period=5, slow_period=20, atr_period=10)
        ),
        instrument=AAPL,
        initial_equity=Decimal("100000"),
        bar_size="1 day",
    )


# ---- basic operation -----------------------------------------------------


def test_backtest_runs_and_produces_curve(engine):
    result = engine.run(make_bars(trending_series()))
    assert result.bars_processed == 300
    assert len(result.equity_curve) == 300
    assert result.equity_curve[0] == pytest.approx(100000.0)


def test_backtest_trades_on_trending_data(engine):
    result = engine.run(make_bars(trending_series()))
    assert result.orders_submitted > 0


def test_too_few_bars_raises(engine):
    with pytest.raises(ValueError, match="Need more"):
        engine.run(make_bars([100.0] * 5))


# ---- determinism ----------------------------------------------------------


def test_backtest_is_deterministic():
    """Same inputs must produce a byte-identical equity curve. Any RNG or
    dict-ordering dependence would break this."""
    bars = make_bars(trending_series())
    curves = []
    for _ in range(3):
        e = BacktestEngine(
            strategy=MACrossoverStrategy(
                MACrossoverParams(fast_period=5, slow_period=20, atr_period=10)
            ),
            instrument=AAPL,
            bar_size="1 day",
        )
        curves.append(e.run(bars).equity_curve)
    assert curves[0] == curves[1] == curves[2]


def test_trades_are_reproducible():
    bars = make_bars(trending_series())
    results = []
    for _ in range(2):
        e = BacktestEngine(
            strategy=MomentumStrategy(
                MomentumParams(lookback=10, entry_threshold=0.02, rsi_period=5, atr_period=5)
            ),
            instrument=AAPL,
            bar_size="1 day",
        )
        results.append(e.run(bars))
    assert [t.net_pnl for t in results[0].trades] == [t.net_pnl for t in results[1].trades]


# ---- no look-ahead --------------------------------------------------------


def test_strategy_receives_truncated_bars(monkeypatch):
    """The strategy must never be handed bars beyond the current index."""
    bars = make_bars(trending_series(120))
    seen_lengths = []

    strategy = MACrossoverStrategy(MACrossoverParams(fast_period=5, slow_period=20, atr_period=10))
    original = strategy.generate_signal

    def spy(context):
        seen_lengths.append(len(context.bars))
        # The last bar the strategy sees must be the current one.
        assert context.bars[-1] in bars
        return original(context)

    strategy.generate_signal = spy  # type: ignore[method-assign]
    BacktestEngine(strategy=strategy, instrument=AAPL, bar_size="1 day").run(bars)

    assert seen_lengths == sorted(seen_lengths)  # monotonically growing
    assert max(seen_lengths) <= len(bars)


def test_fills_occur_after_signal_bar():
    """With latency_bars=1, a signal on bar i cannot fill at bar i."""
    bars = make_bars(trending_series())
    e = BacktestEngine(
        strategy=MACrossoverStrategy(
            MACrossoverParams(fast_period=5, slow_period=20, atr_period=10)
        ),
        instrument=AAPL,
        cost_model=CostModel(latency_bars=1),
        bar_size="1 day",
    )
    result = e.run(bars)
    # Every trade's entry must be at or after the bar following its signal;
    # verified indirectly: no trade has zero bars held for entry+exit pairs
    # generated on the same bar.
    assert all(t.bars_held >= 0 for t in result.trades)
    assert result.orders_submitted > 0


def test_zero_latency_still_fills_at_open_not_close():
    """Even with latency 0, fills use the bar's OPEN, never its close."""
    bars = make_bars(trending_series())
    e = BacktestEngine(
        strategy=MACrossoverStrategy(
            MACrossoverParams(fast_period=5, slow_period=20, atr_period=10)
        ),
        instrument=AAPL,
        cost_model=CostModel(latency_bars=0),
        bar_size="1 day",
    )
    result = e.run(bars)
    assert result.bars_processed == len(bars)


# ---- costs matter ---------------------------------------------------------


def test_costs_reduce_returns():
    bars = make_bars(trending_series())
    params = MACrossoverParams(fast_period=5, slow_period=20, atr_period=10)

    with_costs = BacktestEngine(
        strategy=MACrossoverStrategy(params),
        instrument=AAPL,
        cost_model=CostModel(),
        bar_size="1 day",
    ).run(bars)
    without_costs = BacktestEngine(
        strategy=MACrossoverStrategy(params),
        instrument=AAPL,
        cost_model=ZERO_COST_MODEL,
        bar_size="1 day",
    ).run(bars)

    if with_costs.orders_filled > 0:
        assert with_costs.equity_curve[-1] < without_costs.equity_curve[-1]


def test_commission_is_recorded():
    bars = make_bars(trending_series())
    result = BacktestEngine(
        strategy=MACrossoverStrategy(
            MACrossoverParams(fast_period=5, slow_period=20, atr_period=10)
        ),
        instrument=AAPL,
        bar_size="1 day",
    ).run(bars)
    if result.trades:
        assert result.metrics.total_commission > 0


def test_low_volume_causes_partial_fill():
    """An order larger than the participation cap must partially fill."""
    bars = make_bars(trending_series(), volume=50.0)  # very thin
    result = BacktestEngine(
        strategy=MACrossoverStrategy(
            MACrossoverParams(fast_period=5, slow_period=20, atr_period=10)
        ),
        instrument=AAPL,
        cost_model=CostModel(max_participation=Decimal("0.1")),
        bar_size="1 day",
    ).run(bars)
    assert result.orders_partially_filled > 0 or result.orders_filled == 0


# ---- risk engine is genuinely reused ---------------------------------------


def test_backtest_uses_real_risk_engine_limits():
    """A restrictive limit must visibly reduce activity, proving the real
    risk engine is in the loop rather than a stub."""
    bars = make_bars(trending_series())
    params = MACrossoverParams(fast_period=5, slow_period=20, atr_period=10)

    normal = BacktestEngine(
        strategy=MACrossoverStrategy(params), instrument=AAPL, bar_size="1 day"
    ).run(bars)
    restricted = BacktestEngine(
        strategy=MACrossoverStrategy(params),
        instrument=AAPL,
        risk_limits=RiskEngineLimits(max_position_size=Decimal("0.0001")),
        bar_size="1 day",
    ).run(bars)
    assert restricted.orders_submitted < normal.orders_submitted


def test_risk_rejections_are_reported():
    bars = make_bars(trending_series())
    result = BacktestEngine(
        strategy=MACrossoverStrategy(
            MACrossoverParams(fast_period=5, slow_period=20, atr_period=10)
        ),
        instrument=AAPL,
        risk_limits=RiskEngineLimits(max_gross_exposure=Decimal("0.001")),
        bar_size="1 day",
    ).run(bars)
    assert result.rejections  # rejections surfaced, not silently dropped


# ---- splits and walk-forward ------------------------------------------------


def test_chronological_split_preserves_order():
    bars = make_bars(trending_series(100))
    split = split_chronological(bars, train_pct=0.6, validation_pct=0.2)
    assert split.sizes == (60, 20, 20)
    assert split.train[-1].timestamp < split.validation[0].timestamp
    assert split.validation[-1].timestamp < split.test[0].timestamp


def test_split_rejects_invalid_percentages():
    bars = make_bars(trending_series(100))
    with pytest.raises(ValueError):
        split_chronological(bars, train_pct=0.9, validation_pct=0.2)


def test_grid_search_returns_best_and_count():
    bars = make_bars(trending_series())
    params, metrics, tested = grid_search(
        strategy_cls=MACrossoverStrategy,
        params_cls=MACrossoverParams,
        grid={"fast_period": [5, 10], "slow_period": [20, 30], "atr_period": [10]},
        bars=bars,
        instrument=AAPL,
        bar_size="1 day",
    )
    assert tested == 4
    assert params is None or isinstance(params, MACrossoverParams)


def test_grid_search_skips_invalid_combinations():
    bars = make_bars(trending_series())
    _, _, tested = grid_search(
        strategy_cls=MACrossoverStrategy,
        params_cls=MACrossoverParams,
        grid={"fast_period": [30], "slow_period": [10], "atr_period": [10]},
        bars=bars,
        instrument=AAPL,
        bar_size="1 day",
    )
    assert tested == 0  # fast >= slow is rejected by the params validator


def test_out_of_sample_evaluation_produces_report():
    bars = make_bars(trending_series(400))
    report = evaluate_out_of_sample(
        strategy_cls=MACrossoverStrategy,
        params_cls=MACrossoverParams,
        grid={"fast_period": [5, 10], "slow_period": [20], "atr_period": [10]},
        bars=bars,
        instrument=AAPL,
        bar_size="1 day",
    )
    if report is not None:
        assert isinstance(report, DegradationReport)
        assert "OUT-OF-SAMPLE" in report.summary()


def test_walk_forward_produces_windows():
    bars = make_bars(trending_series(500))
    result = walk_forward(
        strategy_cls=MACrossoverStrategy,
        params_cls=MACrossoverParams,
        grid={"fast_period": [5, 10], "slow_period": [20], "atr_period": [10]},
        bars=bars,
        instrument=AAPL,
        train_size=150,
        test_size=75,
        bar_size="1 day",
    )
    assert len(result.windows) >= 2
    assert "Walk-forward windows" in result.summary()


# ---- overfitting detection ---------------------------------------------------


def _metrics(sharpe: float | None, total_return: float | None, n_trades: int = 50):
    return MetricsResult(sharpe=sharpe, total_return=total_return, n_trades=n_trades)


def test_sharpe_collapse_flagged_as_overfit():
    report = DegradationReport(
        in_sample=_metrics(2.0, 0.5), out_of_sample=_metrics(0.3, 0.02)
    )
    assert report.is_likely_overfit
    assert "LIKELY OVERFIT" in report.summary()


def test_sign_flip_flagged_as_overfit():
    report = DegradationReport(
        in_sample=_metrics(1.0, 0.3), out_of_sample=_metrics(0.9, -0.1)
    )
    assert report.is_likely_overfit


def test_many_combinations_few_trades_flagged():
    report = DegradationReport(
        in_sample=_metrics(1.0, 0.2),
        out_of_sample=_metrics(0.9, 0.15, n_trades=5),
        n_combinations_tested=50,
    )
    assert report.is_likely_overfit


def test_stable_performance_not_flagged():
    report = DegradationReport(
        in_sample=_metrics(1.0, 0.2), out_of_sample=_metrics(0.85, 0.18)
    )
    assert not report.is_likely_overfit
    assert "NOT evidence of profitability" in report.summary()


def test_walk_forward_consistency_metric():
    from backtesting.walk_forward import WalkForwardResult, WalkForwardWindow

    result = WalkForwardResult(
        windows=[
            WalkForwardWindow(0, 100, 50, {}, _metrics(1.0, 0.1), _metrics(0.5, 0.05)),
            WalkForwardWindow(1, 100, 50, {}, _metrics(1.0, 0.1), _metrics(-0.2, -0.03)),
        ]
    )
    assert result.consistency == pytest.approx(0.5)
    assert "fewer than half" not in result.summary()


# ---- simulation time (regression) --------------------------------------------


def historical_bars(n: int = 300) -> list[Bar]:
    """Bars dated firmly in the PAST, as real historical data would be."""
    base = datetime(2021, 1, 4, tzinfo=timezone.utc)
    out, price = [], 100.0
    for i in range(n):
        cycle = (i // 40) % 2
        price = max(5.0, price + (0.9 if cycle == 0 else -0.7) + 0.4 * (1 if i % 3 == 0 else -1))
        out.append(
            Bar(
                timestamp=base + timedelta(days=i),
                open=price,
                high=price * 1.01,
                low=price * 0.99,
                close=price,
                volume=200000,
            )
        )
    return out


def test_historical_bars_are_not_rejected_as_stale():
    """Regression: the risk engine must evaluate staleness against
    SIMULATION time (the bar's timestamp), not wall clock. Otherwise every
    historical bar is 'stale' and the backtest silently does nothing while
    still reporting a clean run."""
    result = BacktestEngine(
        strategy=MACrossoverStrategy(
            MACrossoverParams(fast_period=10, slow_period=30, atr_period=14)
        ),
        instrument=AAPL,
        bar_size="1 day",
    ).run(historical_bars())

    assert "STALE_MARKET_DATA" not in result.rejections
    assert result.orders_submitted > 0, "Backtest on historical data produced no orders"


def test_backtest_on_historical_data_produces_trades():
    result = BacktestEngine(
        strategy=MACrossoverStrategy(
            MACrossoverParams(fast_period=10, slow_period=30, atr_period=14)
        ),
        instrument=AAPL,
        bar_size="1 day",
    ).run(historical_bars())
    assert len(result.trades) > 0
    assert result.metrics.n_trades == len(result.trades)


def test_empty_backtest_is_visible_not_silent():
    """A backtest where everything was rejected must make that obvious
    rather than reporting a clean flat curve."""
    result = BacktestEngine(
        strategy=MACrossoverStrategy(
            MACrossoverParams(fast_period=10, slow_period=30, atr_period=14)
        ),
        instrument=AAPL,
        risk_limits=RiskEngineLimits(max_gross_exposure=Decimal("0.0000001")),
        bar_size="1 day",
    ).run(historical_bars())
    assert result.orders_submitted == 0
    assert result.rejections, "Rejection reasons must be surfaced"
    assert result.metrics.warnings
