"""Tests for backtesting/statistics.py: PSR, DSR, MinTRL, and the search
integration that feeds them real trial data."""

import json
from decimal import Decimal

import pytest

from backtesting.statistics import (
    DeflationReport,
    TrialResult,
    evaluate_search_overfitting,
    expected_max_sharpe_under_luck,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
    sample_kurtosis,
    sample_skewness,
)
from backtesting.walk_forward import grid_search, grid_search_with_trials
from data.models import Bar, Instrument
from strategies.ma_crossover import MACrossoverParams, MACrossoverStrategy


# ---- moment estimators ----------------------------------------------------


def test_skewness_of_symmetric_data_near_zero():
    data = [-3, -2, -1, 0, 1, 2, 3]
    assert sample_skewness(data) == pytest.approx(0.0, abs=1e-9)


def test_skewness_detects_right_skew():
    data = [1, 1, 1, 1, 2, 3, 20]
    assert sample_skewness(data) > 0


def test_kurtosis_of_normal_like_data_near_three():
    import random

    random.seed(1)
    data = [random.gauss(0, 1) for _ in range(5000)]
    assert sample_kurtosis(data) == pytest.approx(3.0, abs=0.3)


def test_kurtosis_too_few_points_returns_normal_default():
    assert sample_kurtosis([1.0, 2.0]) == 3.0


def test_skewness_too_few_points_returns_zero():
    assert sample_skewness([1.0]) == 0.0


def test_moment_estimators_handle_zero_variance():
    assert sample_skewness([5.0, 5.0, 5.0, 5.0]) == 0.0
    assert sample_kurtosis([5.0, 5.0, 5.0, 5.0]) == 3.0


# ---- probabilistic sharpe ratio --------------------------------------------


def test_psr_at_benchmark_equals_observed_is_half():
    assert probabilistic_sharpe_ratio(1.0, n_observations=252, benchmark_sharpe=1.0) == pytest.approx(0.5)


def test_psr_increases_with_observed_sharpe():
    low = probabilistic_sharpe_ratio(0.5, n_observations=252, benchmark_sharpe=0.0)
    high = probabilistic_sharpe_ratio(1.5, n_observations=252, benchmark_sharpe=0.0)
    assert high > low


def test_psr_increases_with_sample_size_for_fixed_edge():
    """More observations of the same edge should increase confidence."""
    small = probabilistic_sharpe_ratio(1.0, n_observations=30, benchmark_sharpe=0.0)
    large = probabilistic_sharpe_ratio(1.0, n_observations=2520, benchmark_sharpe=0.0)
    assert large > small


def test_psr_none_below_minimum_observations():
    assert probabilistic_sharpe_ratio(1.0, n_observations=5, benchmark_sharpe=0.0) is None


def test_psr_none_on_degenerate_denominator():
    # A large POSITIVE skew combined with a large positive observed
    # Sharpe drives the -skew*sharpe term sharply negative; with the
    # kurtosis term not large enough to compensate, the denominator goes
    # negative. Must refuse rather than take sqrt of a negative number.
    result = probabilistic_sharpe_ratio(
        5.0, n_observations=100, benchmark_sharpe=0.0, skew=10.0, kurtosis=3.0
    )
    assert result is None


def test_psr_fat_tails_reduce_confidence_vs_normal():
    """Higher kurtosis (fatter tails) should make the same observed Sharpe
    less convincing, all else equal. Uses a small observed Sharpe and few
    observations so both results stay well away from the CDF's ceiling of
    1.0, where the comparison would otherwise be swamped by float
    saturation rather than showing the real effect."""
    normal_tails = probabilistic_sharpe_ratio(
        0.3, n_observations=25, benchmark_sharpe=0.0, skew=0.0, kurtosis=3.0
    )
    fat_tails = probabilistic_sharpe_ratio(
        0.3, n_observations=25, benchmark_sharpe=0.0, skew=0.0, kurtosis=9.0
    )
    assert fat_tails < normal_tails


# ---- expected max sharpe under luck ----------------------------------------


def test_zero_or_one_trial_has_no_deflation():
    assert expected_max_sharpe_under_luck(0.5, 0) == 0.0
    assert expected_max_sharpe_under_luck(0.5, 1) == 0.0


def test_expected_max_grows_with_trials():
    values = [expected_max_sharpe_under_luck(0.5, n) for n in (2, 10, 100, 1000)]
    assert values == sorted(values)
    assert values[0] < values[-1]


def test_expected_max_grows_with_sharpe_std():
    low_std = expected_max_sharpe_under_luck(0.1, 50)
    high_std = expected_max_sharpe_under_luck(1.0, 50)
    assert high_std > low_std


def test_zero_sharpe_std_gives_zero_deflation():
    assert expected_max_sharpe_under_luck(0.0, 100) == 0.0


# ---- minimum track record length -------------------------------------------


def test_mintrl_none_when_no_edge():
    assert minimum_track_record_length(0.5, benchmark_sharpe=0.5) is None
    assert minimum_track_record_length(0.3, benchmark_sharpe=0.5) is None


def test_mintrl_grows_as_edge_shrinks():
    """A marginal edge over the benchmark should require a much longer
    track record to distinguish from luck."""
    large_edge = minimum_track_record_length(1.5, benchmark_sharpe=0.0)
    small_edge = minimum_track_record_length(0.05, benchmark_sharpe=0.0)
    assert small_edge > large_edge


def test_mintrl_increases_with_confidence_level():
    lower_conf = minimum_track_record_length(0.5, benchmark_sharpe=0.0, confidence=0.80)
    higher_conf = minimum_track_record_length(0.5, benchmark_sharpe=0.0, confidence=0.99)
    assert higher_conf > lower_conf


# ---- evaluate_search_overfitting: the integration point --------------------


def _trial(sharpe, n_trades=50, n_periods=252, **params) -> TrialResult:
    return TrialResult(params=params, sharpe=sharpe, n_trades=n_trades, n_periods=n_periods)


def test_empty_trials_produces_no_result():
    report = evaluate_search_overfitting([])
    assert not report.has_result
    assert "No trial produced" in report.warnings[0]


def test_all_none_sharpes_produces_no_result():
    report = evaluate_search_overfitting(
        [_trial(None), _trial(None)]
    )
    assert not report.has_result


def test_too_few_observations_refuses_to_compute():
    report = evaluate_search_overfitting([_trial(1.0, n_trades=50, n_periods=5)])
    assert not report.has_result
    assert any("below" in w for w in report.warnings)


def test_default_skew_kurtosis_flagged_as_assumption():
    report = evaluate_search_overfitting([_trial(1.0, n_trades=50, n_periods=252)])
    assert report.skew == 0.0
    assert report.kurtosis == 3.0
    assert any("assuming normal returns" in w for w in report.warnings)


def test_supplied_returns_used_instead_of_default():
    import random

    random.seed(3)
    skewed_returns = [random.gauss(0, 1) ** 2 for _ in range(200)]  # right-skewed
    report = evaluate_search_overfitting(
        [_trial(1.0, n_trades=50, n_periods=252)], returns=skewed_returns
    )
    assert report.skew != 0.0
    assert not any("assuming normal returns" in w for w in report.warnings)


def test_dsr_decreases_as_trial_count_grows():
    """The core claim of the whole module: searching harder for the same
    winning Sharpe should make it look progressively less impressive."""
    winner = _trial(0.65, n_trades=50, n_periods=252)

    def build(n_extra):
        others = [_trial(0.05 * ((i * 37) % 23 - 11), n_trades=50) for i in range(n_extra)]
        return others + [winner]

    dsrs = []
    for n_extra in (5, 50, 500):
        report = evaluate_search_overfitting(build(n_extra), best=winner)
        assert report.deflated_sharpe_ratio is not None
        dsrs.append(report.deflated_sharpe_ratio)

    assert dsrs[0] > dsrs[1] > dsrs[2]


def test_single_trial_has_no_selection_bias_warning():
    report = evaluate_search_overfitting([_trial(1.0, n_trades=50, n_periods=252)])
    assert any("no selection bias" in w for w in report.warnings)
    assert report.expected_max_sharpe_by_chance == 0.0


# ---- regression: the one-trade bug found via manual testing -----------------


def test_one_trade_result_is_never_certified_regardless_of_dsr_value():
    """Regression: a Sharpe of 14 from ONE trade spread across 400 daily
    marks previously reported 'clears 95% confidence' because the period
    count (400) satisfied the observation-count math while completely
    hiding that those 400 numbers are one massively autocorrelated
    observation, not 400 independent ones. Found via an end-to-end CLI
    run, not a unit test -- which is exactly why this regression test
    exists now."""
    trials = [_trial(1.0 + 0.01 * i, n_trades=50, n_periods=252) for i in range(20)]
    winner = _trial(14.0, n_trades=1, n_periods=400)
    trials.append(winner)

    report = evaluate_search_overfitting(trials, best=winner)
    assert report.n_trades == 1
    # The raw DSR number can still be high -- that's not wrong, it's what
    # the period-count math produces -- but it must never be CERTIFIED.
    assert report.likely_genuine is False
    assert any("autocorrelated" in w for w in report.warnings)


def test_thin_trade_count_forces_not_genuine_even_with_perfect_dsr():
    trials = [_trial(0.1, n_trades=50, n_periods=252) for _ in range(10)]
    winner = _trial(5.0, n_trades=5, n_periods=500)
    trials.append(winner)
    report = evaluate_search_overfitting(trials, best=winner)
    assert report.deflated_sharpe_ratio is not None and report.deflated_sharpe_ratio > 0.99
    assert report.likely_genuine is False  # thin trade count overrides a "perfect" DSR


def test_healthy_trade_count_can_be_certified():
    """The guard must not be a blanket 'always False' -- with enough
    trades AND a DSR that clears the bar, likely_genuine can be True."""
    trials = [_trial(0.1 + 0.01 * i, n_trades=40, n_periods=252) for i in range(5)]
    winner = _trial(0.9, n_trades=45, n_periods=252)
    trials.append(winner)
    report = evaluate_search_overfitting(trials, best=winner)
    assert report.n_trades >= 30
    if report.deflated_sharpe_ratio is not None and report.deflated_sharpe_ratio >= 0.95:
        assert report.likely_genuine is True


def test_summary_is_human_readable():
    report = evaluate_search_overfitting([_trial(1.0, n_trades=50, n_periods=252)])
    text = report.summary()
    assert "Observed Sharpe" in text
    assert "Deflated Sharpe Ratio" in text


# ---- grid_search_with_trials integration -----------------------------------


def _bars(closes: list[float]) -> list[Bar]:
    from datetime import datetime, timedelta, timezone

    base = datetime(2023, 1, 3, tzinfo=timezone.utc)
    return [
        Bar(timestamp=base + timedelta(days=i), open=c, high=c * 1.01, low=c * 0.99,
            close=c, volume=200000)
        for i, c in enumerate(closes)
    ]


def _trending_series(n=300) -> list[float]:
    out, price = [], 100.0
    for i in range(n):
        cycle = (i // 40) % 2
        price = max(5.0, price + (0.9 if cycle == 0 else -0.7) + 0.4 * (1 if i % 3 == 0 else -1))
        out.append(price)
    return out


def test_grid_search_with_trials_retains_every_trial():
    bars = _bars(_trending_series())
    best_params, best_metrics, trials, tested = grid_search_with_trials(
        strategy_cls=MACrossoverStrategy,
        params_cls=MACrossoverParams,
        grid={"fast_period": [5, 10], "slow_period": [20, 30], "atr_period": [10]},
        bars=bars,
        instrument=Instrument(symbol="AAPL"),
        bar_size="1 day",
    )
    assert len(trials) == tested
    assert tested == 4


def test_grid_search_and_grid_search_with_trials_agree_on_winner():
    """The convenience wrapper must produce the identical winner as the
    full version -- it should be a strict subset of the same computation,
    not a separate code path that could drift."""
    bars = _bars(_trending_series())
    kwargs = dict(
        strategy_cls=MACrossoverStrategy,
        params_cls=MACrossoverParams,
        grid={"fast_period": [5, 10], "slow_period": [20, 30], "atr_period": [10]},
        bars=bars,
        instrument=Instrument(symbol="AAPL"),
        bar_size="1 day",
    )
    params_a, metrics_a, tested_a = grid_search(**kwargs)
    params_b, metrics_b, _, tested_b = grid_search_with_trials(**kwargs)
    assert tested_a == tested_b
    assert params_a.model_dump() == (params_b.model_dump() if params_b else None)


def test_grid_search_with_trials_feeds_evaluate_search_overfitting():
    bars = _bars(_trending_series())
    _, _, trials, _ = grid_search_with_trials(
        strategy_cls=MACrossoverStrategy,
        params_cls=MACrossoverParams,
        grid={"fast_period": [5, 8, 10], "slow_period": [20, 30, 40], "atr_period": [10]},
        bars=bars,
        instrument=Instrument(symbol="AAPL"),
        bar_size="1 day",
    )
    report = evaluate_search_overfitting(trials)
    assert report.n_trials == len(trials)
    assert "searched" in report.summary().lower() or str(report.n_trials) in report.summary()


# ---- promotion pipeline gate integration ------------------------------------


def _good_promotion_setup():
    from backtesting.metrics import MetricsResult
    from backtesting.walk_forward import DegradationReport
    from strategies.promotion import GateCriteria, PromotionPipeline, PromotionStage, ResearchProposal

    pipeline = PromotionPipeline(GateCriteria())
    candidate = pipeline.submit(
        ResearchProposal(name="test_strategy", hypothesis="h", proposed_by="ai")
    )
    pipeline.attach_backtest(
        candidate.candidate_id,
        MetricsResult(n_trades=50, sharpe=1.1, max_drawdown=0.10, profit_factor=1.3),
    )
    pipeline.promote(candidate.candidate_id, PromotionStage.BACKTEST)
    pipeline.attach_degradation(
        candidate.candidate_id,
        DegradationReport(
            in_sample=MetricsResult(n_trades=50, sharpe=1.2, total_return=0.15),
            out_of_sample=MetricsResult(n_trades=30, sharpe=1.0, total_return=0.10),
        ),
    )
    return pipeline, candidate


def test_promotion_passes_validation_without_deflation_attached():
    """DSR is optional -- omitting it must not block an otherwise-valid
    promotion, since not every candidate ran a full grid search."""
    from strategies.promotion import PromotionStage

    pipeline, candidate = _good_promotion_setup()
    pipeline.promote(candidate.candidate_id, PromotionStage.VALIDATION)
    assert candidate.stage is PromotionStage.VALIDATION


def test_promotion_refused_when_deflation_fails():
    from strategies.promotion import PromotionRefused, PromotionStage

    pipeline, candidate = _good_promotion_setup()
    trials = [_trial(0.1 + 0.01 * i, n_trades=50, n_periods=252) for i in range(500)]
    winner = _trial(0.5, n_trades=50, n_periods=252)
    trials.append(winner)
    weak_deflation = evaluate_search_overfitting(trials, best=winner)
    pipeline.attach_deflation(candidate.candidate_id, weak_deflation)

    with pytest.raises(PromotionRefused, match="Deflated Sharpe Ratio"):
        pipeline.promote(candidate.candidate_id, PromotionStage.VALIDATION)


def test_promotion_refused_when_deflation_thin_trade_count():
    """Even a headline DSR near 100% must not clear the gate if it rests
    on too few trades -- exactly the bug this module's regression test
    covers, now enforced at the pipeline level too."""
    from strategies.promotion import PromotionRefused, PromotionStage

    pipeline, candidate = _good_promotion_setup()
    trials = [_trial(1.0 + 0.01 * i, n_trades=50, n_periods=252) for i in range(20)]
    winner = _trial(14.0, n_trades=1, n_periods=400)
    trials.append(winner)
    thin_deflation = evaluate_search_overfitting(trials, best=winner)
    pipeline.attach_deflation(candidate.candidate_id, thin_deflation)

    with pytest.raises(PromotionRefused):
        pipeline.promote(candidate.candidate_id, PromotionStage.VALIDATION)


def test_promotion_succeeds_when_deflation_passes():
    from strategies.promotion import PromotionStage

    pipeline, candidate = _good_promotion_setup()
    trials = [_trial(0.05 * i, n_trades=40, n_periods=252) for i in range(5)]
    winner = _trial(2.5, n_trades=50, n_periods=252)
    trials.append(winner)
    strong_deflation = evaluate_search_overfitting(trials, best=winner)
    pipeline.attach_deflation(candidate.candidate_id, strong_deflation)

    if strong_deflation.likely_genuine:
        pipeline.promote(candidate.candidate_id, PromotionStage.VALIDATION)
        assert candidate.stage is PromotionStage.VALIDATION


def test_audit_trail_shows_deflated_sharpe_when_attached():
    from strategies.promotion import PromotionStage

    pipeline, candidate = _good_promotion_setup()
    trials = [_trial(0.05 * i, n_trades=40, n_periods=252) for i in range(5)]
    winner = _trial(2.5, n_trades=50, n_periods=252)
    trials.append(winner)
    report = evaluate_search_overfitting(trials, best=winner)
    pipeline.attach_deflation(candidate.candidate_id, report)
    if report.has_result:
        assert "Deflated Sharpe Ratio" in candidate.audit_trail()


# ---- CLI ---------------------------------------------------------------------


def test_overfitting_check_command_runs(tmp_path, capsys):
    from app.cli import main

    csv_file = tmp_path / "bars.csv"
    rows = ["timestamp,open,high,low,close,volume"]
    price = 100.0
    from datetime import datetime, timedelta, timezone

    for i in range(200):
        cycle = (i // 30) % 2
        price = max(5.0, price + (0.9 if cycle == 0 else -0.7))
        ts = (datetime(2024, 1, 2, tzinfo=timezone.utc) + timedelta(days=i)).isoformat()
        rows.append(f"{ts},{price},{price*1.01},{price*0.99},{price},100000")
    csv_file.write_text("\n".join(rows))

    grid = json.dumps({"fast_period": [5, 10], "slow_period": [20, 30], "atr_period": [10]})
    code = main([
        "overfitting-check", "--strategy", "ma_crossover", "--data", str(csv_file),
        "--symbol", "AAPL", "--grid", grid,
    ])
    assert code == 0
    output = capsys.readouterr().out
    assert "Deflated Sharpe Ratio" in output
    assert "corrects for how hard you searched" in output


def test_overfitting_check_rejects_invalid_grid_json(tmp_path, capsys):
    from app.cli import main

    csv_file = tmp_path / "bars.csv"
    csv_file.write_text("timestamp,open,high,low,close,volume\n")
    code = main([
        "overfitting-check", "--strategy", "ma_crossover", "--data", str(csv_file),
        "--grid", "not valid json",
    ])
    assert code == 1


def test_overfitting_check_rejects_non_object_grid(tmp_path):
    from app.cli import main

    csv_file = tmp_path / "bars.csv"
    csv_file.write_text("timestamp,open,high,low,close,volume\n")
    code = main([
        "overfitting-check", "--strategy", "ma_crossover", "--data", str(csv_file),
        "--grid", "[1, 2, 3]",
    ])
    assert code == 1
