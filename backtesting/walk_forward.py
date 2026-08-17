"""
Walk-forward analysis and out-of-sample validation.

Design decisions:

- The split is **chronological and non-overlapping**. Random k-fold on
  time series leaks future information into training and is one of the
  fastest ways to produce a beautiful, worthless backtest.

- `walk_forward` re-optimises on each in-sample window and evaluates on
  the *immediately following* out-of-sample window, then rolls forward.
  Only the out-of-sample segments are concatenated into the reported
  result — the in-sample numbers are diagnostics, never headline results.

- `DegradationReport` explicitly quantifies in-sample vs out-of-sample
  decay. A strategy that looks excellent in-sample and mediocre
  out-of-sample is overfit, and the report says so rather than leaving the
  reader to notice. This is the check the AI research loop (Phase 7+) must
  pass before anything is promoted.

- Parameter optimisation here is exhaustive grid search, which is honest
  about what it is: a way to *detect* overfitting by seeing how much the
  best in-sample parameters degrade, not a way to find good parameters.
  The more grid points searched, the more the in-sample result is
  selection noise, and `n_combinations_tested` is reported so a reader can
  discount accordingly.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import structlog

from data.models import Bar, Instrument
from backtesting.costs import CostModel
from backtesting.engine import BacktestEngine, BacktestResult
from backtesting.metrics import MetricsResult
from backtesting.statistics import TrialResult
from risk.risk_engine import RiskEngineLimits
from strategies.base import Strategy, StrategyParams

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class DataSplit:
    """Chronological split. Windows never overlap and never reorder."""

    train: list[Bar]
    validation: list[Bar]
    test: list[Bar]

    @property
    def sizes(self) -> tuple[int, int, int]:
        return len(self.train), len(self.validation), len(self.test)


def split_chronological(
    bars: list[Bar], *, train_pct: float = 0.6, validation_pct: float = 0.2
) -> DataSplit:
    if not 0 < train_pct < 1 or not 0 <= validation_pct < 1:
        raise ValueError("Percentages must be fractions")
    if train_pct + validation_pct >= 1:
        raise ValueError("train + validation must leave room for a test set")

    n = len(bars)
    train_end = int(n * train_pct)
    val_end = train_end + int(n * validation_pct)
    return DataSplit(
        train=bars[:train_end], validation=bars[train_end:val_end], test=bars[val_end:]
    )


@dataclass
class DegradationReport:
    """In-sample vs out-of-sample comparison. The core overfitting check."""

    in_sample: MetricsResult
    out_of_sample: MetricsResult
    n_combinations_tested: int = 1

    @property
    def sharpe_decay(self) -> float | None:
        if self.in_sample.sharpe is None or self.out_of_sample.sharpe is None:
            return None
        if self.in_sample.sharpe == 0:
            return None
        return (self.in_sample.sharpe - self.out_of_sample.sharpe) / abs(self.in_sample.sharpe)

    @property
    def is_likely_overfit(self) -> bool:
        """Heuristic, deliberately strict. Any of:
        - out-of-sample Sharpe collapses to below half of in-sample
        - out-of-sample return is negative while in-sample is positive
        - many parameter combinations were searched for a small sample
        """
        decay = self.sharpe_decay
        if decay is not None and decay > 0.5:
            return True
        if (
            self.in_sample.total_return is not None
            and self.out_of_sample.total_return is not None
            and self.in_sample.total_return > 0
            and self.out_of_sample.total_return < 0
        ):
            return True
        if (
            self.n_combinations_tested > 10
            and self.out_of_sample.n_trades < self.n_combinations_tested
        ):
            return True
        return False

    def summary(self) -> str:
        lines = [
            "=== IN-SAMPLE (diagnostic only, not a result) ===",
            self.in_sample.summary(),
            "",
            "=== OUT-OF-SAMPLE (the only number that matters) ===",
            self.out_of_sample.summary(),
            "",
            f"Parameter combinations tested: {self.n_combinations_tested}",
        ]
        decay = self.sharpe_decay
        if decay is not None:
            lines.append(f"Sharpe decay: {decay:.1%}")
        if self.is_likely_overfit:
            lines.append(
                "VERDICT: LIKELY OVERFIT — out-of-sample performance does not support "
                "the in-sample result. Do not promote."
            )
        else:
            lines.append(
                "VERDICT: no overfitting signature detected. This is NOT evidence of "
                "profitability, only the absence of one specific failure mode."
            )
        return "\n".join(lines)


@dataclass
class WalkForwardWindow:
    index: int
    train_bars: int
    test_bars: int
    params: dict[str, Any]
    in_sample: MetricsResult
    out_of_sample: MetricsResult


@dataclass
class WalkForwardResult:
    windows: list[WalkForwardWindow] = field(default_factory=list)
    combined_out_of_sample_equity: list[float] = field(default_factory=list)

    @property
    def consistency(self) -> float | None:
        """Fraction of out-of-sample windows with a positive return. A
        strategy that makes all its money in one window is fragile, even
        if the aggregate looks good."""
        if not self.windows:
            return None
        positive = sum(
            1
            for w in self.windows
            if w.out_of_sample.total_return is not None and w.out_of_sample.total_return > 0
        )
        return positive / len(self.windows)

    def summary(self) -> str:
        lines = [f"Walk-forward windows: {len(self.windows)}"]
        for w in self.windows:
            oos = w.out_of_sample
            ret = f"{oos.total_return:.2%}" if oos.total_return is not None else "n/a"
            sharpe = f"{oos.sharpe:.2f}" if oos.sharpe is not None else "n/a"
            lines.append(
                f"  Window {w.index}: OOS return {ret}, Sharpe {sharpe}, "
                f"{oos.n_trades} trades, params={w.params}"
            )
        c = self.consistency
        if c is not None:
            lines.append(f"Consistency: {c:.0%} of windows positive out-of-sample")
            if c < 0.5:
                lines.append(
                    "WARNING: fewer than half of out-of-sample windows were positive"
                )
        return "\n".join(lines)


def _run(
    strategy_cls: type[Strategy],
    params: StrategyParams | None,
    bars: list[Bar],
    instrument: Instrument,
    cost_model: CostModel,
    risk_limits: RiskEngineLimits | None,
    bar_size: str,
    initial_equity: Decimal,
) -> BacktestResult | None:
    strategy = strategy_cls(params)
    engine = BacktestEngine(
        strategy=strategy,
        instrument=instrument,
        initial_equity=initial_equity,
        cost_model=cost_model,
        risk_limits=risk_limits,
        bar_size=bar_size,
    )
    try:
        return engine.run(bars)
    except ValueError as exc:
        log.warning("walkforward.window_skipped", reason=str(exc))
        return None


def grid_search(
    *,
    strategy_cls: type[Strategy],
    params_cls: type[StrategyParams],
    grid: dict[str, list[Any]],
    bars: list[Bar],
    instrument: Instrument,
    cost_model: CostModel | None = None,
    risk_limits: RiskEngineLimits | None = None,
    bar_size: str = "1 day",
    initial_equity: Decimal = Decimal("100000"),
    objective: str = "sharpe",
) -> tuple[StrategyParams | None, MetricsResult | None, int]:
    """Exhaustive search. Returns (best_params, best_metrics, n_tested).

    This exists to *measure* overfitting risk, not to find good
    parameters. Every additional grid point makes the best in-sample
    result more likely to be noise.

    For the statistical version of that warning -- an actual Deflated
    Sharpe Ratio rather than just a bigger n_tested number -- use
    `grid_search_with_trials` and feed its output to
    `backtesting.statistics.evaluate_search_overfitting`. This function
    discards every trial but the winner, which is fine for finding a
    starting point but is NOT enough information to correct for how hard
    you searched.
    """
    best_params, best_metrics, _, tested = grid_search_with_trials(
        strategy_cls=strategy_cls, params_cls=params_cls, grid=grid, bars=bars,
        instrument=instrument, cost_model=cost_model, risk_limits=risk_limits,
        bar_size=bar_size, initial_equity=initial_equity, objective=objective,
    )
    return best_params, best_metrics, tested


def grid_search_with_trials(
    *,
    strategy_cls: type[Strategy],
    params_cls: type[StrategyParams],
    grid: dict[str, list[Any]],
    bars: list[Bar],
    instrument: Instrument,
    cost_model: CostModel | None = None,
    risk_limits: RiskEngineLimits | None = None,
    bar_size: str = "1 day",
    initial_equity: Decimal = Decimal("100000"),
    objective: str = "sharpe",
) -> tuple[StrategyParams | None, MetricsResult | None, list[TrialResult], int]:
    """Like `grid_search`, but also returns every scored trial as a list
    of lightweight `TrialResult` objects -- the input
    `evaluate_search_overfitting` needs to compute a real Deflated Sharpe
    Ratio. Kept as a separate function rather than changing `grid_search`'s
    return signature, so existing callers are unaffected."""
    cost_model = cost_model or CostModel()
    keys = list(grid)
    combinations = list(itertools.product(*(grid[k] for k in keys)))

    best_params: StrategyParams | None = None
    best_metrics: MetricsResult | None = None
    best_score = float("-inf")
    tested = 0
    trials: list[TrialResult] = []

    for combo in combinations:
        kwargs = dict(zip(keys, combo))
        try:
            params = params_cls(**kwargs)
        except Exception:
            continue  # invalid parameter combination (e.g. fast >= slow)
        result = _run(
            strategy_cls, params, bars, instrument, cost_model, risk_limits,
            bar_size, initial_equity,
        )
        tested += 1
        if result is None:
            continue
        trials.append(
            TrialResult(
                params=params.model_dump(),
                sharpe=result.metrics.sharpe,
                n_trades=result.metrics.n_trades,
                n_periods=result.metrics.n_periods,
            )
        )
        score = getattr(result.metrics, objective, None)
        if score is None:
            continue
        if score > best_score:
            best_score, best_params, best_metrics = score, params, result.metrics

    return best_params, best_metrics, trials, tested


def evaluate_out_of_sample(
    *,
    strategy_cls: type[Strategy],
    params_cls: type[StrategyParams],
    grid: dict[str, list[Any]],
    bars: list[Bar],
    instrument: Instrument,
    cost_model: CostModel | None = None,
    risk_limits: RiskEngineLimits | None = None,
    bar_size: str = "1 day",
    train_pct: float = 0.7,
) -> DegradationReport | None:
    """Optimise in-sample, evaluate once out-of-sample. The out-of-sample
    set is touched exactly once — reusing it to tune turns it into a
    training set."""
    cost_model = cost_model or CostModel()
    split_at = int(len(bars) * train_pct)
    in_bars, out_bars = bars[:split_at], bars[split_at:]

    best_params, in_metrics, tested = grid_search(
        strategy_cls=strategy_cls,
        params_cls=params_cls,
        grid=grid,
        bars=in_bars,
        instrument=instrument,
        cost_model=cost_model,
        risk_limits=risk_limits,
        bar_size=bar_size,
    )
    if best_params is None or in_metrics is None:
        return None

    out_result = _run(
        strategy_cls, best_params, out_bars, instrument, cost_model, risk_limits,
        bar_size, Decimal("100000"),
    )
    if out_result is None:
        return None

    return DegradationReport(
        in_sample=in_metrics,
        out_of_sample=out_result.metrics,
        n_combinations_tested=tested,
    )


def walk_forward(
    *,
    strategy_cls: type[Strategy],
    params_cls: type[StrategyParams],
    grid: dict[str, list[Any]],
    bars: list[Bar],
    instrument: Instrument,
    train_size: int,
    test_size: int,
    cost_model: CostModel | None = None,
    risk_limits: RiskEngineLimits | None = None,
    bar_size: str = "1 day",
) -> WalkForwardResult:
    """Rolling optimise-then-test. Only out-of-sample windows are
    reported as results."""
    cost_model = cost_model or CostModel()
    result = WalkForwardResult()
    combined: list[float] = []

    start = 0
    window_index = 0
    while start + train_size + test_size <= len(bars):
        train = bars[start : start + train_size]
        test = bars[start + train_size : start + train_size + test_size]

        best_params, in_metrics, _ = grid_search(
            strategy_cls=strategy_cls,
            params_cls=params_cls,
            grid=grid,
            bars=train,
            instrument=instrument,
            cost_model=cost_model,
            risk_limits=risk_limits,
            bar_size=bar_size,
        )
        if best_params is not None and in_metrics is not None:
            out_result = _run(
                strategy_cls, best_params, test, instrument, cost_model, risk_limits,
                bar_size, Decimal("100000"),
            )
            if out_result is not None:
                result.windows.append(
                    WalkForwardWindow(
                        index=window_index,
                        train_bars=len(train),
                        test_bars=len(test),
                        params=best_params.model_dump(),
                        in_sample=in_metrics,
                        out_of_sample=out_result.metrics,
                    )
                )
                combined.extend(out_result.equity_curve)

        start += test_size
        window_index += 1

    result.combined_out_of_sample_equity = combined
    return result
