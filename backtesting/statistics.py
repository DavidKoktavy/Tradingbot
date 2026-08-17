"""
Statistical corrections for backtest overfitting.

Implements the Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio
(DSR) from Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting, and Non-Normality",
plus the companion Minimum Track Record Length statistic.

The problem this solves: a backtest's Sharpe ratio is an ESTIMATE from a
finite, noisy sample, not the true value. Two distinct sources of error
compound:

1. **Sampling noise.** Even a genuinely skilled strategy's measured Sharpe
   bounces around the true Sharpe from one sample to the next, more so
   with fewer observations and with non-normal (fat-tailed, skewed)
   returns -- which most real trade P&L distributions are.

2. **Selection bias.** If you try N parameter combinations and report the
   best one's Sharpe, you have implicitly run a multiple-comparisons
   test. Even N strategies with *zero* true skill will produce a "best"
   Sharpe well above zero by chance alone, and that expected maximum
   grows with N. Reporting the winner's raw Sharpe without correcting
   for how many were tried is a textbook multiple-testing error.

PSR addresses (1): given the observed Sharpe, the sample size, and the
skew/kurtosis of returns, what is the probability the TRUE Sharpe exceeds
some benchmark? DSR addresses both (1) and (2) at once, by setting that
benchmark to the Sharpe you'd EXPECT to see by chance alone from N trials
of skill-less strategies, then asking PSR's question against it.

Design decisions:

- **DSR requires the full set of trial results, not just the winner.**
  This is the entire point of the correction. `walk_forward.grid_search`
  is extended (via `grid_search_with_trials`) to retain every trial's
  Sharpe; asking for DSR after only keeping the best result is not
  possible because the deflation benchmark is a function of the whole
  search, and that information no longer exists once discarded.

- **Below a minimum sample size, this refuses to answer rather than
  returning a falsely precise number.** The normal approximation
  underlying PSR needs a reasonable number of observations; with too few,
  every function here returns `None` with an explanatory warning instead
  of a number that looks authoritative but isn't.

- **Skew and kurtosis default to normal (0, 3) only when the caller
  cannot supply the actual return series, and this is flagged loudly.**
  Real trade P&L is usually more fat-tailed than normal, and assuming
  normality when it isn't UNDERSTATES the true uncertainty -- the honest
  default leans toward a warning, not toward silently assuming the
  friendlier case.

- **This corrects for how many trials you ran. It says nothing about
  the future.** DSR is a statement about confidence given the search that
  produced the reported number. It does not replace out-of-sample
  testing, and a high DSR on in-sample data is not a substitute for
  `walk_forward.evaluate_out_of_sample`. Use both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist

_NORMAL = NormalDist()
_EULER_MASCHERONI = 0.5772156649015329
_MIN_OBSERVATIONS = 20  # below this, a Sharpe estimate is too noisy to correct meaningfully
_MIN_TRADES_FOR_INDEPENDENCE = 30  # below this, per-period returns are too autocorrelated
# with each other (dominated by too few actual position changes) for the
# PSR/DSR independence assumption to be trustworthy, regardless of how
# many return periods were observed. Matches
# MetricsResult.is_statistically_meaningful's threshold deliberately.


def sample_skewness(returns: list[float]) -> float:
    """Plug-in (population-moment) skewness estimator, matching the
    convention used in the PSR/DSR derivation (Mertens, 2002)."""
    n = len(returns)
    if n < 3:
        return 0.0
    mean = sum(returns) / n
    variance = sum((r - mean) ** 2 for r in returns) / n
    std = math.sqrt(variance)
    if std == 0:
        return 0.0
    return (sum((r - mean) ** 3 for r in returns) / n) / std**3


def sample_kurtosis(returns: list[float]) -> float:
    """Plug-in (population-moment) kurtosis estimator, NON-excess (a
    normal distribution has kurtosis 3, not 0), matching the PSR formula's
    convention."""
    n = len(returns)
    if n < 4:
        return 3.0
    mean = sum(returns) / n
    variance = sum((r - mean) ** 2 for r in returns) / n
    std = math.sqrt(variance)
    if std == 0:
        return 3.0
    return (sum((r - mean) ** 4 for r in returns) / n) / std**4


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    *,
    n_observations: int,
    benchmark_sharpe: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float | None:
    """P(true Sharpe > benchmark_sharpe | observed Sharpe, sample size,
    return distribution shape).

    `observed_sharpe` and `benchmark_sharpe` must be in the SAME units as
    the periods used to compute `n_observations` (both per-period, or
    both annualised consistently) -- mixing units silently produces a
    wrong answer, which is why this function takes them as plain floats
    rather than inferring units. Callers are responsible for consistency.
    """
    if n_observations < _MIN_OBSERVATIONS:
        return None
    denom = 1 - skew * observed_sharpe + ((kurtosis - 1) / 4) * observed_sharpe**2
    if denom <= 0:
        # Degenerate: the non-normality correction has overwhelmed the
        # estimate. Refuse rather than return a number computed from a
        # negative variance term.
        return None
    z = (observed_sharpe - benchmark_sharpe) * math.sqrt(n_observations - 1) / math.sqrt(denom)
    return _NORMAL.cdf(z)


def expected_max_sharpe_under_luck(sharpe_std: float, n_trials: int) -> float:
    """The Sharpe ratio you would EXPECT to observe as the maximum across
    `n_trials` independent strategies that all have ZERO true skill,
    given how much the trials' Sharpe estimates vary (`sharpe_std`).

    This is the benchmark DSR tests against. It grows with `n_trials`:
    the more you search, the higher a "best" Sharpe you should expect
    from luck alone, and reporting a Sharpe above the search's own
    expected maximum is a much weaker claim than it looks.
    """
    if n_trials <= 1 or sharpe_std <= 0:
        return 0.0  # no selection bias with zero or one trial
    p1 = min(1 - 1e-12, max(1e-12, 1 - 1 / n_trials))
    p2 = min(1 - 1e-12, max(1e-12, 1 - 1 / (n_trials * math.e)))
    term1 = _NORMAL.inv_cdf(p1)
    term2 = _NORMAL.inv_cdf(p2)
    return sharpe_std * ((1 - _EULER_MASCHERONI) * term1 + _EULER_MASCHERONI * term2)


def minimum_track_record_length(
    observed_sharpe: float,
    *,
    benchmark_sharpe: float = 0.0,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    confidence: float = 0.95,
) -> float | None:
    """How many return observations would be needed before we could be
    `confidence` sure the true Sharpe exceeds `benchmark_sharpe`, given
    the CURRENTLY observed Sharpe and return shape.

    This is the direct, actionable complement to DSR: instead of "your
    Sharpe might be noise", it answers "trade this many more periods
    before the question is even statistically answerable at your current
    edge." As the observed edge over the benchmark shrinks, this grows
    without bound -- which is correct: a marginal edge genuinely requires
    a very long track record to distinguish from luck.
    """
    edge = observed_sharpe - benchmark_sharpe
    if edge <= 0:
        return None  # no measured edge at all; no amount of data helps
    denom = 1 - skew * observed_sharpe + ((kurtosis - 1) / 4) * observed_sharpe**2
    if denom <= 0:
        return None
    z = _NORMAL.inv_cdf(confidence)
    return 1 + (z**2 * denom) / edge**2


@dataclass
class TrialResult:
    """One parameter combination's result from a search. Deliberately
    lightweight (no equity curve retained) so a large grid search can keep
    every trial without exhausting memory -- see module docstring on why
    keeping every trial, not just the winner, is required at all."""

    params: dict
    sharpe: float | None
    n_trades: int
    n_periods: int


@dataclass
class DeflationReport:
    observed_sharpe: float
    n_trials: int
    n_observations: int
    n_trades: int
    skew: float
    kurtosis: float
    probabilistic_sharpe_ratio: float | None
    expected_max_sharpe_by_chance: float | None
    deflated_sharpe_ratio: float | None
    minimum_track_record_length: float | None
    warnings: list[str] = field(default_factory=list)

    @property
    def has_result(self) -> bool:
        return self.deflated_sharpe_ratio is not None

    @property
    def likely_genuine(self) -> bool:
        """Conservative read, at 95% confidence: True only if DSR clears
        that bar AND the result is backed by enough actual trades.

        The DSR/PSR math operates on return PERIODS and assumes they are
        roughly independent. A Sharpe computed from 400 daily returns
        that are almost entirely the mark-to-market of ONE continuous
        position is not 400 independent observations of skill — it is
        one observation, autocorrelated 400 times over. The period-count
        math alone cannot see this, so a minimum trade count is enforced
        here as a separate, blunt guard: DSR is necessary but not
        sufficient. Matches the same `min_trades` convention as
        `MetricsResult.is_statistically_meaningful` elsewhere in this
        codebase, deliberately, so the two checks agree with each other.
        """
        return (
            self.has_result
            and self.deflated_sharpe_ratio >= 0.95
            and self.n_trades >= _MIN_TRADES_FOR_INDEPENDENCE
        )

    def summary(self) -> str:
        lines = [
            f"Observed Sharpe: {self.observed_sharpe:.3f} "
            f"(from {self.n_observations} observations, {self.n_trades} trades, "
            f"{self.n_trials} trials searched)",
        ]
        if self.expected_max_sharpe_by_chance is not None:
            lines.append(
                f"Expected best Sharpe from {self.n_trials} skill-less trials by chance "
                f"alone: {self.expected_max_sharpe_by_chance:.3f}"
            )
        if self.deflated_sharpe_ratio is not None:
            lines.append(
                f"Deflated Sharpe Ratio: {self.deflated_sharpe_ratio:.1%} probability the "
                "true Sharpe exceeds what this search would produce from pure luck"
            )
            lines.append(
                "VERDICT: "
                + (
                    "clears 95% confidence given this search size"
                    if self.likely_genuine
                    else "does not clear 95% confidence -- treat as UNPROVEN given the search size"
                )
            )
        if self.minimum_track_record_length is not None:
            extra = max(0.0, self.minimum_track_record_length - self.n_observations)
            lines.append(
                f"Minimum track record for 95% confidence at this edge: "
                f"{self.minimum_track_record_length:.0f} observations "
                f"({extra:.0f} more than currently available)"
            )
        for w in self.warnings:
            lines.append(f"WARNING: {w}")
        return "\n".join(lines)


def evaluate_search_overfitting(
    trials: list[TrialResult],
    *,
    best: TrialResult | None = None,
    returns: list[float] | None = None,
    confidence: float = 0.95,
) -> DeflationReport:
    """The main entry point: given every trial from a parameter search,
    compute PSR/DSR/MinTRL for the best one.

    `returns` -- the winning trial's per-period return series -- is
    optional but strongly recommended. Without it, skew/kurtosis default
    to normal (0, 3), which UNDERSTATES real fat-tailed risk and is
    flagged as a warning rather than silently assumed.
    """
    warnings: list[str] = []
    scored = [t for t in trials if t.sharpe is not None]
    if not scored:
        return DeflationReport(
            observed_sharpe=0.0, n_trials=len(trials), n_observations=0, n_trades=0,
            skew=0.0, kurtosis=3.0, probabilistic_sharpe_ratio=None,
            expected_max_sharpe_by_chance=None, deflated_sharpe_ratio=None,
            minimum_track_record_length=None,
            warnings=["No trial produced a computable Sharpe ratio"],
        )

    winner = best or max(scored, key=lambda t: t.sharpe)
    n_trials = len(trials)

    if returns is not None and len(returns) >= 4:
        skew = sample_skewness(returns)
        kurt = sample_kurtosis(returns)
    else:
        skew, kurt = 0.0, 3.0
        warnings.append(
            "No return series supplied -- assuming normal returns (skew=0, kurtosis=3), "
            "which UNDERSTATES risk for real, typically fat-tailed trade P&L. Supply "
            "`returns` for an accurate figure."
        )

    n_obs = winner.n_periods
    if winner.n_trades < _MIN_TRADES_FOR_INDEPENDENCE:
        warnings.append(
            f"Only {winner.n_trades} trades produced {n_obs} return observations -- "
            "PSR/DSR assume roughly INDEPENDENT per-period returns, and a Sharpe "
            "dominated by very few trades is really one or two autocorrelated "
            "observations wearing many periods' clothing. The figures below are "
            "computed anyway for reference, but `likely_genuine` is forced False "
            f"until at least {_MIN_TRADES_FOR_INDEPENDENCE} trades back the result."
        )

    if n_obs < _MIN_OBSERVATIONS:
        warnings.append(
            f"Only {n_obs} observations -- below {_MIN_OBSERVATIONS}, PSR/DSR are not "
            "reliable and are not computed."
        )
        return DeflationReport(
            observed_sharpe=winner.sharpe, n_trials=n_trials, n_observations=n_obs,
            n_trades=winner.n_trades, skew=skew, kurtosis=kurt,
            probabilistic_sharpe_ratio=None, expected_max_sharpe_by_chance=None,
            deflated_sharpe_ratio=None, minimum_track_record_length=None,
            warnings=warnings,
        )

    sharpes = [t.sharpe for t in scored]
    if len(sharpes) > 1:
        mean_sharpe = sum(sharpes) / len(sharpes)
        sharpe_std = math.sqrt(sum((s - mean_sharpe) ** 2 for s in sharpes) / (len(sharpes) - 1))
    else:
        sharpe_std = 0.0
        warnings.append(
            "Only one scored trial -- no selection bias to correct for, but this also "
            "means the search itself is too small to detect overfitting risk across "
            "parameter choices."
        )

    sr0 = expected_max_sharpe_under_luck(sharpe_std, n_trials)
    psr = probabilistic_sharpe_ratio(
        winner.sharpe, n_observations=n_obs, benchmark_sharpe=0.0, skew=skew, kurtosis=kurt
    )
    dsr = probabilistic_sharpe_ratio(
        winner.sharpe, n_observations=n_obs, benchmark_sharpe=sr0, skew=skew, kurtosis=kurt
    )
    mintrl = minimum_track_record_length(
        winner.sharpe, benchmark_sharpe=sr0, skew=skew, kurtosis=kurt, confidence=confidence
    )

    return DeflationReport(
        observed_sharpe=winner.sharpe,
        n_trials=n_trials,
        n_observations=n_obs,
        n_trades=winner.n_trades,
        skew=skew,
        kurtosis=kurt,
        probabilistic_sharpe_ratio=psr,
        expected_max_sharpe_by_chance=sr0,
        deflated_sharpe_ratio=dsr,
        minimum_track_record_length=mintrl,
        warnings=warnings,
    )
