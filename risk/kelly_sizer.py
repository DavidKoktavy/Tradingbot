"""
Fractional Kelly position sizing.

Design decisions:

- **Sized from actual trade history, never from a point estimate.** The
  Kelly formula f* = W - (1-W)/R is only correct if W (win probability)
  and R (win/loss ratio) are the TRUE values. In practice they are
  estimated from a finite trade sample, and using the raw sample win rate
  plugs a noisy point estimate into a formula that is extremely sensitive
  to it -- a strategy with 8 wins out of 10 trades has a raw win rate of
  80%, but nowhere near enough evidence to bet as if that were true. This
  sizer uses the WILSON SCORE LOWER BOUND of the win rate at a configured
  confidence level instead of the raw rate, which shrinks toward caution
  automatically as the sample gets smaller and widens toward the raw rate
  as it grows -- exactly the behaviour a responsible estimate should have.

- **Fractional, not full, Kelly -- and the fraction is a hard multiplier,
  not a suggestion.** Full Kelly is the growth-optimal bet size only
  under the assumption that W and R are known exactly; any estimation
  error in either one means full Kelly systematically oversizes and full
  Kelly investors get wiped out by exactly the estimation error this
  sizer is designed to respect. A quarter-Kelly default (0.25) is a
  standard, conservative convention.

- **A strategy without enough trade history is refused sizing entirely**,
  the same "unknown risk gets zero, not a guess" principle as
  `PositionSizer`. There is no fallback to a plausible-looking default
  size when the statistics aren't there yet.

- **Negative or zero edge clips to zero, never to a short.** A negative
  Kelly fraction technically suggests betting the opposite direction, but
  this sizer has no information about whether the OPPOSITE trade has a
  positive edge (that would need its own separate trade history) -- it
  only knows this strategy's history doesn't support taking the trade at
  all. Refusing is correct; inferring a reversal would not be.

- **The output is still bounded by every downstream risk check.** This
  class only decides "how big would fractional Kelly suggest, given what
  we actually know" -- `RiskEngine`'s position-size, gross-exposure, and
  buying-power checks still run afterward exactly as they do for any
  other sizer, and can only shrink the result further, never grow it.

- **Trade statistics are supplied externally and never mutated by the AI
  layer.** This class has no method reachable from
  `ai/decision_engine.py` or `ai/reflection.py`. Feeding it fabricated
  statistics would be a way to manufacture an oversized position; the
  statistics must come from `ai/performance_analyzer.py`'s deterministic
  computation over real trade history, refreshed by whatever owns the
  `RiskEngine` (the control loop, or an operator), not by the model.
"""

from __future__ import annotations

import math
from decimal import ROUND_DOWN, Decimal
from statistics import NormalDist

import structlog
from pydantic import BaseModel

log = structlog.get_logger(__name__)

_NORMAL = NormalDist()


def wilson_lower_bound(successes: int, n: int, *, confidence: float = 0.95) -> float:
    """Lower bound of the Wilson score confidence interval for a binomial
    proportion. Unlike the raw win rate, this correctly reflects sample
    size: it is close to the raw rate with a large sample and pulled
    sharply toward 0 with a small one."""
    if n <= 0:
        return 0.0
    z = _NORMAL.inv_cdf(1 - (1 - confidence) / 2)
    p = successes / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    adjustment = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return max(0.0, (center - adjustment) / denom)


def kelly_fraction_formula(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """The classic Kelly criterion: f* = W - (1-W)/R, where R is the
    win/loss payoff ratio. Clipped at zero -- see module docstring on why
    a negative result means "don't take this trade", not "reverse it"."""
    if avg_loss == 0 or avg_win <= 0:
        return 0.0
    r = avg_win / abs(avg_loss)
    f = win_rate - (1 - win_rate) / r
    return max(0.0, f)


class KellyStats(BaseModel):
    """The minimum a strategy's trade history must supply. Deliberately a
    narrow, explicit shape rather than accepting a full
    `ai.performance_analyzer.StrategyStats` object, so this module has no
    import dependency on the analysis layer and can be fed by anything
    that can produce these four numbers."""

    n_trades: int
    n_wins: int
    average_win: float
    average_loss: float  # expected negative or zero


class KellySizingResult(BaseModel):
    quantity: Decimal
    method: str
    detail: str = ""
    win_rate_lower_bound: float | None = None
    full_kelly_fraction: float | None = None
    applied_fraction: float | None = None
    risk_amount: Decimal = Decimal("0")  # kept for interface parity with SizingResult

    @property
    def is_tradeable(self) -> bool:
        return self.quantity > 0


class KellyPositionSizer:
    def __init__(
        self,
        *,
        kelly_fraction_multiplier: float = 0.25,
        min_trades: int = 30,
        confidence: float = 0.95,
        max_fraction_of_equity: Decimal = Decimal("0.25"),
        allow_fractional: bool = False,
    ) -> None:
        if not 0 < kelly_fraction_multiplier <= 1:
            raise ValueError("kelly_fraction_multiplier must be in (0, 1]")
        self._multiplier = kelly_fraction_multiplier
        self._min_trades = min_trades
        self._confidence = confidence
        # A sanity ceiling independent of RiskEngine's own max_position_size,
        # so this sizer is safe to use standalone (e.g. in a backtest) and
        # not solely reliant on being wrapped by the risk engine.
        self._max_fraction = max_fraction_of_equity
        self._allow_fractional = allow_fractional
        self._stats: dict[str, KellyStats] = {}

    def update_stats(self, strategy: str, stats: KellyStats) -> None:
        """The only way statistics enter this sizer. Called by whatever
        owns the RiskEngine (control loop, or an operator refreshing from
        `ai/performance_analyzer.py`'s output) -- never by the AI layer."""
        self._stats[strategy] = stats

    def stats_for(self, strategy: str) -> KellyStats | None:
        return self._stats.get(strategy)

    def calculate(
        self,
        *,
        equity: Decimal,
        entry_price: Decimal,
        stop_price: Decimal | None = None,
        atr: Decimal | None = None,
        requested_quantity: Decimal | None = None,
        strategy: str | None = None,
    ) -> KellySizingResult:
        if strategy is None or strategy not in self._stats:
            return KellySizingResult(
                quantity=Decimal("0"),
                method="none",
                detail=f"No trade history on file for strategy {strategy!r}; refusing to size",
            )
        stats = self._stats[strategy]

        if stats.n_trades < self._min_trades:
            return KellySizingResult(
                quantity=Decimal("0"),
                method="none",
                detail=(
                    f"Only {stats.n_trades} trades for {strategy!r}; need "
                    f"{self._min_trades} before Kelly sizing is used"
                ),
            )

        if equity <= 0 or entry_price <= 0:
            return KellySizingResult(
                quantity=Decimal("0"), method="none",
                detail="Non-positive equity or entry price",
            )

        w_lb = wilson_lower_bound(stats.n_wins, stats.n_trades, confidence=self._confidence)
        full_kelly = kelly_fraction_formula(w_lb, stats.average_win, stats.average_loss)

        if full_kelly <= 0:
            return KellySizingResult(
                quantity=Decimal("0"),
                method="none",
                detail=(
                    f"No positive edge at the {self._confidence:.0%} conservative win-rate "
                    f"estimate ({w_lb:.1%} lower bound vs {stats.n_wins}/{stats.n_trades} raw)"
                ),
                win_rate_lower_bound=w_lb,
                full_kelly_fraction=0.0,
            )

        raw_fraction = full_kelly * self._multiplier
        fraction = min(Decimal(str(raw_fraction)), self._max_fraction)
        capped_by_ceiling = Decimal(str(raw_fraction)) > self._max_fraction

        notional = equity * fraction
        raw_quantity = notional / entry_price

        reduced_by_request = False
        if requested_quantity is not None and requested_quantity < raw_quantity:
            raw_quantity = requested_quantity
            reduced_by_request = True

        quantity = self._round(raw_quantity)

        details = [
            f"Wilson lower-bound win rate {w_lb:.1%} (raw {stats.n_wins}/{stats.n_trades})",
            f"full Kelly {full_kelly:.1%}, x{self._multiplier} multiplier",
        ]
        if capped_by_ceiling:
            details.append(f"capped at {self._max_fraction:.0%} of equity ceiling")
        if reduced_by_request:
            details.append("limited to requested quantity")

        return KellySizingResult(
            quantity=quantity,
            method="fractional_kelly",
            detail="; ".join(details),
            win_rate_lower_bound=w_lb,
            full_kelly_fraction=full_kelly,
            applied_fraction=float(fraction),
        )

    def _round(self, quantity: Decimal) -> Decimal:
        if quantity <= 0:
            return Decimal("0")
        if self._allow_fractional:
            return quantity.quantize(Decimal("0.0001"), rounding=ROUND_DOWN)
        return quantity.quantize(Decimal("1"), rounding=ROUND_DOWN)
