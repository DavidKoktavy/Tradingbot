"""
Performance metrics.

Design decisions:

- Every ratio that can divide by zero returns `None`, not `0.0` or
  `inf`. A Sharpe ratio of 0.0 reads as "flat performance"; `None` reads
  as "not computable from this sample", which is the truth when there are
  three trades. Silently returning a number invites decisions based on
  noise.

- Sharpe and Sortino are annualised from the observed bar frequency,
  which must be supplied. There is no default `252`, because applying a
  daily annualisation factor to minute bars inflates the ratio by ~20x and
  is a common and expensive mistake.

- Maximum drawdown is computed on the equity curve including open
  positions, not on closed-trade P&L. Closed-trade drawdown understates
  the pain an operator actually experiences and the margin they actually
  need.

- `MetricsResult` carries `n_trades` prominently. Nearly every metric here
  is statistically meaningless below ~30 trades, and the report says so.
"""

from __future__ import annotations

import math
from decimal import Decimal

from pydantic import BaseModel, Field

# Bars per year for common frequencies, for annualisation.
PERIODS_PER_YEAR = {
    "1 min": 252 * 390,
    "5 mins": 252 * 78,
    "15 mins": 252 * 26,
    "1 hour": 252 * 6.5,
    "1 day": 252,
    "1 week": 52,
    "1 month": 12,
}


class TradeRecord(BaseModel):
    """A completed round-trip."""

    instrument: str
    strategy: str
    entry_time: str
    exit_time: str
    side: str
    quantity: Decimal
    entry_price: Decimal
    exit_price: Decimal
    gross_pnl: Decimal
    commission: Decimal
    net_pnl: Decimal
    bars_held: int

    @property
    def is_win(self) -> bool:
        return self.net_pnl > 0


class MetricsResult(BaseModel):
    # Sample size — read this before any other number.
    n_trades: int = 0
    n_periods: int = 0
    is_statistically_meaningful: bool = False

    # Returns
    total_return: float | None = None
    cagr: float | None = None

    # Risk-adjusted
    sharpe: float | None = None
    sortino: float | None = None
    calmar: float | None = None
    volatility_annualised: float | None = None

    # Drawdown
    max_drawdown: float | None = None
    max_drawdown_duration_periods: int = 0

    # Trade statistics
    win_rate: float | None = None
    profit_factor: float | None = None
    average_win: float | None = None
    average_loss: float | None = None
    expectancy: float | None = None
    largest_win: float | None = None
    largest_loss: float | None = None
    average_bars_held: float | None = None

    # Activity
    total_commission: float = 0.0
    exposure: float | None = None
    turnover: float | None = None

    warnings: list[str] = Field(default_factory=list)

    def summary(self) -> str:
        def fmt(x: float | None, pct: bool = False) -> str:
            if x is None:
                return "n/a"
            return f"{x:.2%}" if pct else f"{x:.2f}"

        lines = [
            f"Trades: {self.n_trades}  Periods: {self.n_periods}",
            f"Total return: {fmt(self.total_return, True)}  CAGR: {fmt(self.cagr, True)}",
            f"Sharpe: {fmt(self.sharpe)}  Sortino: {fmt(self.sortino)}  "
            f"Calmar: {fmt(self.calmar)}",
            f"Max drawdown: {fmt(self.max_drawdown, True)}",
            f"Win rate: {fmt(self.win_rate, True)}  Profit factor: {fmt(self.profit_factor)}",
            f"Expectancy: {fmt(self.expectancy)}  Commission: {self.total_commission:.2f}",
        ]
        if self.warnings:
            lines.append("WARNINGS: " + "; ".join(self.warnings))
        return "\n".join(lines)


def _safe_div(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def compute_returns(equity_curve: list[float]) -> list[float]:
    """Period-over-period simple returns."""
    out = []
    for i in range(1, len(equity_curve)):
        prev = equity_curve[i - 1]
        out.append((equity_curve[i] - prev) / prev if prev else 0.0)
    return out


def max_drawdown(equity_curve: list[float]) -> tuple[float | None, int]:
    """Returns (max drawdown as a positive fraction, longest duration in
    periods)."""
    if len(equity_curve) < 2:
        return None, 0
    peak = equity_curve[0]
    max_dd = 0.0
    current_duration = 0
    max_duration = 0
    for value in equity_curve:
        if value > peak:
            peak = value
            current_duration = 0
        else:
            current_duration += 1
            max_duration = max(max_duration, current_duration)
        if peak > 0:
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
    return max_dd, max_duration


def sharpe_ratio(
    returns: list[float], *, periods_per_year: float, risk_free_rate: float = 0.0
) -> float | None:
    if len(returns) < 2:
        return None
    excess = [r - risk_free_rate / periods_per_year for r in returns]
    mean = sum(excess) / len(excess)
    variance = sum((r - mean) ** 2 for r in excess) / (len(excess) - 1)
    std = math.sqrt(variance)
    if std == 0:
        return None
    return (mean / std) * math.sqrt(periods_per_year)


def sortino_ratio(
    returns: list[float], *, periods_per_year: float, risk_free_rate: float = 0.0
) -> float | None:
    """Like Sharpe but penalises only downside deviation."""
    if len(returns) < 2:
        return None
    target = risk_free_rate / periods_per_year
    excess = [r - target for r in returns]
    mean = sum(excess) / len(excess)
    downside = [r for r in excess if r < 0]
    if not downside:
        return None  # no downside observed: ratio is undefined, not infinite
    downside_var = sum(r**2 for r in downside) / len(excess)
    downside_dev = math.sqrt(downside_var)
    if downside_dev == 0:
        return None
    return (mean / downside_dev) * math.sqrt(periods_per_year)


def compute_metrics(
    *,
    equity_curve: list[float],
    trades: list[TradeRecord],
    periods_per_year: float,
    bars_in_market: int = 0,
    total_traded_notional: float = 0.0,
    min_meaningful_trades: int = 30,
) -> MetricsResult:
    result = MetricsResult(n_trades=len(trades), n_periods=len(equity_curve))

    if len(equity_curve) < 2:
        result.warnings.append("Equity curve too short to compute metrics")
        return result

    start, end = equity_curve[0], equity_curve[-1]
    returns = compute_returns(equity_curve)

    result.total_return = _safe_div(end - start, start)

    years = len(equity_curve) / periods_per_year
    if years > 0 and start > 0 and end > 0:
        result.cagr = (end / start) ** (1 / years) - 1

    result.sharpe = sharpe_ratio(returns, periods_per_year=periods_per_year)
    result.sortino = sortino_ratio(returns, periods_per_year=periods_per_year)

    if len(returns) >= 2:
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        result.volatility_annualised = math.sqrt(variance) * math.sqrt(periods_per_year)

    dd, dd_duration = max_drawdown(equity_curve)
    result.max_drawdown = dd
    result.max_drawdown_duration_periods = dd_duration
    if dd and dd > 0 and result.cagr is not None:
        result.calmar = result.cagr / dd

    # Trade statistics
    if trades:
        wins = [float(t.net_pnl) for t in trades if t.is_win]
        losses = [float(t.net_pnl) for t in trades if not t.is_win]
        result.win_rate = len(wins) / len(trades)
        result.average_win = sum(wins) / len(wins) if wins else None
        result.average_loss = sum(losses) / len(losses) if losses else None
        result.largest_win = max(wins) if wins else None
        result.largest_loss = min(losses) if losses else None
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        result.profit_factor = _safe_div(gross_profit, gross_loss)
        result.expectancy = sum(float(t.net_pnl) for t in trades) / len(trades)
        result.average_bars_held = sum(t.bars_held for t in trades) / len(trades)
        result.total_commission = sum(float(t.commission) for t in trades)

    result.exposure = _safe_div(bars_in_market, len(equity_curve))
    result.turnover = _safe_div(total_traded_notional, start)

    # Honest statistical caveats.
    result.is_statistically_meaningful = len(trades) >= min_meaningful_trades
    if not result.is_statistically_meaningful:
        result.warnings.append(
            f"Only {len(trades)} trades — below {min_meaningful_trades}; "
            "these metrics are not statistically meaningful"
        )
    if result.max_drawdown is not None and result.max_drawdown == 0 and len(trades) > 0:
        result.warnings.append(
            "Zero drawdown observed — verify the cost model and data are realistic"
        )
    if result.sharpe is not None and result.sharpe > 3:
        result.warnings.append(
            f"Sharpe of {result.sharpe:.2f} is implausibly high for a real strategy; "
            "check for look-ahead bias, survivorship bias, or unrealistic fills"
        )
    return result
