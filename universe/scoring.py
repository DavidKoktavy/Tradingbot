"""
Deterministic candidate scoring on daily bars.

Design decisions:

- **Eligibility first, ranking second.** A candidate must pass hard
  filters — enough history, minimum price, minimum average dollar volume,
  close above its long-term trend — before its score means anything. An
  ineligible candidate keeps its score for the audit trail but is never
  selected, however high it is.

- **The score is a dip score**, matching what the bot is set up to trade
  (`buy_the_dip`, long only): a deeper short-RSI washout and a larger
  pullback from the recent high both rank higher. It is a ranking, not a
  forecast; no claim is made that a higher score predicts a better trade.

- **Daily bars, independent of the strategy bar size.** Selection asks
  "which names are worth watching today"; the strategy then times entries
  on whatever bar size the loop runs. Mixing the two would make the
  universe flicker with every intraday bar.

- **Pure function, no I/O**, so it is unit-testable and reproducible
  against stored bars.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from data.models import Bar
from strategies.indicators import rsi, sma


class ScoringParams(BaseModel):
    model_config = {"extra": "forbid"}

    trend_period: int = Field(default=200, gt=1)
    rsi_period: int = Field(default=2, gt=0)
    dip_lookback: int = Field(default=20, gt=1)
    liquidity_lookback: int = Field(default=20, gt=0)
    min_price: float = Field(default=5.0, ge=0)
    min_avg_dollar_volume: float = Field(default=50_000_000.0, ge=0)
    # Weighting between RSI washout and pullback depth; must sum to 1.
    rsi_weight: float = Field(default=0.6, ge=0, le=1)
    full_pullback_pct: float = Field(default=0.10, gt=0, le=1)

    @property
    def min_bars(self) -> int:
        return max(self.trend_period, self.dip_lookback + 1, self.liquidity_lookback) + 1


@dataclass
class CandidateScore:
    symbol: str
    eligible: bool = False
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    close: float | None = None
    avg_dollar_volume: float | None = None
    trend_sma: float | None = None
    rsi: float | None = None
    pullback_pct: float | None = None
    scanner_hit: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "eligible": self.eligible,
            "score": round(self.score, 4),
            "reasons": list(self.reasons),
            "close": self.close,
            "avg_dollar_volume": self.avg_dollar_volume,
            "trend_sma": self.trend_sma,
            "rsi": self.rsi,
            "pullback_pct": self.pullback_pct,
            "scanner_hit": self.scanner_hit,
        }


def score_candidate(
    symbol: str, bars: list[Bar], params: ScoringParams | None = None
) -> CandidateScore:
    p = params or ScoringParams()
    result = CandidateScore(symbol=symbol.upper())

    if len(bars) < p.min_bars:
        result.reasons.append(f"insufficient history: {len(bars)} < {p.min_bars} daily bars")
        return result

    closes = [b.close for b in bars]
    close = closes[-1]
    result.close = close

    recent = bars[-p.liquidity_lookback :]
    result.avg_dollar_volume = sum(b.close * b.volume for b in recent) / len(recent)
    result.trend_sma = sma(closes, p.trend_period)[-1]
    result.rsi = rsi(closes, p.rsi_period)[-1]

    # Recent high excludes the current bar (no look-ahead; see buy_the_dip).
    window = bars[-p.dip_lookback - 1 : -1]
    recent_high = max(b.high for b in window)
    result.pullback_pct = max(0.0, (recent_high - close) / recent_high) if recent_high > 0 else 0.0

    if close < p.min_price:
        result.reasons.append(f"price {close:.2f} below minimum {p.min_price:.2f}")
    if result.avg_dollar_volume < p.min_avg_dollar_volume:
        result.reasons.append(
            f"avg dollar volume {result.avg_dollar_volume:,.0f} below "
            f"{p.min_avg_dollar_volume:,.0f}"
        )
    if result.trend_sma is None or close <= result.trend_sma:
        result.reasons.append(f"not in uptrend (close <= SMA{p.trend_period})")
    if result.rsi is None:
        result.reasons.append("RSI undefined")

    rsi_part = 1.0 - (result.rsi or 100.0) / 100.0
    pullback_part = min(result.pullback_pct / p.full_pullback_pct, 1.0)
    result.score = p.rsi_weight * rsi_part + (1.0 - p.rsi_weight) * pullback_part
    result.eligible = not result.reasons
    return result
