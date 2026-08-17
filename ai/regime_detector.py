"""
Market regime detection.

Design decision: regime detection is **deterministic code, not an AI
call**. It feeds the AI's context rather than being produced by it. Three
reasons:

1. It runs every cycle; an API call per cycle per instrument is slow,
   expensive, and a new failure mode in the hot path.
2. It must be reproducible in backtests. An AI-produced regime label
   cannot be replayed identically over historical data, so any strategy
   conditioned on it would be unbacktestable.
3. A regime label derived from indicators is auditable — you can point at
   the numbers that produced it months later.

The AI may *disagree* with the computed regime and say so in its
reasoning; that disagreement is logged. It does not change the label.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ai.schemas import MarketRegime
from data.models import Bar
from strategies.indicators import atr, closes, ema, rolling_std


@dataclass(frozen=True)
class RegimeAssessment:
    regime: MarketRegime
    confidence: float
    features: dict[str, float]
    rationale: str


class RegimeDetector:
    def __init__(
        self,
        *,
        trend_period: int = 50,
        slope_period: int = 20,
        vol_period: int = 20,
        vol_lookback: int = 100,
        trend_threshold: float = 0.02,
        high_vol_percentile: float = 0.8,
        high_vol_ratio: float = 1.5,
    ) -> None:
        self._trend_period = trend_period
        self._slope_period = slope_period
        self._vol_period = vol_period
        self._vol_lookback = vol_lookback
        self._trend_threshold = trend_threshold
        self._high_vol_percentile = high_vol_percentile
        self._high_vol_ratio = high_vol_ratio

    @property
    def min_bars(self) -> int:
        return max(self._trend_period, self._vol_lookback) + 2

    def detect(self, bars: Sequence[Bar]) -> RegimeAssessment:
        if len(bars) < self.min_bars:
            return RegimeAssessment(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                features={},
                rationale=f"Need {self.min_bars} bars, have {len(bars)}",
            )

        price_closes = closes(bars)
        trend = ema(price_closes, self._trend_period)
        current = price_closes[-1]

        features: dict[str, float] = {"close": current}

        # Trend slope: EMA change over the slope window, normalised.
        slope = 0.0
        if trend[-1] is not None and trend[-1 - self._slope_period] is not None:
            past = trend[-1 - self._slope_period]
            if past:
                slope = (trend[-1] - past) / past
        features["trend_slope"] = slope
        if trend[-1] is not None:
            features["trend_ema"] = trend[-1]
            features["distance_from_trend"] = (current - trend[-1]) / trend[-1]

        # Volatility percentile, measured on ATR **as a fraction of price**.
        #
        # Ranking raw ATR is wrong: ATR scales with price, so in any
        # sustained trend it drifts upward and the percentile saturates
        # near 1.0, labelling every trending market as high-volatility.
        # Normalising by price makes the measure comparable across levels.
        atr_series = atr(bars, self._vol_period)
        atr_pct_series: list[float | None] = [
            (a / c if a is not None and c else None)
            for a, c in zip(atr_series, price_closes)
        ]
        recent = [v for v in atr_pct_series[-self._vol_lookback :] if v is not None]
        vol_percentile = 0.0
        current_atr_pct = atr_pct_series[-1]
        if recent and current_atr_pct is not None:
            below = sum(1 for v in recent if v < current_atr_pct)
            vol_percentile = below / len(recent)
            if atr_series[-1] is not None:
                features["atr"] = atr_series[-1]
            features["atr_pct"] = current_atr_pct
        features["vol_percentile"] = vol_percentile

        std = rolling_std(price_closes, self._vol_period)
        if std[-1] is not None and current:
            features["realised_vol_pct"] = std[-1] / current

        # High volatility takes precedence: it changes how every other
        # signal should be sized, so it must not be masked by a trend label.
        #
        # Requires BOTH a high percentile rank AND a meaningful magnitude
        # elevation over the median. Rank alone is not enough: on any
        # smoothly drifting series the current value is always the highest
        # or lowest of its window, so a percentile saturates at 0 or 1 and
        # carries no information about how *much* volatility changed.
        elevated = False
        if recent and current_atr_pct is not None:
            ordered = sorted(recent)
            median = ordered[len(ordered) // 2]
            if median > 0:
                ratio = current_atr_pct / median
                features["vol_ratio_to_median"] = ratio
                elevated = ratio >= self._high_vol_ratio

        if vol_percentile >= self._high_vol_percentile and elevated:
            return RegimeAssessment(
                regime=MarketRegime.HIGH_VOLATILITY,
                confidence=min(1.0, vol_percentile),
                features=features,
                rationale=(
                    f"Volatility at {vol_percentile:.0%} of its {self._vol_lookback}-bar "
                    f"range and {features.get('vol_ratio_to_median', 0):.1f}x the median"
                ),
            )

        if slope >= self._trend_threshold:
            return RegimeAssessment(
                regime=MarketRegime.TRENDING_UP,
                confidence=min(1.0, slope / (self._trend_threshold * 3)),
                features=features,
                rationale=f"Trend EMA rising {slope:.2%} over {self._slope_period} bars",
            )
        if slope <= -self._trend_threshold:
            return RegimeAssessment(
                regime=MarketRegime.TRENDING_DOWN,
                confidence=min(1.0, abs(slope) / (self._trend_threshold * 3)),
                features=features,
                rationale=f"Trend EMA falling {slope:.2%} over {self._slope_period} bars",
            )

        return RegimeAssessment(
            regime=MarketRegime.RANGING,
            confidence=1.0 - min(1.0, abs(slope) / self._trend_threshold),
            features=features,
            rationale=f"Trend slope {slope:.2%} below threshold {self._trend_threshold:.2%}",
        )
