"""
Trend following (Donchian-style breakout with a trend filter).

**Framework demonstration only — no profitability is claimed or implied.**

Construction: enter on a breakout of the N-bar high/low, filtered by a
long EMA so we only take breakouts in the direction of the longer trend.
Exit on a breakout of the shorter opposite channel.

Design note on look-ahead: the breakout channel is computed from bars
`[-lookback-1:-1]`, i.e. *excluding the current bar*. Including the
current bar would mean the current high is always part of the channel it
is being compared against, so a breakout could never be detected — or
worse, with intrabar data, would use information not yet available at
decision time. This is the single most common look-ahead bug in breakout
systems.
"""

from __future__ import annotations

from pydantic import Field

from strategies.base import (
    Signal,
    SignalDirection,
    Strategy,
    StrategyContext,
    StrategyParams,
)
from strategies.indicators import atr, ema
from strategies.registry import register_strategy


class TrendFollowingParams(StrategyParams):
    entry_lookback: int = Field(default=20, gt=1)
    exit_lookback: int = Field(default=10, gt=1)
    trend_ema_period: int = Field(default=100, gt=1)
    atr_period: int = Field(default=14, gt=1)
    allow_short: bool = True
    use_trend_filter: bool = True


@register_strategy
class TrendFollowingStrategy(Strategy):
    name = "trend_following"
    version = "0.1.0"

    def __init__(self, params: StrategyParams | None = None) -> None:
        super().__init__(params or TrendFollowingParams())

    @property
    def min_bars(self) -> int:
        p = self.params
        return max(p.entry_lookback, p.exit_lookback, p.trend_ema_period, p.atr_period) + 2

    def calculate_features(self, context: StrategyContext) -> dict[str, float]:
        p = self.params
        bars = context.bars
        closes = context.closes

        # Channels EXCLUDE the current bar — see module docstring.
        entry_window = bars[-p.entry_lookback - 1 : -1]
        exit_window = bars[-p.exit_lookback - 1 : -1]

        features: dict[str, float] = {"close": closes[-1]}
        if entry_window:
            features["channel_high"] = max(b.high for b in entry_window)
            features["channel_low"] = min(b.low for b in entry_window)
        if exit_window:
            features["exit_high"] = max(b.high for b in exit_window)
            features["exit_low"] = min(b.low for b in exit_window)

        trend = ema(closes, p.trend_ema_period)
        if trend[-1] is not None:
            features["trend_ema"] = trend[-1]

        atr_series = atr(bars, p.atr_period)
        if atr_series[-1] is not None:
            features["atr"] = atr_series[-1]
        return features

    def generate_signal(self, context: StrategyContext) -> Signal:
        if not self.has_enough_history(context):
            return self.no_signal(context, f"need {self.min_bars} bars")

        f = self.calculate_features(context)
        if "channel_high" not in f or "channel_low" not in f:
            return self.no_signal(context, "channel not yet defined")

        p = self.params
        close = f["close"]

        # Exits take priority over entries.
        if context.is_long and "exit_low" in f and close < f["exit_low"]:
            return Signal(
                instrument=context.instrument,
                direction=SignalDirection.FLAT,
                strength=1.0,
                strategy=self.name,
                features=f,
                rationale=f"close {close:.2f} broke below {p.exit_lookback}-bar low",
            )
        if context.is_short and "exit_high" in f and close > f["exit_high"]:
            return Signal(
                instrument=context.instrument,
                direction=SignalDirection.FLAT,
                strength=1.0,
                strategy=self.name,
                features=f,
                rationale=f"close {close:.2f} broke above {p.exit_lookback}-bar high",
            )

        trend_ema = f.get("trend_ema")
        uptrend = trend_ema is None or close > trend_ema
        downtrend = trend_ema is None or close < trend_ema
        if not p.use_trend_filter:
            uptrend = downtrend = True

        if close > f["channel_high"] and context.is_flat:
            if not uptrend:
                return self.no_signal(context, "breakout against longer-term trend")
            direction = SignalDirection.LONG
            rationale = f"close {close:.2f} broke {p.entry_lookback}-bar high {f['channel_high']:.2f}"
        elif close < f["channel_low"] and context.is_flat:
            if not p.allow_short:
                return self.no_signal(context, "short entries disabled")
            if not downtrend:
                return self.no_signal(context, "breakdown against longer-term trend")
            direction = SignalDirection.SHORT
            rationale = f"close {close:.2f} broke {p.entry_lookback}-bar low {f['channel_low']:.2f}"
        else:
            return self.no_signal(context, "no breakout")

        atr_value = f.get("atr", 0.0)
        strength = 0.5
        if atr_value:
            excess = abs(close - (f["channel_high"] if direction is SignalDirection.LONG else f["channel_low"]))
            strength = min(1.0, 0.4 + excess / atr_value)

        return Signal(
            instrument=context.instrument,
            direction=direction,
            strength=strength,
            strategy=self.name,
            features=f,
            rationale=rationale,
        )
