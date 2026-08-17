"""
Momentum.

**Framework demonstration only — no profitability is claimed or implied.**

Construction: rate-of-change over a lookback, gated by an RSI filter that
suppresses entries into already-extended moves. The RSI gate is included
mainly to demonstrate multi-feature signal logic and how features are
surfaced for the audit trail and AI analysis.
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
from strategies.indicators import atr, roc, rsi
from strategies.registry import register_strategy


class MomentumParams(StrategyParams):
    lookback: int = Field(default=20, gt=1)
    entry_threshold: float = Field(default=0.03, gt=0)
    rsi_period: int = Field(default=14, gt=1)
    rsi_overbought: float = Field(default=75.0, gt=50, le=100)
    rsi_oversold: float = Field(default=25.0, ge=0, lt=50)
    atr_period: int = Field(default=14, gt=1)
    allow_short: bool = True


@register_strategy
class MomentumStrategy(Strategy):
    name = "momentum"
    version = "0.1.0"

    def __init__(self, params: StrategyParams | None = None) -> None:
        super().__init__(params or MomentumParams())

    @property
    def min_bars(self) -> int:
        p = self.params
        return max(p.lookback, p.rsi_period, p.atr_period) + 2

    def calculate_features(self, context: StrategyContext) -> dict[str, float]:
        p = self.params
        closes = context.closes
        roc_series = roc(closes, p.lookback)
        rsi_series = rsi(closes, p.rsi_period)
        atr_series = atr(context.bars, p.atr_period)

        features: dict[str, float] = {}
        if roc_series[-1] is not None:
            features["roc"] = roc_series[-1]
        if rsi_series[-1] is not None:
            features["rsi"] = rsi_series[-1]
        if atr_series[-1] is not None:
            features["atr"] = atr_series[-1]
            if closes[-1]:
                features["atr_pct"] = atr_series[-1] / closes[-1]
        return features

    def generate_signal(self, context: StrategyContext) -> Signal:
        if not self.has_enough_history(context):
            return self.no_signal(context, f"need {self.min_bars} bars")

        f = self.calculate_features(context)
        if "roc" not in f or "rsi" not in f:
            return self.no_signal(context, "indicators not yet defined")

        p = self.params
        momentum, rsi_value = f["roc"], f["rsi"]

        if momentum >= p.entry_threshold:
            if rsi_value >= p.rsi_overbought:
                return self.no_signal(
                    context, f"positive momentum but RSI {rsi_value:.1f} overbought"
                )
            direction = SignalDirection.LONG
            rationale = f"momentum {momentum:.2%} over {p.lookback} bars, RSI {rsi_value:.1f}"
        elif momentum <= -p.entry_threshold:
            if rsi_value <= p.rsi_oversold:
                return self.no_signal(
                    context, f"negative momentum but RSI {rsi_value:.1f} oversold"
                )
            if not p.allow_short:
                return self.no_signal(context, "short entries disabled")
            direction = SignalDirection.SHORT
            rationale = f"momentum {momentum:.2%} over {p.lookback} bars, RSI {rsi_value:.1f}"
        else:
            return self.no_signal(context, f"momentum {momentum:.2%} below threshold")

        strength = min(1.0, abs(momentum) / (p.entry_threshold * 3))
        return Signal(
            instrument=context.instrument,
            direction=direction,
            strength=strength,
            strategy=self.name,
            features=f,
            rationale=rationale,
        )
