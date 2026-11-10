"""
Buy the dip (short-term pullback inside a long-term uptrend).

**Framework demonstration only — no profitability is claimed or implied.**
Dip-buying is a mean-reversion construction and inherits its failure
mode: many small wins, then a large loss when a "dip" turns out to be the
start of a decline. The trend filter and the trend-break exit exist to
cap that loss, not to remove it.

Construction:

- **Trend filter.** Only buy when the close is above its long SMA
  (`trend_period`, default 200). This is the difference from
  `mean_reversion`, which buys any stretched price — including one that
  is collapsing.
- **Dip trigger.** A very short RSI (`rsi_period`, default 2) below
  `entry_rsi`, optionally combined with a minimum pullback from the
  recent `dip_lookback`-bar high. Short RSI is used because a 14-period
  RSI barely moves on a two- or three-bar pullback.
- **Exits.** Close when the bounce has happened (RSI above `exit_rsi` or
  close back above the short `exit_ma_period` SMA), or when the trend
  breaks (close below the trend SMA). The ATR-derived bracket stop from
  `Strategy.generate_order_intent` remains the hard floor.

Design decisions:

- **Long only, by construction.** This strategy never emits SHORT. If it
  finds itself holding a short (e.g. one adopted during reconciliation),
  it returns NONE and leaves it alone: covering someone else's short is
  not a dip-buying decision.
- **The pullback high excludes the current bar**, for the same look-ahead
  reason as the trend-following channel: including it would let the
  current bar define the high it is measured against.
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
from strategies.indicators import atr, rsi, sma
from strategies.registry import register_strategy


class BuyTheDipParams(StrategyParams):
    trend_period: int = Field(default=200, gt=1)
    rsi_period: int = Field(default=2, gt=0)
    entry_rsi: float = Field(default=10.0, gt=0, lt=100)
    exit_rsi: float = Field(default=70.0, gt=0, lt=100)
    exit_ma_period: int = Field(default=5, gt=1)
    dip_lookback: int = Field(default=20, gt=1)
    min_pullback_pct: float = Field(default=0.0, ge=0, lt=1)
    atr_period: int = Field(default=14, gt=1)

    def model_post_init(self, _: object) -> None:
        if self.exit_rsi <= self.entry_rsi:
            raise ValueError("exit_rsi must be greater than entry_rsi")
        if self.exit_ma_period >= self.trend_period:
            raise ValueError("exit_ma_period must be shorter than trend_period")


@register_strategy
class BuyTheDipStrategy(Strategy):
    name = "buy_the_dip"
    version = "0.1.0"

    def __init__(self, params: StrategyParams | None = None) -> None:
        super().__init__(params or BuyTheDipParams())

    @property
    def min_bars(self) -> int:
        p = self.params
        return max(p.trend_period, p.dip_lookback + 1, p.atr_period, p.rsi_period) + 2

    def calculate_features(self, context: StrategyContext) -> dict[str, float]:
        p = self.params
        closes = context.closes
        features: dict[str, float] = {}
        if not closes:
            return features

        trend = sma(closes, p.trend_period)[-1]
        exit_ma = sma(closes, p.exit_ma_period)[-1]
        rsi_value = rsi(closes, p.rsi_period)[-1]
        atr_value = atr(context.bars, p.atr_period)[-1]

        if trend is not None:
            features["trend_sma"] = trend
            features["above_trend"] = 1.0 if closes[-1] > trend else 0.0
        if exit_ma is not None:
            features["exit_sma"] = exit_ma
        if rsi_value is not None:
            features["rsi"] = rsi_value
        if atr_value is not None:
            features["atr"] = atr_value

        window = context.bars[-p.dip_lookback - 1 : -1]
        if len(window) == p.dip_lookback:
            recent_high = max(b.high for b in window)
            if recent_high > 0:
                features["pullback_pct"] = max(0.0, (recent_high - closes[-1]) / recent_high)
        return features

    def generate_signal(self, context: StrategyContext) -> Signal:
        if not self.has_enough_history(context):
            return self.no_signal(context, f"need {self.min_bars} bars")

        f = self.calculate_features(context)
        required = ("trend_sma", "exit_sma", "rsi", "pullback_pct")
        if any(k not in f for k in required):
            return self.no_signal(context, "indicators not yet defined")

        p = self.params
        close = context.closes[-1]
        r = f["rsi"]

        if context.is_short:
            return self.no_signal(context, "holding a short; long-only strategy leaves it alone")

        if context.is_long:
            if close < f["trend_sma"]:
                return self._flat(
                    context,
                    f,
                    f"trend broken: close {close:.2f} < SMA{p.trend_period} {f['trend_sma']:.2f}",
                )
            if r >= p.exit_rsi:
                return self._flat(
                    context, f, f"bounce done: RSI{p.rsi_period} {r:.1f} >= {p.exit_rsi}"
                )
            if close > f["exit_sma"]:
                return self._flat(
                    context,
                    f,
                    f"bounce done: close {close:.2f} > SMA{p.exit_ma_period} {f['exit_sma']:.2f}",
                )
            return self.no_signal(context, f"holding dip; RSI{p.rsi_period} {r:.1f}")

        if close <= f["trend_sma"]:
            return self.no_signal(context, "below trend SMA; not buying a falling stock")
        if r > p.entry_rsi:
            return self.no_signal(context, f"no dip: RSI{p.rsi_period} {r:.1f} > {p.entry_rsi}")
        if f["pullback_pct"] < p.min_pullback_pct:
            return self.no_signal(
                context,
                f"pullback {f['pullback_pct']:.1%} below minimum {p.min_pullback_pct:.1%}",
            )

        # Deeper RSI washout -> higher conviction, floored at 0.5.
        strength = min(1.0, 0.5 + 0.5 * (p.entry_rsi - r) / p.entry_rsi)
        return Signal(
            instrument=context.instrument,
            direction=SignalDirection.LONG,
            strength=strength,
            strategy=self.name,
            features=f,
            rationale=(
                f"dip in uptrend: RSI{p.rsi_period} {r:.1f} <= {p.entry_rsi}, "
                f"{f['pullback_pct']:.1%} off {p.dip_lookback}-bar high, "
                f"above SMA{p.trend_period}"
            ),
        )

    def _flat(self, context: StrategyContext, features: dict[str, float], why: str) -> Signal:
        return Signal(
            instrument=context.instrument,
            direction=SignalDirection.FLAT,
            strength=1.0,
            strategy=self.name,
            features=features,
            rationale=why,
        )
