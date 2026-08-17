"""
Moving-average crossover.

**This is a framework demonstration, not a profitable strategy.** MA
crossover is the textbook example precisely because it is simple and
well-understood; it is widely documented to perform poorly net of costs in
most markets and regimes. It exists here to show the interface.

Design note: the crossover is detected by comparing the *previous* bar's
relationship to the current bar's, so a signal fires on the bar where the
cross completes — not on every subsequent bar while fast > slow. Firing
continuously would produce a duplicate order on every tick, which the
order store would reject, masking the underlying logic error.
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
from strategies.indicators import atr, sma
from strategies.registry import register_strategy


class MACrossoverParams(StrategyParams):
    fast_period: int = Field(default=20, gt=1)
    slow_period: int = Field(default=50, gt=1)
    atr_period: int = Field(default=14, gt=1)
    allow_short: bool = True

    def model_post_init(self, _: object) -> None:
        if self.fast_period >= self.slow_period:
            raise ValueError("fast_period must be less than slow_period")


@register_strategy
class MACrossoverStrategy(Strategy):
    name = "ma_crossover"
    version = "0.1.0"

    def __init__(self, params: StrategyParams | None = None) -> None:
        super().__init__(params or MACrossoverParams())
        assert isinstance(self.params, MACrossoverParams)

    @property
    def min_bars(self) -> int:
        p = self.params
        return max(p.slow_period, p.atr_period) + 2

    def calculate_features(self, context: StrategyContext) -> dict[str, float]:
        p = self.params
        closes = context.closes
        fast = sma(closes, p.fast_period)
        slow = sma(closes, p.slow_period)
        atr_series = atr(context.bars, p.atr_period)

        features: dict[str, float] = {}
        if fast[-1] is not None:
            features["fast_ma"] = fast[-1]
        if slow[-1] is not None:
            features["slow_ma"] = slow[-1]
        if fast[-2] is not None and slow[-2] is not None:
            features["prev_fast_ma"] = fast[-2]
            features["prev_slow_ma"] = slow[-2]
        if atr_series[-1] is not None:
            features["atr"] = atr_series[-1]
        if fast[-1] is not None and slow[-1] is not None and slow[-1] != 0:
            features["ma_spread_pct"] = (fast[-1] - slow[-1]) / slow[-1]
        return features

    def generate_signal(self, context: StrategyContext) -> Signal:
        if not self.has_enough_history(context):
            return self.no_signal(context, f"need {self.min_bars} bars")

        f = self.calculate_features(context)
        required = ("fast_ma", "slow_ma", "prev_fast_ma", "prev_slow_ma")
        if not all(k in f for k in required):
            return self.no_signal(context, "indicators not yet defined")

        was_above = f["prev_fast_ma"] > f["prev_slow_ma"]
        is_above = f["fast_ma"] > f["slow_ma"]

        if is_above and not was_above:
            direction, rationale = SignalDirection.LONG, "fast MA crossed above slow MA"
        elif not is_above and was_above:
            if self.params.allow_short:
                direction, rationale = SignalDirection.SHORT, "fast MA crossed below slow MA"
            else:
                direction, rationale = SignalDirection.FLAT, "fast MA crossed below; exiting long"
        else:
            return self.no_signal(context, "no crossover on this bar")

        strength = min(1.0, abs(f.get("ma_spread_pct", 0.0)) * 20)
        return Signal(
            instrument=context.instrument,
            direction=direction,
            strength=strength,
            strategy=self.name,
            features=f,
            rationale=rationale,
        )
