"""
Mean reversion.

**Framework demonstration only — no profitability is claimed or implied.**
Mean-reversion constructions are particularly prone to looking excellent
in a backtest and failing live, because they sell volatility: many small
wins punctuated by rare large losses when the mean shifts. Treat backtest
results from this strategy with more suspicion than usual.

Construction: z-score of price against its own rolling mean. Entry when
price is stretched beyond `entry_z`; exit when it reverts inside `exit_z`.
The separate exit band prevents the position from being closed and
reopened repeatedly around a single threshold.
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
from strategies.indicators import atr, rsi, zscore
from strategies.registry import register_strategy


class MeanReversionParams(StrategyParams):
    lookback: int = Field(default=20, gt=2)
    entry_z: float = Field(default=2.0, gt=0)
    exit_z: float = Field(default=0.5, ge=0)
    atr_period: int = Field(default=14, gt=1)
    allow_short: bool = True

    def model_post_init(self, _: object) -> None:
        if self.exit_z >= self.entry_z:
            raise ValueError("exit_z must be less than entry_z")


@register_strategy
class MeanReversionStrategy(Strategy):
    name = "mean_reversion"
    version = "0.1.0"

    def __init__(self, params: StrategyParams | None = None) -> None:
        super().__init__(params or MeanReversionParams())

    @property
    def min_bars(self) -> int:
        p = self.params
        return max(p.lookback, p.atr_period) + 2

    def calculate_features(self, context: StrategyContext) -> dict[str, float]:
        p = self.params
        closes = context.closes
        z_series = zscore(closes, p.lookback)
        atr_series = atr(context.bars, p.atr_period)

        features: dict[str, float] = {}
        if z_series[-1] is not None:
            features["zscore"] = z_series[-1]
        if atr_series[-1] is not None:
            features["atr"] = atr_series[-1]
        return features

    def generate_signal(self, context: StrategyContext) -> Signal:
        if not self.has_enough_history(context):
            return self.no_signal(context, f"need {self.min_bars} bars")

        f = self.calculate_features(context)
        if "zscore" not in f:
            return self.no_signal(context, "z-score not yet defined")

        p = self.params
        z = f["zscore"]

        # Exit first: if we're in a position and price has reverted, close.
        if not context.is_flat and abs(z) <= p.exit_z:
            return Signal(
                instrument=context.instrument,
                direction=SignalDirection.FLAT,
                strength=1.0,
                strategy=self.name,
                features=f,
                rationale=f"z-score {z:.2f} reverted inside exit band {p.exit_z}",
            )

        if z <= -p.entry_z and context.is_flat:
            direction = SignalDirection.LONG
            rationale = f"z-score {z:.2f} stretched below -{p.entry_z}"
        elif z >= p.entry_z and context.is_flat:
            if not p.allow_short:
                return self.no_signal(context, "short entries disabled")
            direction = SignalDirection.SHORT
            rationale = f"z-score {z:.2f} stretched above {p.entry_z}"
        else:
            return self.no_signal(context, f"z-score {z:.2f} within bands")

        strength = min(1.0, (abs(z) - p.entry_z) / p.entry_z + 0.5)
        return Signal(
            instrument=context.instrument,
            direction=direction,
            strength=strength,
            strategy=self.name,
            features=f,
            rationale=rationale,
        )
