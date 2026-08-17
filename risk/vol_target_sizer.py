"""
Volatility-targeted position sizing.

The idea, used by most systematic risk-parity and CTA-style strategies:
size each position so it contributes roughly the SAME risk to the
portfolio, rather than the same dollar notional or the same fixed
fraction of equity. A volatile instrument gets a smaller position; a
quiet one gets a larger position; both end up contributing a similar
amount of expected variance. This is a different axis from
`PositionSizer`'s risk-per-trade sizing (which sizes off distance to a
stop) and from `KellyPositionSizer`'s edge-based sizing (which sizes off
historical win rate) -- all three answer "how big" from a different
question, and any of them can be plugged into `RiskEngine` interchangeably
since they share the same `.calculate()` shape.

Design decisions:

- **Volatility is derived from ATR, reusing data the risk engine already
  has.** `RiskEngine.evaluate()` already threads an `atr` value through to
  the sizer for the existing `PositionSizer`'s volatility-adjusted stop
  path; this sizer converts that same dollar-denominated ATR into an
  annualised percentage estimate via `atr / price * sqrt(periods_per_year)`
  rather than requiring separate plumbing for a whole return series. This
  is standard practice (ATR is essentially an average absolute daily
  move) but is an approximation, not a fitted volatility model, and is
  documented as such.

- **A volatility floor prevents the inverse relationship from blowing up.**
  As volatility approaches zero (e.g. stale or thin data producing a tiny
  ATR), 1/vol grows without bound, and an ordinary systematic sizer would
  hugely oversize into exactly the situation -- illiquid or barely-moving
  markets -- where that is most dangerous. A floor keeps the implied
  position bounded even when the volatility input looks artificially low.

- **No ATR, no size.** Matches every other sizer here: unknown risk gets
  zero, not a plausible-looking default.

- **This is still bounded by every downstream risk check** in exactly the
  same way `KellyPositionSizer` is -- see that module's docstring for why
  that matters.
"""

from __future__ import annotations

import math
from decimal import ROUND_DOWN, Decimal

import structlog
from pydantic import BaseModel

log = structlog.get_logger(__name__)


class VolTargetSizingResult(BaseModel):
    quantity: Decimal
    method: str
    detail: str = ""
    annualised_volatility: float | None = None
    target_weight: float | None = None
    risk_amount: Decimal = Decimal("0")  # interface parity with SizingResult

    @property
    def is_tradeable(self) -> bool:
        return self.quantity > 0


class VolatilityTargetPositionSizer:
    def __init__(
        self,
        *,
        target_annual_volatility: Decimal = Decimal("0.10"),
        periods_per_year: float = 252.0,
        min_annual_volatility: Decimal = Decimal("0.03"),
        max_position_size: Decimal = Decimal("0.10"),
        allow_fractional: bool = False,
    ) -> None:
        if target_annual_volatility <= 0:
            raise ValueError("target_annual_volatility must be positive")
        self._target_vol = target_annual_volatility
        self._periods_per_year = periods_per_year
        self._min_vol = min_annual_volatility
        self._max_position_size = max_position_size
        self._allow_fractional = allow_fractional

    def calculate(
        self,
        *,
        equity: Decimal,
        entry_price: Decimal,
        stop_price: Decimal | None = None,
        atr: Decimal | None = None,
        requested_quantity: Decimal | None = None,
        strategy: str | None = None,
    ) -> VolTargetSizingResult:
        if equity <= 0 or entry_price <= 0:
            return VolTargetSizingResult(
                quantity=Decimal("0"), method="none",
                detail="Non-positive equity or entry price",
            )
        if atr is None or atr <= 0:
            return VolTargetSizingResult(
                quantity=Decimal("0"),
                method="none",
                detail="No ATR-based volatility estimate available; refusing to size",
            )

        daily_vol_pct = float(atr / entry_price)
        annualised_vol = Decimal(str(daily_vol_pct * math.sqrt(self._periods_per_year)))
        floored = annualised_vol < self._min_vol
        effective_vol = max(annualised_vol, self._min_vol)

        weight = min(self._target_vol / effective_vol, self._max_position_size)
        capped_by_position_limit = (self._target_vol / effective_vol) > self._max_position_size

        notional = equity * weight
        raw_quantity = notional / entry_price

        reduced_by_request = False
        if requested_quantity is not None and requested_quantity < raw_quantity:
            raw_quantity = requested_quantity
            reduced_by_request = True

        quantity = self._round(raw_quantity)

        details = [
            f"annualised vol {float(annualised_vol):.1%}"
            + (f" (floored from {daily_vol_pct*math.sqrt(self._periods_per_year):.1%})"
               if floored else ""),
            f"target {float(self._target_vol):.1%} -> weight {float(weight):.1%} of equity",
        ]
        if capped_by_position_limit:
            details.append(f"capped at max_position_size {float(self._max_position_size):.0%}")
        if reduced_by_request:
            details.append("limited to requested quantity")

        return VolTargetSizingResult(
            quantity=quantity,
            method="volatility_target",
            detail="; ".join(details),
            annualised_volatility=float(effective_vol),
            target_weight=float(weight),
        )

    def _round(self, quantity: Decimal) -> Decimal:
        if quantity <= 0:
            return Decimal("0")
        if self._allow_fractional:
            return quantity.quantize(Decimal("0.0001"), rounding=ROUND_DOWN)
        return quantity.quantize(Decimal("1"), rounding=ROUND_DOWN)
