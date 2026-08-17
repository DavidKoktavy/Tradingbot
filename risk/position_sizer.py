"""
Position sizing.

Design decisions:

- Sizing is *deterministic and independent of the AI's requested size*.
  The AI or a strategy may propose a quantity; the sizer computes what is
  actually permissible and the smaller of the two wins. An AI that asks
  for 10,000 shares gets whatever the risk budget allows, silently and
  without argument.

- The primary method is risk-based: risk per trade is a fraction of
  equity, and the distance to the stop determines share count. This makes
  position size inversely proportional to the risk being taken, which is
  the whole point — a wide stop should mean a smaller position, not the
  same position with more risk.

- If no stop is supplied we fall back to a volatility-based stop distance
  (ATR multiple). If neither a stop nor volatility is available, we return
  zero rather than guessing a size. Sizing a position with no idea of its
  risk is exactly the situation where "prefer stopping" applies.

- Every result is floored at zero and capped by max_position_size as a
  fraction of equity, so no single sizing path can produce an
  outsized position even if the inputs are wrong.

- Fractional shares are truncated *down* to whole shares. Rounding up
  would systematically breach limits by a fraction of a share on every
  trade.
"""

from __future__ import annotations

from decimal import ROUND_DOWN, Decimal

import structlog
from pydantic import BaseModel

log = structlog.get_logger(__name__)


class SizingResult(BaseModel):
    quantity: Decimal
    method: str
    detail: str = ""
    risk_amount: Decimal = Decimal("0")

    @property
    def is_tradeable(self) -> bool:
        return self.quantity > 0


class PositionSizer:
    def __init__(
        self,
        *,
        max_risk_per_trade: Decimal,
        max_position_size: Decimal,
        default_atr_stop_multiple: Decimal = Decimal("2"),
        allow_fractional: bool = False,
    ) -> None:
        self._max_risk_per_trade = Decimal(str(max_risk_per_trade))
        self._max_position_size = Decimal(str(max_position_size))
        self._atr_multiple = Decimal(str(default_atr_stop_multiple))
        self._allow_fractional = allow_fractional

    def calculate(
        self,
        *,
        equity: Decimal,
        entry_price: Decimal,
        stop_price: Decimal | None = None,
        atr: Decimal | None = None,
        requested_quantity: Decimal | None = None,
        strategy: str | None = None,  # unused here; accepted for interface
        # parity with KellyPositionSizer/VolatilityTargetPositionSizer, so
        # RiskEngine can call any of the three identically and they can be
        # swapped via RiskEngine(position_sizer=...) with no other change.
    ) -> SizingResult:
        if equity <= 0 or entry_price <= 0:
            return SizingResult(
                quantity=Decimal("0"),
                method="none",
                detail="Non-positive equity or entry price",
            )

        risk_budget = equity * self._max_risk_per_trade
        stop_distance, method = self._stop_distance(entry_price, stop_price, atr)

        if stop_distance is None or stop_distance <= 0:
            return SizingResult(
                quantity=Decimal("0"),
                method="none",
                detail=(
                    "No stop price and no volatility estimate available — "
                    "refusing to size a position of unknown risk"
                ),
            )

        raw_quantity = risk_budget / stop_distance

        # Hard cap: notional must not exceed max_position_size of equity.
        max_notional = equity * self._max_position_size
        notional_cap_qty = max_notional / entry_price
        quantity = min(raw_quantity, notional_cap_qty)
        capped_by_notional = notional_cap_qty < raw_quantity

        # Never exceed what was actually requested.
        reduced_by_request = False
        if requested_quantity is not None and requested_quantity < quantity:
            quantity = requested_quantity
            reduced_by_request = True

        quantity = self._round(quantity)

        details = []
        if capped_by_notional:
            details.append("capped by max_position_size")
        if reduced_by_request:
            details.append("limited to requested quantity")

        result = SizingResult(
            quantity=quantity,
            method=method,
            risk_amount=quantity * stop_distance,
            detail="; ".join(details) or f"risk budget {risk_budget} / stop distance {stop_distance}",
        )
        log.debug(
            "sizing.calculated",
            method=method,
            quantity=str(quantity),
            risk_budget=str(risk_budget),
            stop_distance=str(stop_distance),
        )
        return result

    def _stop_distance(
        self, entry_price: Decimal, stop_price: Decimal | None, atr: Decimal | None
    ) -> tuple[Decimal | None, str]:
        if stop_price is not None and stop_price > 0:
            distance = abs(entry_price - stop_price)
            if distance > 0:
                return distance, "risk_based"
            return None, "none"
        if atr is not None and atr > 0:
            return atr * self._atr_multiple, "volatility_adjusted"
        return None, "none"

    def _round(self, quantity: Decimal) -> Decimal:
        if quantity <= 0:
            return Decimal("0")
        if self._allow_fractional:
            return quantity.quantize(Decimal("0.0001"), rounding=ROUND_DOWN)
        # Truncate down: rounding up would breach limits on every trade.
        return quantity.quantize(Decimal("1"), rounding=ROUND_DOWN)
