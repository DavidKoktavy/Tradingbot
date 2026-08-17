"""
Position accounting.

Design decisions:
- Average-cost basis (not FIFO lot tracking) for Phase 3. This matches how
  IBKR reports position average cost, which keeps reconciliation simple —
  we compare one number against one number. FIFO/tax-lot accounting is a
  reporting concern that can be layered on later from the fill history,
  which we retain in full.
- Realized P&L is booked only on quantity that actually closes exposure.
  A position flip (long 100 -> short 50 via a sell of 150) books P&L on
  the 100 closed and opens a new 50 short at the fill price, rather than
  producing a nonsense average cost.
- All arithmetic in Decimal. See execution_models.py for why.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from pydantic import BaseModel, Field

from data.models import Instrument
from execution.execution_models import Fill, OrderSide


class Position(BaseModel):
    """Net position in a single instrument. quantity > 0 is long,
    quantity < 0 is short, 0 is flat."""

    instrument: Instrument
    quantity: Decimal = Decimal("0")
    average_cost: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    total_commission: Decimal = Decimal("0")
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_flat(self) -> bool:
        return self.quantity == 0

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    def market_value(self, price: Decimal) -> Decimal:
        return self.quantity * price

    def exposure(self, price: Decimal) -> Decimal:
        """Absolute exposure — shorts consume risk budget too."""
        return abs(self.quantity * price)

    def unrealized_pnl(self, price: Decimal) -> Decimal:
        if self.is_flat:
            return Decimal("0")
        return (price - self.average_cost) * self.quantity

    def apply_fill(self, fill: Fill, side: OrderSide) -> None:
        """Update the position from a broker-confirmed fill."""
        signed_qty = fill.quantity if side is OrderSide.BUY else -fill.quantity
        self.total_commission += fill.commission

        if self.is_flat:
            # Opening a new position.
            self.quantity = signed_qty
            self.average_cost = fill.price

        elif (self.quantity > 0) == (signed_qty > 0):
            # Adding to the existing position — blend the cost basis.
            total_cost = self.average_cost * self.quantity + fill.price * signed_qty
            self.quantity += signed_qty
            self.average_cost = total_cost / self.quantity

        else:
            # Reducing, closing, or flipping.
            closing_qty = min(abs(signed_qty), abs(self.quantity))
            direction = Decimal("1") if self.quantity > 0 else Decimal("-1")
            self.realized_pnl += (fill.price - self.average_cost) * closing_qty * direction

            new_quantity = self.quantity + signed_qty
            if new_quantity == 0:
                self.quantity = Decimal("0")
                self.average_cost = Decimal("0")
            elif (new_quantity > 0) == (self.quantity > 0):
                # Partial reduction — cost basis unchanged.
                self.quantity = new_quantity
            else:
                # Flipped through zero: the remainder opens at fill price.
                self.quantity = new_quantity
                self.average_cost = fill.price

        self.updated_at = datetime.now(timezone.utc)
