"""
Transaction cost model.

Design decisions:

- Costs are applied **pessimistically and always**. There is no
  "frictionless" mode, because a frictionless backtest is not a simplified
  backtest — it is a different and much more profitable strategy than the
  one you would actually run. Anyone wanting zero costs must explicitly
  construct a `CostModel` with zeros and will see that in the config.

- Fills cross the spread. A buy pays the ask, a sell receives the bid.
  Filling at the mid (or worse, the close) is the most common way a
  backtest manufactures returns that don't exist: for a strategy trading
  frequently, half the spread per side is often larger than the entire
  edge.

- Slippage has a fixed component (in basis points) and a size-dependent
  component (market impact scaled by participation in the bar's volume).
  A strategy that only looks profitable at small size should show
  degradation as size grows, rather than scaling linearly forever.

- Latency is modelled as a delay between decision and execution: a signal
  generated on bar `i` executes at bar `i + latency_bars`, defaulting to
  the *next* bar's open. Executing at the close of the bar that generated
  the signal is look-ahead — that price was not knowable at decision time.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, Field

from execution.execution_models import OrderSide


class CostModel(BaseModel):
    """All costs a simulated fill incurs. Defaults are deliberately
    conservative for liquid US equities; they are not a promise of what
    any particular broker charges."""

    # Commission
    commission_per_share: Decimal = Decimal("0.005")
    min_commission: Decimal = Decimal("1.00")
    commission_pct: Decimal = Decimal("0")

    # Spread: if the data has no bid/ask, assume this fraction of price.
    assumed_spread_bps: Decimal = Decimal("2")

    # Slippage
    fixed_slippage_bps: Decimal = Decimal("1")
    impact_coefficient: Decimal = Decimal("0.1")
    max_participation: Decimal = Field(default=Decimal("0.1"), gt=0, le=1)

    # Latency: bars between decision and execution. 1 = execute next bar.
    latency_bars: int = Field(default=1, ge=0)

    def commission(self, quantity: Decimal, price: Decimal) -> Decimal:
        per_share = self.commission_per_share * quantity
        percentage = self.commission_pct * quantity * price
        return max(per_share + percentage, self.min_commission)

    def half_spread(self, price: Decimal, bid: Decimal | None, ask: Decimal | None) -> Decimal:
        if bid is not None and ask is not None and ask > bid > 0:
            return (ask - bid) / 2
        return price * self.assumed_spread_bps / Decimal("10000")

    def slippage(
        self, price: Decimal, quantity: Decimal, bar_volume: Decimal | None
    ) -> Decimal:
        """Fixed cost plus a market-impact term that grows with the
        fraction of the bar's volume we are demanding."""
        fixed = price * self.fixed_slippage_bps / Decimal("10000")
        impact = Decimal("0")
        if bar_volume and bar_volume > 0:
            participation = quantity / bar_volume
            impact = price * self.impact_coefficient * participation
        return fixed + impact

    def fill_price(
        self,
        *,
        reference_price: Decimal,
        side: OrderSide,
        quantity: Decimal,
        bid: Decimal | None = None,
        ask: Decimal | None = None,
        bar_volume: Decimal | None = None,
    ) -> Decimal:
        """Simulated fill price: cross the spread, then pay slippage.
        Always adverse to the trader, in both directions."""
        half = self.half_spread(reference_price, bid, ask)
        slip = self.slippage(reference_price, quantity, bar_volume)
        adverse = half + slip
        price = reference_price + adverse if side is OrderSide.BUY else reference_price - adverse
        return max(price, Decimal("0.01"))

    def max_fillable(self, bar_volume: Decimal | None) -> Decimal | None:
        """Cap on how much of a bar's volume a single order may consume.
        Returns None when volume is unknown (no cap applied, but the
        caller is warned)."""
        if not bar_volume or bar_volume <= 0:
            return None
        return bar_volume * self.max_participation


ZERO_COST_MODEL = CostModel(
    commission_per_share=Decimal("0"),
    min_commission=Decimal("0"),
    assumed_spread_bps=Decimal("0"),
    fixed_slippage_bps=Decimal("0"),
    impact_coefficient=Decimal("0"),
)
"""Explicitly frictionless. Provided ONLY for isolating logic in unit
tests. Results from this model are not a backtest of anything tradeable."""
