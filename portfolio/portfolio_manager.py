"""
Portfolio manager: the authoritative in-process view of what we own.

Design decisions:
- The portfolio is derived from broker-confirmed fills only. It never
  updates optimistically on order submission, because an order that was
  submitted is not an order that was filled.
- `daily_pnl` is measured against an explicit `start_of_day_equity`
  snapshot rather than "whatever equity was when the process started".
  Otherwise a mid-session restart silently resets the daily-loss limit,
  which would let the system lose its daily budget twice in one day. This
  is a financial-risk bug of the exact kind the spec warns about, so the
  value is set explicitly and persisted (Phase 5 database work).
- Exposure calculations require a price for every open position. If a
  price is missing we raise rather than substituting a stale or zero
  price — the risk engine must not compute exposure from incomplete data
  and conclude everything is fine.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal

import structlog
from pydantic import BaseModel, Field

from data.models import Instrument
from execution.execution_models import Fill, Order, OrderSide
from portfolio.positions import Position

log = structlog.get_logger(__name__)


class MissingPriceError(Exception):
    """Raised when a portfolio-level calculation needs a mark for an
    instrument and none is available. Fail closed — do not guess."""


class AccountState(BaseModel):
    """Broker-reported account values. Refreshed by reconciliation, not
    computed locally, because the broker is the authority on margin."""

    equity: Decimal = Decimal("0")
    cash: Decimal = Decimal("0")
    buying_power: Decimal = Decimal("0")
    maintenance_margin: Decimal = Decimal("0")
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PortfolioManager:
    def __init__(self, *, start_of_day_equity: Decimal | None = None) -> None:
        self._positions: dict[str, Position] = {}
        self._account = AccountState()
        self._start_of_day_equity: Decimal | None = start_of_day_equity
        self._session_date: date = datetime.now(timezone.utc).date()

    # ---- positions --------------------------------------------------------

    @staticmethod
    def _key(instrument: Instrument) -> str:
        return str(instrument)

    def get_position(self, instrument: Instrument) -> Position:
        key = self._key(instrument)
        if key not in self._positions:
            self._positions[key] = Position(instrument=instrument)
        return self._positions[key]

    @property
    def positions(self) -> dict[str, Position]:
        return dict(self._positions)

    def open_positions(self) -> list[Position]:
        return [p for p in self._positions.values() if not p.is_flat]

    @property
    def open_position_count(self) -> int:
        return len(self.open_positions())

    def apply_fill(self, order: Order, fill: Fill) -> Position:
        """Update portfolio state from a broker-confirmed fill."""
        position = self.get_position(order.intent.instrument)
        position.apply_fill(fill, order.intent.side)
        log.info(
            "portfolio.fill_applied",
            instrument=str(order.intent.instrument),
            side=order.intent.side,
            quantity=str(fill.quantity),
            price=str(fill.price),
            new_position=str(position.quantity),
            realized_pnl=str(position.realized_pnl),
        )
        return position

    # ---- account ----------------------------------------------------------

    @property
    def account(self) -> AccountState:
        return self._account

    def update_account(self, account: AccountState) -> None:
        self._account = account
        if self._start_of_day_equity is None:
            self._start_of_day_equity = account.equity
            log.info("portfolio.start_of_day_equity_set", equity=str(account.equity))

    def start_new_session(self, equity: Decimal) -> None:
        """Explicitly roll the trading day. Called by the scheduler at
        session open — never implicitly on process start."""
        self._start_of_day_equity = equity
        self._session_date = datetime.now(timezone.utc).date()
        log.info("portfolio.new_session", equity=str(equity), date=str(self._session_date))

    @property
    def start_of_day_equity(self) -> Decimal | None:
        return self._start_of_day_equity

    # ---- P&L and exposure --------------------------------------------------

    @property
    def realized_pnl(self) -> Decimal:
        return sum((p.realized_pnl for p in self._positions.values()), Decimal("0"))

    def unrealized_pnl(self, prices: dict[str, Decimal]) -> Decimal:
        total = Decimal("0")
        for key, position in self._positions.items():
            if position.is_flat:
                continue
            if key not in prices:
                raise MissingPriceError(f"No mark price for open position {key}")
            total += position.unrealized_pnl(prices[key])
        return total

    def daily_pnl(self, prices: dict[str, Decimal]) -> Decimal:
        """Change in equity since session start, including open positions.
        Raises if start-of-day equity was never established, because a
        daily-loss limit that silently evaluates against None would be
        worse than no limit at all."""
        if self._start_of_day_equity is None:
            raise MissingPriceError(
                "start_of_day_equity not set — cannot evaluate daily loss limit"
            )
        current = self._account.equity + self.unrealized_pnl(prices)
        return current - self._start_of_day_equity

    def daily_pnl_pct(self, prices: dict[str, Decimal]) -> Decimal:
        if not self._start_of_day_equity:
            raise MissingPriceError("start_of_day_equity not set or zero")
        return self.daily_pnl(prices) / self._start_of_day_equity

    def gross_exposure(self, prices: dict[str, Decimal]) -> Decimal:
        """Sum of absolute position values — longs and shorts both count."""
        total = Decimal("0")
        for key, position in self._positions.items():
            if position.is_flat:
                continue
            if key not in prices:
                raise MissingPriceError(f"No mark price for open position {key}")
            total += position.exposure(prices[key])
        return total

    def net_exposure(self, prices: dict[str, Decimal]) -> Decimal:
        """Signed exposure — longs and shorts offset."""
        total = Decimal("0")
        for key, position in self._positions.items():
            if position.is_flat:
                continue
            if key not in prices:
                raise MissingPriceError(f"No mark price for open position {key}")
            total += position.market_value(prices[key])
        return total

    def gross_exposure_pct(self, prices: dict[str, Decimal]) -> Decimal:
        if not self._account.equity:
            raise MissingPriceError("Account equity is zero/unset — cannot compute exposure %")
        return self.gross_exposure(prices) / self._account.equity
