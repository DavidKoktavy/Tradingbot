"""
Execution domain models and the order state machine.

Design decisions:

- `OrderIntent` is the *only* thing a strategy or the AI layer may produce.
  It is a request, not an order. It carries no broker IDs and cannot be
  submitted. Converting an intent into an `Order` happens only after the
  risk engine and order validator approve it (Phase 4), which is what
  structurally prevents any component from bypassing risk controls.

- `OrderState` transitions are validated by an explicit adjacency table
  (`_LEGAL_TRANSITIONS`) rather than by ad-hoc `if` checks scattered
  around the codebase. An illegal transition raises rather than being
  silently coerced: if we ever see FILLED -> SUBMITTED, that indicates a
  reconciliation bug and we want it loud, not swallowed.

- States are terminal-aware. Once an order is FILLED / CANCELLED /
  REJECTED / EXPIRED, it can never transition again. This is what makes
  "never assume an order was filled until confirmed" enforceable: the
  only way into FILLED is an explicit broker-confirmed fill event.

- Quantities use Decimal, not float. Fractional share accounting and
  average-price math accumulate float error in ways that eventually
  produce a position size that disagrees with the broker's. Decimal
  keeps local and broker state reconcilable.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from data.models import Instrument


class OrderSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class TimeInForce(StrEnum):
    DAY = "DAY"
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"


class OrderState(StrEnum):
    CREATED = "CREATED"
    VALIDATING = "VALIDATING"
    APPROVED = "APPROVED"
    SUBMITTED = "SUBMITTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    ERROR = "ERROR"


TERMINAL_STATES: frozenset[OrderState] = frozenset(
    {
        OrderState.FILLED,
        OrderState.REJECTED,
        OrderState.CANCELLED,
        OrderState.EXPIRED,
        OrderState.ERROR,
    }
)

_LEGAL_TRANSITIONS: dict[OrderState, frozenset[OrderState]] = {
    OrderState.CREATED: frozenset({OrderState.VALIDATING, OrderState.REJECTED, OrderState.ERROR}),
    OrderState.VALIDATING: frozenset(
        {OrderState.APPROVED, OrderState.REJECTED, OrderState.ERROR}
    ),
    OrderState.APPROVED: frozenset(
        {OrderState.SUBMITTED, OrderState.CANCELLED, OrderState.ERROR}
    ),
    OrderState.SUBMITTED: frozenset(
        {
            OrderState.ACKNOWLEDGED,
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.REJECTED,
            OrderState.CANCEL_REQUESTED,
            OrderState.CANCELLED,
            OrderState.EXPIRED,
            OrderState.ERROR,
        }
    ),
    OrderState.ACKNOWLEDGED: frozenset(
        {
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCEL_REQUESTED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
            OrderState.ERROR,
        }
    ),
    OrderState.PARTIALLY_FILLED: frozenset(
        {
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCEL_REQUESTED,
            OrderState.CANCELLED,
            OrderState.EXPIRED,
            OrderState.ERROR,
        }
    ),
    OrderState.CANCEL_REQUESTED: frozenset(
        {
            OrderState.CANCELLED,
            # A cancel can lose the race with a fill — the broker is the
            # authority, so we must accept these.
            OrderState.FILLED,
            OrderState.PARTIALLY_FILLED,
            OrderState.ERROR,
        }
    ),
    OrderState.FILLED: frozenset(),
    OrderState.REJECTED: frozenset(),
    OrderState.CANCELLED: frozenset(),
    OrderState.EXPIRED: frozenset(),
    OrderState.ERROR: frozenset(),
}


class IllegalStateTransition(Exception):
    """Raised when an order is asked to make a transition that the state
    machine forbids. Never caught-and-ignored: it means local state and
    reality have diverged."""


def can_transition(src: OrderState, dst: OrderState) -> bool:
    return dst in _LEGAL_TRANSITIONS[src]


class OrderIntent(BaseModel):
    """A *proposal* to trade. Produced by strategies and by the AI layer.
    Deliberately has no broker id and no submit() method."""

    model_config = {"frozen": True}

    intent_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    instrument: Instrument
    side: OrderSide
    quantity: Decimal = Field(gt=0)
    order_type: OrderType = OrderType.MARKET
    limit_price: Decimal | None = None
    # `stop_price` is the TRIGGER price for STOP/STOP_LIMIT order types.
    stop_price: Decimal | None = None
    time_in_force: TimeInForce = TimeInForce.DAY

    # Protective levels, independent of order type. A MARKET entry with an
    # attached protective stop is a normal bracket, so these must NOT be
    # conflated with `stop_price`. `stop_loss` is what the position sizer
    # measures risk against; without it (and without a volatility estimate)
    # the sizer refuses to size the trade at all.
    stop_loss: Decimal | None = None
    take_profit: Decimal | None = None

    # Provenance — required for the audit trail ("why did the agent trade?").
    source: str = "unknown"  # strategy name, or "ai"
    strategy: str | None = None
    signal_id: str | None = None

    @model_validator(mode="after")
    def _prices_match_order_type(self) -> "OrderIntent":
        if self.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and self.limit_price is None:
            raise ValueError(f"{self.order_type} requires limit_price")
        if self.order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and self.stop_price is None:
            raise ValueError(f"{self.order_type} requires stop_price")
        if self.order_type is OrderType.MARKET and (
            self.limit_price is not None or self.stop_price is not None
        ):
            raise ValueError("MARKET order must not carry limit_price/stop_price")
        for name in ("limit_price", "stop_price", "stop_loss", "take_profit"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")

        # Protective levels must sit on the correct side of the trade.
        # A "stop loss" above the entry on a long is not a stop loss; it
        # is an instruction to sell into a profit at a loss-shaped price,
        # and almost always indicates a sign error upstream.
        if self.stop_loss is not None and self.take_profit is not None:
            if self.side is OrderSide.BUY and self.stop_loss >= self.take_profit:
                raise ValueError("BUY: stop_loss must be below take_profit")
            if self.side is OrderSide.SELL and self.stop_loss <= self.take_profit:
                raise ValueError("SELL: stop_loss must be above take_profit")
        return self

    def dedupe_key(self) -> str:
        """Identity used for duplicate-order detection. Two intents with
        the same instrument/side/qty/type from the same source within the
        dedupe window are treated as the same intended trade."""
        return (
            f"{self.instrument}|{self.side}|{self.quantity}|"
            f"{self.order_type}|{self.source}|{self.strategy}"
        )


class Fill(BaseModel):
    """A broker-confirmed execution. Only the broker creates these."""

    model_config = {"frozen": True}

    fill_id: str
    order_id: str
    timestamp: datetime
    quantity: Decimal = Field(gt=0)
    price: Decimal = Field(gt=0)
    commission: Decimal = Decimal("0")


class Order(BaseModel):
    """A risk-approved order with a lifecycle. Created only by the
    execution layer from an approved OrderIntent."""

    order_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    intent: OrderIntent
    state: OrderState = OrderState.CREATED
    broker_order_id: str | None = None

    filled_quantity: Decimal = Decimal("0")
    average_fill_price: Decimal | None = None
    fills: list[Fill] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    state_history: list[tuple[OrderState, datetime]] = Field(default_factory=list)
    error_message: str | None = None

    @property
    def remaining_quantity(self) -> Decimal:
        return self.intent.quantity - self.filled_quantity

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def is_active(self) -> bool:
        """Active = live at the broker (or believed to be). Used by
        reconciliation and duplicate detection."""
        return self.state in {
            OrderState.SUBMITTED,
            OrderState.ACKNOWLEDGED,
            OrderState.PARTIALLY_FILLED,
            OrderState.CANCEL_REQUESTED,
        }

    def transition_to(self, new_state: OrderState, *, error_message: str | None = None) -> None:
        if not can_transition(self.state, new_state):
            raise IllegalStateTransition(
                f"Order {self.order_id}: illegal transition {self.state} -> {new_state}"
            )
        self.state_history.append((self.state, self.updated_at))
        self.state = new_state
        self.updated_at = datetime.now(timezone.utc)
        if error_message is not None:
            self.error_message = error_message

    def apply_fill(self, fill: Fill) -> None:
        """Apply a broker-confirmed fill. This is the ONLY path to
        PARTIALLY_FILLED/FILLED — we never infer a fill from a timeout,
        a submission ack, or an absence of a rejection."""
        if self.is_terminal:
            raise IllegalStateTransition(
                f"Order {self.order_id}: cannot apply fill to terminal state {self.state}"
            )
        new_filled = self.filled_quantity + fill.quantity
        if new_filled > self.intent.quantity:
            raise ValueError(
                f"Order {self.order_id}: overfill — {new_filled} > {self.intent.quantity}"
            )

        # Volume-weighted average price across all fills.
        prior_notional = (self.average_fill_price or Decimal("0")) * self.filled_quantity
        self.average_fill_price = (prior_notional + fill.price * fill.quantity) / new_filled
        self.filled_quantity = new_filled
        self.fills.append(fill)

        self.transition_to(
            OrderState.FILLED if new_filled == self.intent.quantity else OrderState.PARTIALLY_FILLED
        )
