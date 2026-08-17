"""
Abstract interfaces around the broker.

Design decision: nothing outside `broker/` should import ib_async directly,
and nothing outside `broker/` should import `broker.ibkr_client` directly
either — everyone else depends on these interfaces. That makes it possible
to:
  - Unit-test strategies, risk engine, portfolio manager, etc. against a
    fake/in-memory broker with zero network or TWS/Gateway dependency.
  - Later add a second broker without touching consumers.

Phase 2 defines the connection + market-data surface. Order management
(interfaces + implementation) is Phase 3.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from datetime import datetime

from data.models import Bar, Instrument, MarketSnapshot


class ConnectionState:
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"


class BrokerConnection(ABC):
    """Lifecycle + health surface for a broker connection."""

    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def disconnect(self) -> None: ...

    @property
    @abstractmethod
    def state(self) -> str: ...

    @property
    def is_connected(self) -> bool:
        return self.state == ConnectionState.CONNECTED


class MarketDataProvider(ABC):
    """Read-only market data surface, decoupled from the broker SDK."""

    @abstractmethod
    async def subscribe(self, instrument: Instrument) -> None:
        """Begin streaming updates for an instrument."""

    @abstractmethod
    async def unsubscribe(self, instrument: Instrument) -> None: ...

    @abstractmethod
    def get_snapshot(self, instrument: Instrument) -> MarketSnapshot | None:
        """Return the latest known snapshot, or None if never received."""

    @abstractmethod
    async def get_historical_bars(
        self,
        instrument: Instrument,
        *,
        duration: str,
        bar_size: str,
        end: datetime | None = None,
    ) -> list[Bar]:
        """Fetch historical OHLCV bars. `duration`/`bar_size` follow IBKR's
        own string formats (e.g. "30 D", "1 hour") since translating those
        into a broker-neutral vocabulary buys little and adds ambiguity."""

    @abstractmethod
    def snapshot_stream(self, instrument: Instrument) -> AsyncIterator[MarketSnapshot]:
        """Async-iterate live snapshots as they arrive, for event-driven
        consumers (control loop, strategies) rather than polling."""
