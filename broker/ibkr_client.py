"""
IBKRClient: the one place in the system that owns a real ib_async.IB()
instance and exposes it to the rest of the app only through the
BrokerConnection / MarketDataProvider interfaces.

Order management (broker/order_manager.py, broker/execution_listener.py)
is Phase 3 and will extend this client rather than duplicate the
connection handling here.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime

import structlog

from app.config import IBKRSettings
from broker.connection_manager import ConnectionManager
from broker.interfaces import BrokerConnection, MarketDataProvider
from broker.market_data import MarketDataService, MarketDataType
from data.models import Bar, Instrument, MarketSnapshot

log = structlog.get_logger(__name__)


class IBKRClient(BrokerConnection, MarketDataProvider):
    def __init__(
        self,
        settings: IBKRSettings,
        *,
        max_reconnect_attempts: int = 5,
        market_data_type: MarketDataType = MarketDataType.LIVE,
    ) -> None:
        from ib_async import IB  # local import: only this module touches ib_async directly

        self._ib = IB()
        self._connection = ConnectionManager(
            self._ib,
            host=settings.host,
            port=settings.port,
            client_id=settings.client_id,
            max_reconnect_attempts=max_reconnect_attempts,
            on_disconnect=self._on_disconnect,
        )
        self._market_data = MarketDataService(self._ib, market_data_type=market_data_type)
        self._disconnect_hooks: list[object] = []

    def set_market_data_type(self, market_data_type: MarketDataType) -> None:
        """See MarketDataService.set_market_data_type. Exposed here so
        callers never need to reach past this client into the internal
        market data service."""
        self._market_data.set_market_data_type(market_data_type)

    @property
    def market_data_type(self) -> MarketDataType:
        return self._market_data.market_data_type

    # ---- BrokerConnection -------------------------------------------------

    async def connect(self) -> None:
        await self._connection.connect()

    async def disconnect(self) -> None:
        await self._connection.disconnect()

    @property
    def state(self) -> str:
        return self._connection.state

    async def _on_disconnect(self) -> None:
        log.warning("ibkr_client.connection_lost_stopping_new_trades")
        # In later phases this notifies the risk engine / control loop so
        # no new orders are generated while we're not sure of broker state.

    # ---- MarketDataProvider -------------------------------------------------

    async def subscribe(self, instrument: Instrument) -> None:
        await self._market_data.subscribe(instrument)

    async def unsubscribe(self, instrument: Instrument) -> None:
        await self._market_data.unsubscribe(instrument)

    def get_snapshot(self, instrument: Instrument) -> MarketSnapshot | None:
        return self._market_data.get_snapshot(instrument)

    def is_stale(
        self, instrument: Instrument, max_age_seconds: float, *, now: datetime | None = None
    ) -> bool:
        return self._market_data.is_stale(instrument, max_age_seconds, now=now)

    async def get_historical_bars(
        self,
        instrument: Instrument,
        *,
        duration: str,
        bar_size: str,
        end: datetime | None = None,
    ) -> list[Bar]:
        return await self._market_data.get_historical_bars(
            instrument, duration=duration, bar_size=bar_size, end=end
        )

    def snapshot_stream(self, instrument: Instrument) -> AsyncIterator[MarketSnapshot]:
        return self._market_data.snapshot_stream(instrument)
