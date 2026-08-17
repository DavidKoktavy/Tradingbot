"""Fakes for the narrow ib_async surfaces our broker modules depend on.
None of these tests open a socket or require TWS/IB Gateway running."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any


class FakeEvent:
    """Minimal stand-in for ib_async's eventkit.Event (`+=` subscribes)."""

    def __init__(self) -> None:
        self._callbacks: list[Any] = []

    def __iadd__(self, callback: Any) -> "FakeEvent":
        self._callbacks.append(callback)
        return self

    async def emit(self, *args: Any) -> None:
        for cb in self._callbacks:
            result = cb(*args)
            if asyncio.iscoroutine(result):
                await result


@dataclass
class FakeTicker:
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    volume: float | None = None
    updateEvent: FakeEvent = field(default_factory=FakeEvent)


class FakeIB:
    """Fakes both ConnectionManager's IBLike and MarketDataService's
    IBMarketDataLike so a single object can back an end-to-end test."""

    def __init__(self, *, fail_connect_times: int = 0) -> None:
        self._connected = False
        self.disconnectedEvent = FakeEvent()
        self._fail_connect_times = fail_connect_times
        self._connect_calls = 0
        self.historical_bars: list[Any] = []
        self.requested_market_data_types: list[int] = []

    # -- ConnectionManager surface --
    def isConnected(self) -> bool:
        return self._connected

    async def connectAsync(self, host: str, port: int, clientId: int, timeout: float = 4.0) -> None:
        self._connect_calls += 1
        if self._connect_calls <= self._fail_connect_times:
            raise ConnectionRefusedError("simulated TWS not reachable")
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    # -- MarketDataService surface --
    async def qualifyContractsAsync(self, *contracts: Any) -> list[Any]:
        return list(contracts)

    def reqMktData(self, contract: Any, genericTickList: str = "", snapshot: bool = False) -> Any:
        return FakeTicker()

    def reqMarketDataType(self, marketDataType: int) -> None:
        self.requested_market_data_types.append(marketDataType)

    def cancelMktData(self, contract: Any) -> None:
        pass

    async def reqHistoricalDataAsync(
        self,
        contract: Any,
        endDateTime: str,
        durationStr: str,
        barSizeSetting: str,
        whatToShow: str,
        useRTH: bool,
    ) -> list[Any]:
        return self.historical_bars
