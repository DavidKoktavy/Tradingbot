"""
Market data service.

Design decisions:
- ib_async's `Ticker`/`BarData` objects are translated into our normalized
  `MarketSnapshot`/`Bar` models immediately on receipt (see data/models.py
  for why). No ib_async type ever leaves this module.
- Only STOCK is implemented in Phase 2 — FOREX/FUTURE/OPTION/CRYPTO raise
  NotImplementedError rather than silently mis-mapping to a Stock contract.
  This is a case of "don't leave critical functionality as pseudocode",
  applied in reverse: an honest NotImplementedError is safer than a
  plausible-looking contract that resolves to the wrong instrument.
- Each subscribed instrument keeps (a) the latest snapshot for synchronous
  `get_snapshot()` reads used by the risk engine's freshness check, and
  (b) an asyncio.Queue for `snapshot_stream()` so the event-driven control
  loop doesn't have to poll.
- Historical bar requests go through ib_async's pacing-aware
  `reqHistoricalDataAsync`, which respects IBKR's API pacing limits
  internally; we don't add our own request loop that could violate them.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Protocol

import structlog

from broker.interfaces import MarketDataProvider
from data.models import AssetClass, Bar, Instrument, MarketSnapshot

log = structlog.get_logger(__name__)


class MarketDataType(IntEnum):
    """Matches IBKR's own reqMarketDataType() codes exactly.

    Design note, found via a real user hitting Error 10168 on a fresh
    paper account: the TWS API defaults to LIVE and does NOT automatically
    fall back to delayed data the way the TWS GUI does for a human
    manually looking up a quote. A client with no real-time market data
    subscription gets nothing at all from the API unless it explicitly
    requests DELAYED. This is a session-wide setting on the IB connection,
    not a per-symbol one.
    """

    LIVE = 1
    FROZEN = 2
    DELAYED = 3
    DELAYED_FROZEN = 4


class IBMarketDataLike(Protocol):
    """Narrow surface of ib_async.IB this module depends on — kept small
    and separate from ConnectionManager's IBLike so each module can be
    tested/faked independently."""

    async def qualifyContractsAsync(self, *contracts: Any) -> list[Any]: ...

    def reqMktData(self, contract: Any, genericTickList: str = "", snapshot: bool = False) -> Any: ...

    def cancelMktData(self, contract: Any) -> None: ...

    def reqMarketDataType(self, marketDataType: int) -> None: ...

    async def reqHistoricalDataAsync(
        self,
        contract: Any,
        endDateTime: str,
        durationStr: str,
        barSizeSetting: str,
        whatToShow: str,
        useRTH: bool,
    ) -> list[Any]: ...


def _to_ib_contract(instrument: Instrument) -> Any:
    if instrument.asset_class is not AssetClass.STOCK:
        raise NotImplementedError(
            f"Instrument asset class {instrument.asset_class} is not yet "
            "supported by MarketDataService (Phase 2 covers STOCK only)."
        )
    from ib_async import Stock  # local import: keep ib_async out of module-level

    return Stock(instrument.symbol, instrument.exchange, instrument.currency)


def _ticker_to_snapshot(instrument: Instrument, ticker: Any) -> MarketSnapshot:
    def _clean(x: float | None) -> float | None:
        # ib_async uses NaN / -1 sentinels for "no value yet".
        if x is None:
            return None
        try:
            if x != x or x < 0:  # NaN check via x != x
                return None
        except TypeError:
            return None
        return float(x)

    return MarketSnapshot(
        instrument=instrument,
        timestamp=datetime.now(timezone.utc),
        bid=_clean(getattr(ticker, "bid", None)),
        ask=_clean(getattr(ticker, "ask", None)),
        last=_clean(getattr(ticker, "last", None)),
        volume=_clean(getattr(ticker, "volume", None)),
    )


def _bar_data_to_bar(bar: Any) -> Bar:
    ts = bar.date
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return Bar(
        timestamp=ts,
        open=float(bar.open),
        high=float(bar.high),
        low=float(bar.low),
        close=float(bar.close),
        volume=float(getattr(bar, "volume", 0.0) or 0.0),
    )


class MarketDataService(MarketDataProvider):
    def __init__(
        self, ib: IBMarketDataLike, *, market_data_type: MarketDataType = MarketDataType.LIVE
    ) -> None:
        self._ib = ib
        self._contracts: dict[str, Any] = {}
        self._tickers: dict[str, Any] = {}
        self._latest: dict[str, MarketSnapshot] = {}
        self._queues: dict[str, asyncio.Queue[MarketSnapshot]] = {}
        # Session-wide on the IB connection, not per-symbol. Defaults to
        # LIVE, matching the API's own default — an operator or the
        # smoke-test script must opt into DELAYED explicitly, the same
        # way they'd have to opt into it being aware their data is stale.
        self._market_data_type = market_data_type

    def set_market_data_type(self, market_data_type: MarketDataType) -> None:
        """Change the requested type for future subscriptions. Existing
        subscriptions are not retroactively changed — call subscribe()
        again (or unsubscribe/resubscribe) to apply it to a symbol
        already being watched."""
        self._market_data_type = market_data_type
        log.info("marketdata.type_changed", market_data_type=market_data_type.name)

    @property
    def market_data_type(self) -> MarketDataType:
        return self._market_data_type

    @staticmethod
    def _key(instrument: Instrument) -> str:
        return str(instrument)

    async def subscribe(self, instrument: Instrument) -> None:
        key = self._key(instrument)
        if key in self._tickers:
            return  # already subscribed

        contract = _to_ib_contract(instrument)
        qualified = await self._ib.qualifyContractsAsync(contract)
        if not qualified:
            raise ValueError(f"IBKR could not qualify contract for {instrument}")
        contract = qualified[0]

        # Must be set before reqMktData, and on every subscribe (not
        # cached as "already sent this session") because it's cheap,
        # idempotent, and set_market_data_type() may have changed it
        # since the last subscription.
        self._ib.reqMarketDataType(int(self._market_data_type))
        ticker = self._ib.reqMktData(contract, "", False)
        self._contracts[key] = contract
        self._tickers[key] = ticker
        self._queues[key] = asyncio.Queue(maxsize=1000)

        update_event = getattr(ticker, "updateEvent", None)
        if update_event is not None:

            def _on_update(t: Any, *_a: object) -> None:
                snapshot = _ticker_to_snapshot(instrument, t)
                self._latest[key] = snapshot
                queue = self._queues[key]
                if queue.full():
                    # Drop the oldest rather than block the event callback;
                    # consumers care about the latest state, not every tick.
                    queue.get_nowait()
                queue.put_nowait(snapshot)

            update_event += _on_update  # type: ignore[operator]

        log.info("marketdata.subscribed", instrument=key)

    async def unsubscribe(self, instrument: Instrument) -> None:
        key = self._key(instrument)
        contract = self._contracts.pop(key, None)
        self._tickers.pop(key, None)
        self._queues.pop(key, None)
        if contract is not None:
            self._ib.cancelMktData(contract)
        log.info("marketdata.unsubscribed", instrument=key)

    def get_snapshot(self, instrument: Instrument) -> MarketSnapshot | None:
        return self._latest.get(self._key(instrument))

    def is_stale(
        self, instrument: Instrument, max_age_seconds: float, *, now: datetime | None = None
    ) -> bool:
        """True if we have no snapshot at all, OR the snapshot we have is
        older than max_age_seconds. No-data-yet counts as stale — the
        caller must not treat 'never received' as 'safe to trade'."""
        snapshot = self.get_snapshot(instrument)
        if snapshot is None:
            return True
        return snapshot.is_stale(max_age_seconds, now=now)

    async def get_historical_bars(
        self,
        instrument: Instrument,
        *,
        duration: str,
        bar_size: str,
        end: datetime | None = None,
    ) -> list[Bar]:
        contract = _to_ib_contract(instrument)
        qualified = await self._ib.qualifyContractsAsync(contract)
        if not qualified:
            raise ValueError(f"IBKR could not qualify contract for {instrument}")
        end_str = end.strftime("%Y%m%d %H:%M:%S") if end else ""
        raw_bars = await self._ib.reqHistoricalDataAsync(
            qualified[0],
            endDateTime=end_str,
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=True,
        )
        return [_bar_data_to_bar(b) for b in raw_bars]

    async def snapshot_stream(self, instrument: Instrument) -> AsyncIterator[MarketSnapshot]:
        key = self._key(instrument)
        if key not in self._queues:
            await self.subscribe(instrument)
        queue = self._queues[key]
        while True:
            yield await queue.get()
