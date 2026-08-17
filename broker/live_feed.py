"""
Live market-data feed for the control loop.

Design decisions:

- **The loop depends on a tiny read interface** (`snapshot`, `history`),
  not on IBKR. `LiveMarketDataFeed` adapts the broker to it. That is why
  every loop test runs against an in-memory feed with no network.

- **Historical bars are fetched once at startup and appended from live
  data**, rather than re-requested every cycle. IBKR enforces strict
  pacing on historical data requests (roughly 60 per 10 minutes, with
  further per-contract limits); a naive per-cycle refetch trips pacing
  violations, which IBKR responds to by throttling or disconnecting the
  session. Getting this wrong looks like random data outages.

- **A bar is only appended when its period has closed.** Appending a
  partially-formed current bar would feed strategies a close price that
  changes underneath them, producing signals that flicker and backtests
  that can never reproduce live behaviour.

- **Staleness is the loop's decision, not the feed's.** The feed reports
  what it has and when it arrived; the risk engine owns the age threshold.
  A feed that silently withheld stale data would hide an outage.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import structlog

from broker.interfaces import MarketDataProvider
from data.models import Bar, Instrument, MarketSnapshot

log = structlog.get_logger(__name__)

# Bar-size string -> period length, for deciding when a bar has closed.
_BAR_PERIODS = {
    "1 min": timedelta(minutes=1),
    "5 mins": timedelta(minutes=5),
    "15 mins": timedelta(minutes=15),
    "30 mins": timedelta(minutes=30),
    "1 hour": timedelta(hours=1),
    "1 day": timedelta(days=1),
}


class LiveMarketDataFeed:
    """Adapts a `MarketDataProvider` to the control loop's feed interface."""

    def __init__(
        self,
        provider: MarketDataProvider,
        instruments: list[Instrument],
        *,
        bar_size: str = "1 min",
        history_duration: str = "2 D",
        max_bars: int = 2000,
    ) -> None:
        self._provider = provider
        self._instruments = instruments
        self._bar_size = bar_size
        self._history_duration = history_duration
        self._max_bars = max_bars
        self._bars: dict[str, list[Bar]] = {}
        self._building: dict[str, dict] = {}
        self._warmed_up = False

    @property
    def is_warmed_up(self) -> bool:
        return self._warmed_up

    async def start(self) -> None:
        """Subscribe to live data and fetch history once.

        Raises if history cannot be fetched for any instrument: starting
        the loop with no history means strategies emit nothing while
        appearing healthy, which is worse than failing loudly.
        """
        for instrument in self._instruments:
            await self._provider.subscribe(instrument)
            bars = await self._provider.get_historical_bars(
                instrument,
                duration=self._history_duration,
                bar_size=self._bar_size,
            )
            if not bars:
                raise RuntimeError(
                    f"No historical bars returned for {instrument}. Refusing to start "
                    "with no history: strategies would silently emit nothing."
                )
            self._bars[str(instrument)] = bars[-self._max_bars :]
            log.info(
                "feed.warmed_up",
                instrument=str(instrument),
                bars=len(bars),
                bar_size=self._bar_size,
            )
        self._warmed_up = True

    async def stop(self) -> None:
        for instrument in self._instruments:
            try:
                await self._provider.unsubscribe(instrument)
            except Exception as exc:  # noqa: BLE001
                log.error("feed.unsubscribe_failed", instrument=str(instrument), error=str(exc))

    # ---- control-loop interface -------------------------------------------

    def snapshot(self, instrument: Instrument) -> MarketSnapshot | None:
        return self._provider.get_snapshot(instrument)

    def history(self, instrument: Instrument) -> list[Bar]:
        return self._bars.get(str(instrument), [])

    # ---- bar aggregation ---------------------------------------------------

    def ingest(self, snapshot: MarketSnapshot) -> Bar | None:
        """Fold a tick into the current bar, returning a completed bar when
        its period closes. Only closed bars reach `history()`."""
        key = str(snapshot.instrument)
        price = snapshot.mid
        if price is None:
            return None

        period = _BAR_PERIODS.get(self._bar_size, timedelta(minutes=1))
        bucket = self._floor(snapshot.timestamp, period)
        current = self._building.get(key)

        if current is None:
            self._building[key] = self._new_bar(bucket, price, snapshot.volume)
            return None

        if bucket > current["bucket"]:
            completed = Bar(
                timestamp=current["bucket"],
                open=current["open"],
                high=current["high"],
                low=current["low"],
                close=current["close"],
                volume=current["volume"],
            )
            self._append(key, completed)
            self._building[key] = self._new_bar(bucket, price, snapshot.volume)
            return completed

        current["high"] = max(current["high"], price)
        current["low"] = min(current["low"], price)
        current["close"] = price
        if snapshot.volume:
            current["volume"] = max(current["volume"], snapshot.volume)
        return None

    def _append(self, key: str, bar: Bar) -> None:
        bars = self._bars.setdefault(key, [])
        bars.append(bar)
        if len(bars) > self._max_bars:
            del bars[: len(bars) - self._max_bars]

    @staticmethod
    def _new_bar(bucket: datetime, price: float, volume: float | None) -> dict:
        return {
            "bucket": bucket,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": volume or 0.0,
        }

    @staticmethod
    def _floor(ts: datetime, period: timedelta) -> datetime:
        if period >= timedelta(days=1):
            return ts.replace(hour=0, minute=0, second=0, microsecond=0)
        seconds = int(period.total_seconds())
        epoch = int(ts.timestamp())
        return datetime.fromtimestamp(epoch - (epoch % seconds), tz=timezone.utc)

    async def run_ingest(self, poll_seconds: float = 1.0) -> None:
        """Background task: poll snapshots and fold them into bars.

        Polling rather than consuming the provider's async stream keeps
        this decoupled from the provider's event model and bounded in
        rate; the snapshot itself is event-driven underneath.
        """
        while True:
            for instrument in self._instruments:
                snapshot = self._provider.get_snapshot(instrument)
                if snapshot is not None:
                    try:
                        self.ingest(snapshot)
                    except Exception as exc:  # noqa: BLE001
                        log.error("feed.ingest_failed", error=str(exc))
            await asyncio.sleep(poll_seconds)
