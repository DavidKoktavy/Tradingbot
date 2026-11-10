"""
Applies universe selections to a running control loop.

Design decisions:

- **Order of operations is the safety property.** Adding: subscribe and
  warm up the feed, wait for a first snapshot, *then* hand the instrument
  to the loop. Removing: take it out of the loop first, *then* unsubscribe.
  The reverse order lets a cycle reach an instrument with no data, which
  sets a global STALE_MARKET_DATA halt and blocks every other symbol.

- **Protected symbols are never removed**: anything with an open position
  or a working order. The loop's own `remove_instrument` refuses these as
  well; the check is repeated here so the feed is never unsubscribed from
  something the loop still trades.

- **Two cadences.** A full `refresh()` (scanner + history + scoring) runs
  rarely, by default once a day, to stay well inside IBKR's request
  pacing. A cheap `sync_held()` runs every minute and only makes sure
  held symbols are tracked — e.g. a position adopted by reconciliation
  after a restart.

- **Never raises out of `run()`.** A failed refresh keeps the current
  universe; the loop carries on trading what it already tracks.
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

import structlog

from data.models import Instrument
from universe.selector import UniverseSelection, UniverseSelector

log = structlog.get_logger(__name__)


class DynamicFeed(Protocol):
    def snapshot(self, instrument: Instrument) -> Any: ...
    async def add_instrument(self, instrument: Instrument) -> None: ...
    async def remove_instrument(self, instrument: Instrument) -> None: ...


class UniverseRefresher:
    def __init__(
        self,
        selector: UniverseSelector,
        loop: Any,
        feed: DynamicFeed,
        *,
        portfolio: Any,
        order_store: Any,
        ai_engine: Any | None = None,
        refresh_seconds: float = 86_400.0,
        sync_seconds: float = 60.0,
        first_snapshot_timeout: float = 15.0,
    ) -> None:
        self._selector = selector
        self._loop = loop
        self._feed = feed
        self._portfolio = portfolio
        self._store = order_store
        self._ai = ai_engine
        self._refresh_seconds = refresh_seconds
        self._sync_seconds = sync_seconds
        self._snapshot_timeout = first_snapshot_timeout
        self.last_selection: UniverseSelection | None = None

    # ---- public -------------------------------------------------------------

    def protected_symbols(self) -> list[str]:
        held = [p.instrument.symbol for p in self._portfolio.open_positions()]
        working = [o.intent.instrument.symbol for o in self._store.active_orders()]
        out: list[str] = []
        for s in [*held, *working]:
            if s not in out:
                out.append(s)
        return out

    async def refresh(self) -> UniverseSelection:
        selection = await self._selector.select(held=self.protected_symbols())
        self.last_selection = selection
        await self._apply(selection.symbols)
        return selection

    async def sync_held(self) -> None:
        current = {i.symbol for i in self._loop.instruments}
        missing = [s for s in self.protected_symbols() if s not in current]
        for symbol in missing:
            await self._add(symbol, protected=True)
        if missing:
            self._sync_ai()

    async def run(self, *, initial_refresh: bool = False) -> None:
        if initial_refresh:
            await self._safe(self.refresh)
        elapsed = 0.0
        while True:
            await asyncio.sleep(self._sync_seconds)
            elapsed += self._sync_seconds
            await self._safe(self.sync_held)
            if elapsed >= self._refresh_seconds:
                elapsed = 0.0
                await self._safe(self.refresh)

    # ---- internals ----------------------------------------------------------

    async def _safe(self, fn: Any) -> None:
        try:
            await fn()
        except Exception as exc:  # noqa: BLE001 — keep the current universe
            log.exception("universe.refresh_failed", step=fn.__name__, error=str(exc))

    async def _apply(self, target: list[str]) -> None:
        protected = set(self.protected_symbols())
        target_set = set(target) | protected
        current = {i.symbol: i for i in self._loop.instruments}

        for symbol, instrument in current.items():
            if symbol in target_set or symbol in protected:
                continue
            if self._loop.remove_instrument(instrument):
                await self._feed.remove_instrument(instrument)

        for symbol in [*target, *sorted(protected)]:
            if symbol not in current:
                await self._add(symbol, protected=symbol in protected)

        self._sync_ai()
        log.info(
            "universe.applied",
            tracking=[i.symbol for i in self._loop.instruments],
            protected=sorted(protected),
        )

    async def _add(self, symbol: str, *, protected: bool) -> None:
        instrument = Instrument(symbol=symbol)
        if any(i.symbol == symbol for i in self._loop.instruments):
            return
        try:
            await self._feed.add_instrument(instrument)
        except Exception as exc:  # noqa: BLE001
            log.error("universe.add_failed", symbol=symbol, error=str(exc), protected=protected)
            if protected:
                # Must be tracked regardless: a stale-data halt is the
                # correct reaction to a held position we cannot price.
                self._loop.add_instrument(instrument)
            return

        if not await self._wait_for_snapshot(instrument) and not protected:
            log.warning("universe.no_snapshot", symbol=symbol, timeout=self._snapshot_timeout)
            await self._feed.remove_instrument(instrument)
            return
        self._loop.add_instrument(instrument)

    async def _wait_for_snapshot(self, instrument: Instrument) -> bool:
        waited = 0.0
        step = 0.25
        while waited <= self._snapshot_timeout:
            if self._feed.snapshot(instrument) is not None:
                return True
            await asyncio.sleep(step)
            waited += step
        return False

    def _sync_ai(self) -> None:
        if self._ai is not None and hasattr(self._ai, "set_allowed_symbols"):
            self._ai.set_allowed_symbols({i.symbol for i in self._loop.instruments})
