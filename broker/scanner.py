"""
IBKR market scanner adapter.

Design decisions:

- **Lives in broker/** because it touches ib_async, and nothing outside
  broker/ may (see broker/interfaces.py). It returns plain symbol strings;
  no ib_async type leaves this module.

- **Proposes, never decides.** Output goes to `universe.UniverseSelector`,
  which intersects it with the approved list. This module has no notion of
  what is allowed to trade, by design: a scanner that could widen the
  tradeable universe would be a bypass of an operator-reviewed boundary.

- **Several scan codes, merged in order and de-duplicated.** Defaults
  suit a long-only dip strategy: today's biggest percentage losers, then
  most active. IBKR caps a scan at 50 rows.

- **Requires market-scanner data permissions on the account.** Without
  them the request errors; the selector catches that and falls back to
  the approved list, so a missing permission degrades selection rather
  than stopping the bot.
"""

from __future__ import annotations

from typing import Any, Protocol

import structlog

log = structlog.get_logger(__name__)

DEFAULT_SCAN_CODES = ("TOP_PERC_LOSE", "MOST_ACTIVE")
MAX_ROWS = 50


class IBScannerLike(Protocol):
    async def reqScannerDataAsync(self, subscription: Any) -> list[Any]: ...


class IBKRScanner:
    def __init__(
        self,
        ib: IBScannerLike,
        *,
        scan_codes: tuple[str, ...] | list[str] = DEFAULT_SCAN_CODES,
        location_code: str = "STK.US.MAJOR",
        rows: int = MAX_ROWS,
        above_price: float = 5.0,
        above_volume: int = 1_000_000,
    ) -> None:
        if not scan_codes:
            raise ValueError("At least one scan code is required")
        self._ib = ib
        self._scan_codes = tuple(scan_codes)
        self._location = location_code
        self._rows = max(1, min(rows, MAX_ROWS))
        self._above_price = above_price
        self._above_volume = above_volume

    def _subscription(self, scan_code: str) -> Any:
        from ib_async import ScannerSubscription  # local: only broker/ touches ib_async

        return ScannerSubscription(
            instrument="STK",
            locationCode=self._location,
            scanCode=scan_code,
            numberOfRows=self._rows,
            abovePrice=self._above_price,
            aboveVolume=self._above_volume,
        )

    async def scan(self) -> list[str]:
        symbols: list[str] = []
        failures = 0
        for code in self._scan_codes:
            try:
                rows = await self._ib.reqScannerDataAsync(self._subscription(code))
            except Exception as exc:  # noqa: BLE001
                failures += 1
                log.warning("scanner.scan_failed", scan_code=code, error=str(exc))
                continue
            for row in rows or []:
                try:
                    symbol = str(row.contractDetails.contract.symbol).upper()
                except AttributeError:
                    continue
                if symbol and symbol not in symbols:
                    symbols.append(symbol)
            log.info("scanner.scanned", scan_code=code, rows=len(rows or []))
        if failures == len(self._scan_codes):
            raise RuntimeError(
                "Every scanner request failed — check market scanner permissions in TWS"
            )
        return symbols
