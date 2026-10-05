"""
Real exchange sessions, from a real exchange calendar.

Design decisions:

- **The fixed UTC window it replaces was wrong in three ways, all of them
  silent.** 13:30-20:00 UTC is 09:30-16:00 Eastern only during US standard
  time; for roughly eight months of the year the real session is an hour
  earlier in UTC, so the fixed window permitted an hour of trading after
  the close and blocked the first hour of it. It also knew nothing about
  the ~10 market holidays a year, or the half-days around Thanksgiving,
  Christmas Eve and July 4th, when the close is 13:00 ET. The visible
  symptom of all three is the same and easy to misread: orders rejected or
  accepted at the wrong times, attributed to "the broker being odd".

- **This fails closed.** An unknown exchange, a missing
  `exchange_calendars` install, or a naive datetime all result in
  `is_open()` returning False, never True. A trading-hours check that
  cannot determine the time should refuse to trade, not guess; a guess
  here means orders placed into a closed or illiquid market.

- **Calendars are cached.** `exchange_calendars` builds a multi-decade
  schedule on construction, which is far too slow to do per risk check but
  trivial once per process.

- `ExchangeCalendar` is a thin wrapper rather than the library type
  directly, so the risk engine depends on our two-method surface and
  stays testable with a fake.
"""

from __future__ import annotations

from datetime import date, datetime
from functools import lru_cache
from typing import Any

import structlog

log = structlog.get_logger(__name__)


# IBKR exchange codes do not match ISO MIC codes used by exchange_calendars.
# "SMART" is IBKR's order router rather than a venue; for US equities its
# session is the NYSE session, which is the correct conservative mapping.
_CALENDAR_ALIASES: dict[str, str] = {
    "SMART": "XNYS",
    "NYSE": "XNYS",
    "ARCA": "XNYS",
    "AMEX": "XNYS",
    "BATS": "XNYS",
    "IEX": "XNYS",
    "NASDAQ": "XNAS",
    "ISLAND": "XNAS",
    "LSE": "XLON",
    "LSEETF": "XLON",
    "TSE": "XTSE",
    "SEHK": "XHKG",
    "EUREX": "XEUR",
    "IBIS": "XETR",
    "SBF": "XPAR",
    "AEB": "XAMS",
}


def resolve_calendar_name(exchange: str) -> str:
    """Map an IBKR exchange code to an exchange_calendars name."""
    key = (exchange or "").strip().upper()
    return _CALENDAR_ALIASES.get(key, key)


@lru_cache(maxsize=16)
def _load_calendar(name: str) -> Any | None:
    """Construct and cache a calendar, or None if unavailable.

    Returns None rather than raising so a missing optional dependency
    degrades to the fixed-window fallback with a loud log line, instead of
    taking the process down at import time.
    """
    try:
        import exchange_calendars as xcals
    except ImportError:
        log.error(
            "calendar.library_missing",
            name=name,
            hint="pip install 'exchange_calendars>=4.5'",
        )
        return None

    try:
        return xcals.get_calendar(name)
    except Exception as exc:  # noqa: BLE001 - library raises several types
        log.error("calendar.unknown_exchange", name=name, error=str(exc))
        return None


class ExchangeCalendar:
    """Session membership for one exchange.

    Every method fails closed: if the calendar could not be loaded, or the
    timestamp is not timezone-aware, the answer is "not open".
    """

    def __init__(self, exchange: str = "SMART") -> None:
        self._requested = exchange
        self._name = resolve_calendar_name(exchange)
        self._calendar = _load_calendar(self._name)

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_available(self) -> bool:
        """False when the real calendar could not be loaded. Callers use
        this to decide whether to warn about the degraded fallback."""
        return self._calendar is not None

    def is_open(self, moment: datetime) -> bool:
        """True only if `moment` falls inside a real trading session.

        Half-days are handled for free: the calendar's close for that
        session is the actual 13:00 ET close, not a nominal 16:00.
        """
        if self._calendar is None:
            return False
        if moment.tzinfo is None:
            # A naive timestamp cannot be placed on a session boundary, and
            # assuming a zone is exactly the bug this module exists to fix.
            log.error("calendar.naive_timestamp", name=self._name)
            return False
        try:
            import pandas as pd

            return bool(self._calendar.is_open_on_minute(pd.Timestamp(moment)))
        except Exception as exc:  # noqa: BLE001
            log.error("calendar.is_open_failed", name=self._name, error=str(exc))
            return False

    def is_trading_day(self, day: date) -> bool:
        """True if `day` is a session (not a weekend or holiday)."""
        if self._calendar is None:
            return False
        try:
            import pandas as pd

            return bool(self._calendar.is_session(pd.Timestamp(day)))
        except Exception as exc:  # noqa: BLE001
            log.error("calendar.is_session_failed", name=self._name, error=str(exc))
            return False

    def describe(self, moment: datetime) -> str:
        """Human-readable reason, for rejection messages and audit records.

        The distinction between "holiday" and "outside hours" matters when
        reading an audit trail: one means today was never going to trade,
        the other means the bot was running at the wrong time of day.
        """
        if self._calendar is None:
            return f"calendar {self._name} unavailable"
        if not self.is_trading_day(moment.date()):
            return f"{moment.date()} is not a trading session on {self._name}"
        return f"{moment.isoformat()} is outside {self._name} session hours"
