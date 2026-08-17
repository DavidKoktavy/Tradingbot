"""
Order rate limiting.

Design decision: a sliding window over submission timestamps rather than
a fixed bucket. A fixed per-minute bucket permits 2x the limit across a
boundary (20 at 10:59:59 and 20 at 11:00:00), which is exactly the burst
that a runaway loop produces and that IBKR's pacing rules punish.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone


class OrderRateLimiter:
    def __init__(self, *, max_orders_per_minute: int) -> None:
        self._max = max_orders_per_minute
        self._window = timedelta(seconds=60)
        self._timestamps: deque[datetime] = deque()

    def _prune(self, now: datetime) -> None:
        cutoff = now - self._window
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    def would_exceed(self, *, now: datetime | None = None) -> bool:
        now = now or datetime.now(timezone.utc)
        self._prune(now)
        return len(self._timestamps) >= self._max

    def record(self, *, now: datetime | None = None) -> None:
        now = now or datetime.now(timezone.utc)
        self._prune(now)
        self._timestamps.append(now)

    @property
    def current_count(self) -> int:
        self._prune(datetime.now(timezone.utc))
        return len(self._timestamps)
