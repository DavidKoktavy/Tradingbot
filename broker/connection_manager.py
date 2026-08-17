"""
Connection lifecycle management for TWS / IB Gateway.

Design decisions:
- `ConnectionManager` depends on a narrow `IBLike` Protocol, not on
  `ib_async.IB` directly. In production we inject a real `ib_async.IB()`;
  in tests we inject a fake that doesn't touch the network. This is the
  "interfaces around the broker so the system is testable without
  connecting to IBKR" requirement from the spec (section 27).
- Reconnection uses exponential backoff (tenacity) and is triggered both
  by an explicit `connect()` failure and by the underlying client's
  disconnect event, since IBKR sessions can drop at any time (network
  blip, TWS restart, daily API reset around midnight ET, etc).
- We track connection state explicitly (`ConnectionState`) rather than
  inferring it, so other components (health checks, the control loop) can
  synchronously check `is_connected` without an RPC.
- We deliberately do NOT catch-and-continue on repeated reconnect failure
  forever: after `max_reconnect_attempts` we surface the failure so a
  supervising component can halt trading and alert an operator, per the
  fail-closed principle. We do not auto-retry indefinitely in silence.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol

import structlog
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from broker.interfaces import ConnectionState

log = structlog.get_logger(__name__)


class IBConnectionError(Exception):
    """Raised when the broker connection cannot be established or is lost
    and cannot be recovered within the configured retry budget."""


class IBLike(Protocol):
    """The subset of ib_async.IB's surface ConnectionManager depends on.
    Keeping this narrow is what makes ConnectionManager unit-testable
    without a real TWS/Gateway instance."""

    def isConnected(self) -> bool: ...

    async def connectAsync(
        self, host: str, port: int, clientId: int, timeout: float = 4.0
    ) -> None: ...

    def disconnect(self) -> None: ...

    @property
    def disconnectedEvent(self) -> object: ...


class ConnectionManager:
    def __init__(
        self,
        ib: IBLike,
        *,
        host: str,
        port: int,
        client_id: int,
        max_reconnect_attempts: int = 5,
        on_disconnect: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._ib = ib
        self._host = host
        self._port = port
        self._client_id = client_id
        self._max_reconnect_attempts = max_reconnect_attempts
        self._on_disconnect = on_disconnect
        self._state = ConnectionState.DISCONNECTED

        # ib_async events support += for subscribing a callback.
        disconnected_event = getattr(self._ib, "disconnectedEvent", None)
        if disconnected_event is not None:
            disconnected_event += self._handle_unexpected_disconnect  # type: ignore[operator]

    @property
    def state(self) -> str:
        return self._state

    @property
    def is_connected(self) -> bool:
        return self._state == ConnectionState.CONNECTED

    async def connect(self) -> None:
        """Connect with exponential backoff. Raises IBConnectionError if
        the retry budget is exhausted — callers must treat that as
        'cannot trade', not retry forever themselves."""
        if self._ib.isConnected():
            self._state = ConnectionState.CONNECTED
            return

        self._state = ConnectionState.CONNECTING
        log.info(
            "ibkr.connecting", host=self._host, port=self._port, client_id=self._client_id
        )
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self._max_reconnect_attempts),
                wait=wait_exponential(multiplier=1, min=1, max=30),
                retry=retry_if_exception_type(Exception),
                reraise=True,
            ):
                with attempt:
                    await self._ib.connectAsync(
                        self._host, self._port, clientId=self._client_id, timeout=4.0
                    )
        except Exception as exc:  # noqa: BLE001 - deliberately broad, translated below
            self._state = ConnectionState.DISCONNECTED
            log.error("ibkr.connect_failed", error=str(exc), attempts=self._max_reconnect_attempts)
            raise IBConnectionError(
                f"Failed to connect to IBKR at {self._host}:{self._port} "
                f"after {self._max_reconnect_attempts} attempts"
            ) from exc

        self._state = ConnectionState.CONNECTED
        log.info("ibkr.connected", host=self._host, port=self._port)

    async def disconnect(self) -> None:
        log.info("ibkr.disconnecting")
        self._ib.disconnect()
        self._state = ConnectionState.DISCONNECTED

    async def _handle_unexpected_disconnect(self, *_args: object) -> None:
        """Callback wired to ib_async's disconnectedEvent. Fires on any
        unexpected drop (not on our own explicit disconnect() calls, since
        by then state is already DISCONNECTED and we don't re-enter)."""
        if self._state == ConnectionState.DISCONNECTED:
            return  # We initiated this; nothing to recover.

        log.warning("ibkr.unexpected_disconnect")
        self._state = ConnectionState.RECONNECTING
        if self._on_disconnect is not None:
            # Let the caller (e.g. the control loop / health monitor) stop
            # new trading immediately, before we even attempt to reconnect.
            await self._on_disconnect()
        try:
            await self.connect()
        except IBConnectionError:
            # State is already DISCONNECTED at this point (set in connect()).
            # We do not loop forever here — the caller's health check /
            # supervisor is responsible for deciding what happens next.
            log.error("ibkr.reconnect_exhausted")
