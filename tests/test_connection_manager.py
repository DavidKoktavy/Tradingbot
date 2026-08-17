import pytest

from broker.connection_manager import ConnectionManager, IBConnectionError
from broker.interfaces import ConnectionState
from tests.fakes import FakeIB


async def test_connect_succeeds_first_try():
    ib = FakeIB(fail_connect_times=0)
    cm = ConnectionManager(ib, host="127.0.0.1", port=7497, client_id=1)
    await cm.connect()
    assert cm.state == ConnectionState.CONNECTED
    assert cm.is_connected


async def test_connect_retries_then_succeeds():
    ib = FakeIB(fail_connect_times=2)  # fails twice, succeeds on 3rd attempt
    cm = ConnectionManager(ib, host="127.0.0.1", port=7497, client_id=1, max_reconnect_attempts=5)
    await cm.connect()
    assert cm.is_connected
    assert ib._connect_calls == 3


async def test_connect_exhausts_retry_budget_and_raises():
    ib = FakeIB(fail_connect_times=99)  # never succeeds
    cm = ConnectionManager(ib, host="127.0.0.1", port=7497, client_id=1, max_reconnect_attempts=3)
    with pytest.raises(IBConnectionError):
        await cm.connect()
    assert cm.state == ConnectionState.DISCONNECTED


async def test_disconnect_sets_state():
    ib = FakeIB()
    cm = ConnectionManager(ib, host="127.0.0.1", port=7497, client_id=1)
    await cm.connect()
    await cm.disconnect()
    assert cm.state == ConnectionState.DISCONNECTED
    assert not ib.isConnected()


async def test_unexpected_disconnect_triggers_on_disconnect_hook_and_reconnects():
    ib = FakeIB()
    calls: list[str] = []

    async def on_disconnect() -> None:
        calls.append("stopped_new_trades")

    cm = ConnectionManager(
        ib, host="127.0.0.1", port=7497, client_id=1, on_disconnect=on_disconnect
    )
    await cm.connect()
    assert cm.is_connected

    # Simulate TWS dropping the connection unexpectedly.
    ib._connected = False
    await ib.disconnectedEvent.emit()

    assert calls == ["stopped_new_trades"]
    assert cm.is_connected  # reconnect succeeded (FakeIB always reconnects here)


async def test_explicit_disconnect_does_not_trigger_reconnect_hook():
    ib = FakeIB()
    calls: list[str] = []

    async def on_disconnect() -> None:
        calls.append("called")

    cm = ConnectionManager(
        ib, host="127.0.0.1", port=7497, client_id=1, on_disconnect=on_disconnect
    )
    await cm.connect()
    await cm.disconnect()

    # Even if the underlying disconnectedEvent fires after our own explicit
    # disconnect (as it would in ib_async), we must not treat that as an
    # unexpected drop and try to reconnect.
    await ib.disconnectedEvent.emit()
    assert calls == []
    assert cm.state == ConnectionState.DISCONNECTED
