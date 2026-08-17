import pytest

from risk.kill_switch import (
    EmergencyPolicy,
    HaltReason,
    KillSwitch,
    KillSwitchTrigger,
    TradingHalt,
)


def test_kill_switch_starts_inactive():
    ks = KillSwitch()
    assert not ks.is_active


def test_activate_records_event():
    ks = KillSwitch()
    event = ks.activate(KillSwitchTrigger.DAILY_LOSS_LIMIT, "daily loss -2.1%")
    assert ks.is_active
    assert event.trigger is KillSwitchTrigger.DAILY_LOSS_LIMIT
    assert len(ks.history) == 1


def test_reactivation_preserves_original_cause():
    ks = KillSwitch()
    first = ks.activate(KillSwitchTrigger.MAX_DRAWDOWN, "original")
    second = ks.activate(KillSwitchTrigger.SYSTEM_ERROR, "later noise")
    # The first cause is the one worth investigating.
    assert second is first
    assert second.trigger is KillSwitchTrigger.MAX_DRAWDOWN
    assert len(ks.history) == 1


def test_cannot_be_cleared_without_operator_confirmation():
    ks = KillSwitch()
    ks.activate(KillSwitchTrigger.MANUAL)
    with pytest.raises(PermissionError):
        ks.deactivate()
    assert ks.is_active  # still active after the failed attempt


def test_operator_can_clear():
    ks = KillSwitch()
    ks.activate(KillSwitchTrigger.MANUAL)
    ks.deactivate(operator_confirmed=True)
    assert not ks.is_active
    assert ks.history[0].deactivated_at is not None


def test_auto_resettable_switch_can_self_clear():
    ks = KillSwitch(auto_resettable=True)
    ks.activate(KillSwitchTrigger.BROKER_DISCONNECT)
    ks.deactivate()
    assert not ks.is_active


def test_default_emergency_policy_does_not_liquidate():
    # Flattening into whatever caused the emergency can be worse than
    # holding; it must be an explicit operator choice.
    assert KillSwitch().emergency_policy is EmergencyPolicy.CANCEL_ONLY


def test_flatten_policy_is_opt_in():
    ks = KillSwitch(emergency_policy=EmergencyPolicy.FLATTEN_ALL)
    assert ks.emergency_policy is EmergencyPolicy.FLATTEN_ALL


# ---- trading halt -------------------------------------------------------


def test_halt_starts_clear():
    assert not TradingHalt().is_halted


def test_multiple_halt_reasons_must_all_clear():
    halt = TradingHalt()
    halt.set(HaltReason.BROKER_DISCONNECTED, "socket closed")
    halt.set(HaltReason.STALE_MARKET_DATA, "no ticks 30s")
    assert halt.is_halted

    halt.clear(HaltReason.BROKER_DISCONNECTED)
    assert halt.is_halted  # stale data still active

    halt.clear(HaltReason.STALE_MARKET_DATA)
    assert not halt.is_halted


def test_clearing_unknown_reason_is_safe():
    halt = TradingHalt()
    halt.clear(HaltReason.STARTUP)
    assert not halt.is_halted
