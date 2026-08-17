"""Phase 10 tests: logging, audit trail, persistence, container, CLI."""

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from app.cli import build_parser, main
from app.config import Settings, TradingMode
from app.dependency_container import build_container
from app.logging_config import REDACTED, redact_secrets
from data.models import Instrument
from database.repository import Repository
from execution.execution_models import Fill, Order, OrderIntent, OrderSide, OrderState
from execution.order_store import OrderStore
from execution.order_validator import OrderValidator
from monitoring.audit import (
    DecisionRecord,
    DecisionRecorder,
    RiskCheckRecord,
    SignalRecord,
    compute_slippage_bps,
)
from portfolio.positions import Position
from risk.decisions import RiskAssessment

AAPL = Instrument(symbol="AAPL")


def make_order(quantity: str = "100") -> Order:
    validator = OrderValidator(OrderStore())
    intent = OrderIntent(
        instrument=AAPL,
        side=OrderSide.BUY,
        quantity=Decimal(quantity),
        stop_loss=Decimal("95"),
        source="momentum",
        strategy="momentum",
    )
    return validator.build_order(
        intent,
        RiskAssessment(
            approved=True,
            approved_quantity=Decimal(quantity),
            requested_quantity=Decimal(quantity),
        ),
    )


# ---- logging redaction ------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    ["api_key", "anthropic_api_key", "password", "secret", "token",
     "database_url", "account_id", "authorization"],
)
def test_secrets_are_redacted(key):
    event = redact_secrets(None, "info", {key: "super-secret-value"})
    assert event[key] == REDACTED
    assert "super-secret-value" not in json.dumps(event)


def test_nested_secret_key_names_redacted():
    event = redact_secrets(None, "info", {"ibkr_account_id": "U1234567"})
    assert event["ibkr_account_id"] == REDACTED


def test_normal_fields_untouched():
    event = redact_secrets(None, "info", {"symbol": "AAPL", "quantity": 100})
    assert event["symbol"] == "AAPL"
    assert event["quantity"] == 100


def test_oversized_values_truncated():
    """A runaway AI response must not flood the log."""
    event = redact_secrets(None, "info", {"response": "x" * 50000})
    assert len(event["response"]) < 5000
    assert "truncated" in event["response"]


def test_logging_configures_without_error():
    from app.logging_config import clear_context, configure_logging

    configure_logging(level="INFO", json_output=True, mode="PAPER")
    clear_context()


# ---- audit trail --------------------------------------------------------------


def test_decision_record_is_immutable():
    record = DecisionRecord(instrument="AAPL:SMART:USD")
    with pytest.raises(Exception):
        record.outcome = "TAMPERED"


def test_recorder_has_no_delete_or_update_method():
    """The spec forbids the AI deleting audit logs or hiding losing
    trades. The guarantee is that no such code exists."""
    forbidden = ("delete", "remove", "update", "modify", "purge", "clear", "drop")
    methods = [m for m in dir(DecisionRecorder) if not m.startswith("_")]
    assert not [m for m in methods if any(f in m.lower() for f in forbidden)]


def test_record_persists_to_disk(tmp_path):
    path = tmp_path / "decisions.jsonl"
    recorder = DecisionRecorder(path=path, emit_to_log=False)
    recorder.record(DecisionRecord(instrument="AAPL:SMART:USD", outcome="SUBMITTED"))
    recorder.record(DecisionRecord(instrument="MSFT:SMART:USD", outcome="REJECTED"))

    assert path.exists()
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2


def test_records_replay_from_disk(tmp_path):
    path = tmp_path / "decisions.jsonl"
    recorder = DecisionRecorder(path=path, emit_to_log=False)
    original = DecisionRecord(
        instrument="AAPL:SMART:USD",
        outcome="FILLED",
        intent_source="ai",
        ai_reasoning="Momentum was strong",
    )
    recorder.record(original)

    replayed = DecisionRecorder(path=path).replay()
    assert len(replayed) == 1
    assert replayed[0].record_id == original.record_id
    assert replayed[0].ai_reasoning == "Momentum was strong"


def test_rejections_are_recorded_not_only_fills(tmp_path):
    """'Why did the agent NOT trade' is the more common question."""
    recorder = DecisionRecorder(path=tmp_path / "d.jsonl", emit_to_log=False)
    recorder.record(
        DecisionRecord(
            instrument="AAPL:SMART:USD",
            intent_id="i1",
            risk_approved=False,
            risk_rejection_reason="MAX_DAILY_LOSS_BREACHED",
            outcome="RISK_REJECTED",
        )
    )
    assert len(recorder.rejections()) == 1


def test_write_failure_is_counted_not_raised(tmp_path):
    """Losing the audit sink must not stop trading, but must be visible."""
    path = tmp_path / "sub" / "d.jsonl"
    recorder = DecisionRecorder(path=path, emit_to_log=False)
    path.parent.rmdir()  # make writes fail
    recorder.record(DecisionRecord(instrument="AAPL"))  # must not raise
    assert recorder.write_failures == 1


def test_explain_produces_full_narrative():
    record = DecisionRecord(
        instrument="AAPL:SMART:USD",
        cycle=42,
        trading_mode="PAPER",
        mid="100.50",
        regime="TRENDING_UP",
        regime_confidence=0.8,
        signals=[
            SignalRecord(
                strategy="momentum",
                direction="LONG",
                strength=0.7,
                rationale="ROC 4% over 20 bars",
            )
        ],
        ai_consulted=True,
        ai_accepted=True,
        ai_action="BUY",
        ai_confidence=0.85,
        ai_reasoning="Trend and momentum aligned",
        intent_id="i1",
        intent_source="ai",
        intent_side="BUY",
        requested_quantity="1000",
        stop_loss="96.00",
        risk_checks=[
            RiskCheckRecord(check_name="kill_switch", approved=True),
            RiskCheckRecord(check_name="position_size", approved=True, detail="4.2%"),
        ],
        risk_approved=True,
        approved_quantity="100",
        was_reduced=True,
        submitted=True,
        broker_order_id="IB-123",
        outcome="FILLED",
    )
    text = record.explain()
    for expected in (
        "TRENDING_UP", "momentum", "ROC 4%", "BUY", "Trend and momentum aligned",
        "kill_switch", "APPROVED 100", "reduced", "IB-123", "FILLED",
    ):
        assert expected in text, expected


def test_slippage_sign_normalised_by_side():
    # Buying above expectation is adverse.
    assert compute_slippage_bps(
        expected_price=Decimal("100"), fill_price=Decimal("100.10"), side="BUY"
    ) == pytest.approx(10.0)
    # Selling below expectation is equally adverse -> positive.
    assert compute_slippage_bps(
        expected_price=Decimal("100"), fill_price=Decimal("99.90"), side="SELL"
    ) == pytest.approx(10.0)
    # Price improvement is negative.
    assert compute_slippage_bps(
        expected_price=Decimal("100"), fill_price=Decimal("99.90"), side="BUY"
    ) == pytest.approx(-10.0)


def test_unserialisable_field_is_stringified_not_dropped():
    class Weird:
        def __repr__(self):
            return "WEIRD"

    from monitoring.audit import _jsonable

    assert _jsonable({"x": Weird()}) == {"x": "WEIRD"}


# ---- persistence ----------------------------------------------------------------


@pytest.fixture
def repo(tmp_path):
    r = Repository(f"sqlite:///{tmp_path}/test.db", mode="PAPER")
    r.create_schema()
    return r


def test_schema_creates(repo):
    assert repo.health_check()


def test_save_and_load_order(repo):
    order = make_order()
    assert repo.save_order(order)
    row = repo.get_order(order.order_id)
    assert row is not None
    assert row.symbol == "AAPL"
    assert Decimal(str(row.quantity)) == Decimal("100")
    assert row.trading_mode == "PAPER"


def test_order_update_reflects_state_change(repo):
    order = make_order()
    repo.save_order(order)
    order.transition_to(OrderState.SUBMITTED)
    repo.save_order(order)
    assert repo.get_order(order.order_id).state == "SUBMITTED"


def test_open_orders_query(repo):
    live = make_order()
    live.transition_to(OrderState.SUBMITTED)
    repo.save_order(live)

    # A broker can only reject an order it received, so it must be
    # submitted first — the state machine enforces this.
    dead = make_order("50")
    dead.transition_to(OrderState.SUBMITTED)
    dead.transition_to(OrderState.REJECTED)
    repo.save_order(dead)

    assert [o.order_id for o in repo.open_orders()] == [live.order_id]


def test_fills_are_idempotent(repo):
    order = make_order()
    repo.save_order(order)
    fill = Fill(
        fill_id="f1",
        order_id=order.order_id,
        timestamp=datetime.now(timezone.utc),
        quantity=Decimal("100"),
        price=Decimal("100.50"),
        commission=Decimal("1"),
    )
    repo.save_fill(fill)
    repo.save_fill(fill)  # same id
    assert len(repo.fills_for_order(order.order_id)) == 1


def test_decimal_precision_preserved(repo):
    """Money must not round-trip through a float."""
    order = make_order()
    repo.save_order(order)
    fill = Fill(
        fill_id="f-precise",
        order_id=order.order_id,
        timestamp=datetime.now(timezone.utc),
        quantity=Decimal("100"),
        price=Decimal("123.45678901"),
        commission=Decimal("0.00000001"),
    )
    repo.save_fill(fill)
    stored = repo.fills_for_order(order.order_id)[0]
    assert Decimal(str(stored.price)) == Decimal("123.45678901")


def test_position_upsert(repo):
    position = Position(
        instrument=AAPL, quantity=Decimal("100"), average_cost=Decimal("50")
    )
    repo.save_position(position)
    position.quantity = Decimal("150")
    repo.save_position(position)
    rows = repo.load_positions()
    assert len(rows) == 1
    assert Decimal(str(rows[0].quantity)) == Decimal("150")


def test_decision_persistence_and_query(repo):
    for reason in ("MAX_DAILY_LOSS_BREACHED", "MAX_DAILY_LOSS_BREACHED", "STALE_MARKET_DATA"):
        repo.save_decision(
            DecisionRecord(
                instrument="AAPL:SMART:USD",
                intent_id="i",
                risk_approved=False,
                risk_rejection_reason=reason,
                outcome="RISK_REJECTED",
            )
        )
    counts = repo.rejection_counts()
    assert counts["MAX_DAILY_LOSS_BREACHED"] == 2
    assert counts["STALE_MARKET_DATA"] == 1


def test_decision_payload_round_trips(repo):
    record = DecisionRecord(
        instrument="AAPL:SMART:USD",
        ai_reasoning="Because the trend was up",
        signals=[SignalRecord(strategy="momentum", direction="LONG", strength=0.6)],
    )
    repo.save_decision(record)
    rows = repo.recent_decisions()
    assert rows[0].payload["ai_reasoning"] == "Because the trend was up"
    assert rows[0].payload["signals"][0]["strategy"] == "momentum"


def test_repository_has_no_audit_delete_methods():
    forbidden = ("delete_decision", "delete_fill", "delete_risk_event",
                 "update_decision", "purge")
    methods = [m for m in dir(Repository) if not m.startswith("_")]
    for f in forbidden:
        assert f not in methods


def test_risk_events_persist(repo):
    repo.save_risk_event(
        event_type="KILL_SWITCH", severity="CRITICAL", detail="Daily loss breached"
    )
    events = repo.risk_events()
    assert len(events) == 1
    assert events[0].event_type == "KILL_SWITCH"


def test_database_failure_does_not_raise(tmp_path):
    """A database outage must not stop trading, but must be counted."""
    repo = Repository("sqlite:////nonexistent/path/db.sqlite", mode="PAPER")
    assert repo.save_order(make_order()) is False
    assert repo.write_failures == 1
    assert repo.is_available is False


def test_equity_curve_reconstruction(repo):
    for equity in (100000, 101000, 99500):
        repo.save_account_snapshot(
            equity=Decimal(str(equity)),
            cash=Decimal(str(equity)),
            buying_power=Decimal(str(equity)),
        )
    curve = repo.equity_curve()
    assert len(curve) == 3
    assert curve[-1][1] == Decimal("99500.00000000")


def test_mode_is_recorded_on_every_row(repo):
    """Paper and live records must never be silently mixed."""
    order = make_order()
    repo.save_order(order)
    repo.save_decision(DecisionRecord(instrument="AAPL"))
    repo.save_risk_event(event_type="TEST", severity="INFO", detail="x")
    assert repo.get_order(order.order_id).trading_mode == "PAPER"
    assert repo.recent_decisions()[0].trading_mode == "PAPER"
    assert repo.risk_events()[0].trading_mode == "PAPER"


# ---- container ---------------------------------------------------------------------


def test_container_builds_from_settings():
    container = build_container(Settings(), symbols=["AAPL"])
    assert container.mode_gate.mode is TradingMode.PAPER
    assert container.instruments[0].symbol == "AAPL"


def test_container_construction_has_no_side_effects():
    """`trading_agent status` must not be able to place an order."""
    container = build_container(Settings(), symbols=["AAPL"])
    assert container.order_store.all_orders() == []
    assert container.portfolio.open_position_count == 0


def test_container_uses_null_ai_when_unconfigured():
    container = build_container(Settings(), symbols=["AAPL"])
    assert container.ai_engine.provider_available is False


def test_container_restricts_ai_symbols_to_universe():
    container = build_container(Settings(), symbols=["AAPL", "MSFT"])
    assert container.ai_engine._allowed == {"AAPL", "MSFT"}


def test_container_defaults_to_simulated_gateway():
    from broker.simulated_broker import SimulatedBrokerGateway

    container = build_container(Settings(), symbols=["AAPL"])
    assert isinstance(container.gateway, SimulatedBrokerGateway)


# ---- CLI ---------------------------------------------------------------------------


def test_parser_has_no_live_command():
    """Running live must require deliberate environment configuration, not
    a convenient subcommand."""
    parser = build_parser()
    actions = [a for a in parser._actions if hasattr(a, "choices") and a.choices]
    commands = set()
    for action in actions:
        commands.update(action.choices)
    assert "live" not in commands
    assert "go-live" not in commands
    assert "promote" not in commands


def test_parser_exposes_expected_commands():
    parser = build_parser()
    commands = set()
    for action in parser._actions:
        if hasattr(action, "choices") and action.choices:
            commands.update(action.choices)
    for expected in (
        "status", "backtest", "paper", "simulate", "strategies",
        "positions", "risk", "kill-switch", "reconcile", "explain", "dashboard",
    ):
        assert expected in commands, expected


def test_kill_switch_command_has_no_deactivation_flag():
    parser = build_parser()
    for action in parser._actions:
        if hasattr(action, "choices") and action.choices and "kill-switch" in action.choices:
            sub = action.choices["kill-switch"]
            flags = {o for a in sub._actions for o in a.option_strings}
            assert not any("deactivate" in f or "reset" in f or "clear" in f for f in flags)


def test_status_command_runs(capsys):
    assert main(["status", "--symbols", "AAPL"]) == 0
    output = capsys.readouterr().out
    assert "MODE: PAPER" in output
    assert "Health:" in output


def test_strategies_command_lists_all(capsys):
    assert main(["strategies"]) == 0
    output = capsys.readouterr().out
    for name in ("ma_crossover", "momentum", "mean_reversion", "trend_following"):
        assert name in output
    assert "none is claimed to be profitable" in output


def test_risk_command_shows_limits(capsys):
    assert main(["risk"]) == 0
    output = capsys.readouterr().out
    assert "max_daily_loss" in output
    assert "Emergency policy" in output


def test_positions_command_runs(capsys):
    assert main(["positions"]) == 0
    assert "No open positions" in capsys.readouterr().out


def test_kill_switch_command_activates(capsys):
    assert main(["kill-switch", "--reason", "test incident"]) == 0
    output = capsys.readouterr().out
    assert "ACTIVATED" in output
    assert "no CLI command to deactivate" in output


def test_reconcile_command_runs(capsys):
    assert main(["reconcile"]) == 0
    assert "Reconciliation clean" in capsys.readouterr().out


def test_every_command_prints_mode_banner(capsys):
    for argv in (["status"], ["strategies"], ["risk"], ["positions"]):
        capsys.readouterr()
        main(argv)
        assert "MODE:" in capsys.readouterr().out, argv


def test_backtest_command_with_csv(tmp_path, capsys):
    csv_file = tmp_path / "bars.csv"
    rows = ["timestamp,open,high,low,close,volume"]
    price = 100.0
    for i in range(120):
        cycle = (i // 30) % 2
        price = max(5.0, price + (0.9 if cycle == 0 else -0.7))
        ts = (datetime(2024, 1, 2, tzinfo=timezone.utc) + timedelta(days=i)).isoformat()
        rows.append(f"{ts},{price},{price*1.01},{price*0.99},{price},100000")
    csv_file.write_text("\n".join(rows))

    code = main(["backtest", "--strategy", "ma_crossover", "--data", str(csv_file),
                 "--symbol", "AAPL"])
    assert code == 0
    output = capsys.readouterr().out
    assert "MODE: BACKTEST" in output
    assert "not evidence of future profitability" in output


def test_backtest_with_missing_file_fails_cleanly(capsys):
    assert main(["backtest", "--strategy", "ma_crossover", "--data", "/nope.csv"]) == 1


def test_explain_command_reads_audit_trail(tmp_path, capsys):
    path = tmp_path / "decisions.jsonl"
    recorder = DecisionRecorder(path=path, emit_to_log=False)
    recorder.record(
        DecisionRecord(
            instrument="AAPL:SMART:USD",
            intent_source="momentum",
            risk_approved=True,
            approved_quantity="100",
            outcome="FILLED",
        )
    )
    assert main(["explain", "--audit-log", str(path)]) == 0
    assert "FILLED" in capsys.readouterr().out
