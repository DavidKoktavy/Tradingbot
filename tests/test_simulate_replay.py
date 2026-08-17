"""Tests for `simulate --data`: replaying historical bars through the live
control loop, autonomously, with real fill draining."""

from datetime import datetime, timedelta, timezone

import pytest

from app.cli import main


def trending_series(n: int = 400) -> list[float]:
    out, price = [], 100.0
    for i in range(n):
        cycle = (i // 45) % 2
        price = max(5.0, price + (0.9 if cycle == 0 else -0.7) + 0.4 * (1 if i % 3 == 0 else -1))
        out.append(price)
    return out


def write_csv(path, closes: list[float]) -> None:
    rows = ["timestamp,open,high,low,close,volume"]
    base = datetime(2023, 1, 3, tzinfo=timezone.utc)
    d = base
    for c in closes:
        rows.append(f"{d.isoformat()},{c},{c*1.01},{c*0.99},{c},200000")
        d += timedelta(days=1)
    path.write_text("\n".join(rows))


def test_simulate_without_data_explains_nothing_will_trade(capsys):
    """Honest messaging: no data source means no evaluation, and the tool
    must say so rather than silently producing zero orders with no
    explanation."""
    code = main(["simulate", "--symbols", "AAPL", "--max-cycles", "1"])
    assert code == 0
    output = capsys.readouterr().out
    assert "No --data file given" in output
    assert "nothing will be evaluated" in output


def test_simulate_with_data_replays_and_can_produce_orders(tmp_path, capsys):
    """End-to-end: a real crossover in the data must flow all the way
    through strategy -> risk -> order -> simulated fill -> portfolio."""
    csv_file = tmp_path / "bars.csv"
    write_csv(csv_file, trending_series())

    code = main([
        "simulate", "--symbols", "AAPL", "--strategies", "ma_crossover",
        "--data", str(csv_file),
    ])
    assert code == 0
    output = capsys.readouterr().out
    assert "Replaying" in output
    assert "Cycles:" in output
    assert "Final position:" in output
    assert "Realised P&L:" in output


def test_simulate_replay_reports_fills_matching_orders(tmp_path, capsys):
    csv_file = tmp_path / "bars.csv"
    write_csv(csv_file, trending_series())

    main([
        "simulate", "--symbols", "AAPL", "--strategies", "ma_crossover",
        "--data", str(csv_file),
    ])
    output = capsys.readouterr().out
    line = [l for l in output.splitlines() if "Orders submitted" in l][0]
    assert "Fills:" in line


def test_simulate_replay_rejects_too_few_bars(tmp_path, capsys):
    csv_file = tmp_path / "short.csv"
    write_csv(csv_file, [100.0] * 10)  # far fewer than ma_crossover's warm-up

    code = main([
        "simulate", "--symbols", "AAPL", "--strategies", "ma_crossover",
        "--data", str(csv_file),
    ])
    assert code == 1
    assert "nothing will trade" in capsys.readouterr().err


def test_simulate_replay_rejects_missing_file(capsys):
    code = main([
        "simulate", "--symbols", "AAPL", "--data", "/nonexistent/file.csv",
    ])
    assert code == 1
    assert "No bars loaded" in capsys.readouterr().err


def test_simulate_replay_never_connects_to_ibkr(tmp_path):
    """Structural guarantee: --data replay must remain fully offline,
    exactly like the no-data path."""
    from unittest.mock import patch

    csv_file = tmp_path / "bars.csv"
    write_csv(csv_file, trending_series())

    with patch("broker.ibkr_client.IBKRClient") as mock_client:
        main([
            "simulate", "--symbols", "AAPL", "--strategies", "ma_crossover",
            "--data", str(csv_file),
        ])
        mock_client.assert_not_called()


# ---- regression: the staleness bug found via manual end-to-end testing -----


def test_replayed_bars_are_not_rejected_as_stale(tmp_path, capsys):
    """Regression: ControlLoop checks data freshness against the REAL
    wall clock (unlike BacktestEngine, which accepts simulation time).
    Stamping replayed snapshots with each bar's own historical timestamp
    (e.g. 2023) made every single one look catastrophically stale and
    get silently rejected — the replay ran to completion, reported zero
    orders, and gave no indication anything was wrong. This is the exact
    same bug class as the Phase 6 BacktestEngine simulation-time issue,
    recurring in a new piece of code, and was only found by actually
    running the command rather than by a unit test."""
    csv_file = tmp_path / "bars.csv"
    write_csv(csv_file, trending_series())  # deliberately old, historical dates

    code = main([
        "simulate", "--symbols", "AAPL", "--strategies", "ma_crossover",
        "--data", str(csv_file),
    ])
    assert code == 0
    output = capsys.readouterr().out
    line = [l for l in output.splitlines() if "Orders submitted" in l][0]
    submitted = int(line.split("Orders submitted:")[1].split()[0])
    assert submitted > 0, "orders were rejected — the staleness bug has regressed"
