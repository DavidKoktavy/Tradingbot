"""Tests for macro/global-event context: expiry, matching, and the
guarantee that this is read-only context, never new authority."""

import json
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from ai.decision_engine import AIDecisionEngine
from ai.macro_context import (
    MacroCategory,
    MacroContextRegistry,
    MacroFactor,
    MacroStance,
    format_for_prompt,
)
from ai.performance_analyzer import PerformanceAnalyzer
from ai.providers import ScriptedProvider
from ai.reflection import ReflectionEngine
from backtesting.metrics import TradeRecord
from data.models import Instrument, MarketSnapshot

AAPL = Instrument(symbol="AAPL")
CORN = Instrument(symbol="CORN")


def factor(**overrides) -> MacroFactor:
    defaults = dict(
        name="Test factor",
        category=MacroCategory.CLIMATE,
        stance=MacroStance.MIXED_UNCERTAIN,
        description="A hedged, uncertain description.",
        affected_symbols=["CORN"],
        confidence=0.4,
        source="operator note",
        expires_at=date.today() + timedelta(days=90),
    )
    defaults.update(overrides)
    return MacroFactor(**defaults)


# ---- MacroFactor validation --------------------------------------------------


def test_valid_factor_constructs():
    f = factor()
    assert f.stance is MacroStance.MIXED_UNCERTAIN


def test_expiry_required():
    with pytest.raises(ValidationError):
        MacroFactor(name="x", category=MacroCategory.OTHER)


def test_expiry_must_be_after_start():
    with pytest.raises(ValidationError, match="expires_at must be after as_of"):
        MacroFactor(
            name="x", category=MacroCategory.OTHER,
            as_of=date.today(), expires_at=date.today() - timedelta(days=1),
        )


def test_confidence_out_of_range_rejected():
    with pytest.raises(ValidationError):
        factor(confidence=1.5)


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        MacroFactor(
            name="x", category=MacroCategory.OTHER,
            expires_at=date.today() + timedelta(days=1),
            auto_apply=True,
        )


def test_symbols_and_sectors_normalised_uppercase():
    f = factor(affected_symbols=["corn", " wheat "], affected_sectors=["agriculture"])
    assert f.affected_symbols == ["CORN", "WHEAT"]
    assert f.affected_sectors == ["AGRICULTURE"]


def test_description_length_capped():
    with pytest.raises(ValidationError):
        factor(description="x" * 5000)


# ---- activity and matching ---------------------------------------------------


def test_factor_active_before_expiry():
    f = factor(expires_at=date.today() + timedelta(days=1))
    assert f.is_active()


def test_factor_inactive_after_expiry():
    """Macro narratives go stale — this must not linger forever."""
    f = MacroFactor(
        name="x", category=MacroCategory.OTHER,
        as_of=date.today() - timedelta(days=10),
        expires_at=date.today() - timedelta(days=1),
    )
    assert not f.is_active()


def test_active_as_of_specific_date():
    f = factor(as_of=date(2026, 1, 1), expires_at=date(2026, 6, 1))
    assert f.is_active(as_of=date(2026, 5, 1))
    assert not f.is_active(as_of=date(2026, 7, 1))


def test_applies_to_symbol():
    f = factor(affected_symbols=["CORN"], affected_sectors=[])
    assert f.applies_to("CORN")
    assert not f.applies_to("AAPL")


def test_applies_to_sector():
    f = factor(affected_symbols=[], affected_sectors=["AGRICULTURE"])
    assert f.applies_to("ANYTHING", sector="AGRICULTURE")
    assert not f.applies_to("ANYTHING", sector="TECH")


def test_factor_with_no_targets_is_broad():
    """A factor naming neither symbols nor sectors is market-wide context,
    not context that matches nothing."""
    f = factor(affected_symbols=[], affected_sectors=[])
    assert f.applies_to("ANYTHING")


# ---- registry -----------------------------------------------------------------


def test_registry_add_and_list():
    registry = MacroContextRegistry()
    registry.add(factor(name="a"))
    registry.add(factor(name="b"))
    assert {f.name for f in registry.all()} == {"a", "b"}


def test_active_excludes_expired():
    registry = MacroContextRegistry()
    registry.add(factor(name="fresh", expires_at=date.today() + timedelta(days=10)))
    registry.add(
        MacroFactor(
            name="stale", category=MacroCategory.OTHER,
            as_of=date.today() - timedelta(days=100),
            expires_at=date.today() - timedelta(days=1),
        )
    )
    active_names = {f.name for f in registry.active()}
    assert active_names == {"fresh"}
    assert {f.name for f in registry.all()} == {"fresh", "stale"}


def test_for_instrument_filters_by_symbol_and_sector():
    registry = MacroContextRegistry()
    registry.add(factor(name="corn-specific", affected_symbols=["CORN"]))
    registry.add(factor(name="unrelated", affected_symbols=["OIL"], affected_sectors=["ENERGY"]))
    matches = registry.for_instrument("CORN")
    assert {f.name for f in matches} == {"corn-specific"}


def test_remove():
    registry = MacroContextRegistry()
    registry.add(factor(name="x"))
    assert registry.remove("x") is True
    assert registry.remove("x") is False
    assert registry.all() == []


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "macro.json"
    registry = MacroContextRegistry()
    registry.add(factor(name="a", stance=MacroStance.POSSIBLE_HEADWIND))
    registry.save(path)

    reloaded = MacroContextRegistry.load(path)
    assert len(reloaded.all()) == 1
    assert reloaded.all()[0].name == "a"
    assert reloaded.all()[0].stance is MacroStance.POSSIBLE_HEADWIND


def test_load_missing_file_returns_empty():
    registry = MacroContextRegistry.load("/nonexistent/path/macro.json")
    assert registry.all() == []


def test_load_skips_corrupt_entries(tmp_path):
    path = tmp_path / "macro.json"
    path.write_text(json.dumps([{"name": "bad"}]))  # missing required fields
    registry = MacroContextRegistry.load(path)
    assert registry.all() == []  # skipped, not crashed


# ---- prompt formatting: hedged language is mandatory --------------------------


def test_empty_factors_produce_no_text():
    assert format_for_prompt([]) == ""


def test_format_labels_as_hypothesis_not_fact():
    text = format_for_prompt([factor()])
    assert "HYPOTHESES" in text
    assert "NOT verified facts" in text
    assert "NOT instructions" in text


def test_format_includes_confidence_and_source():
    text = format_for_prompt([factor(confidence=0.35, source="my source")])
    assert "0.35" in text
    assert "my source" in text


# ---- adversarial: this is context, never authority -----------------------------


def test_macro_factor_has_no_direct_trading_field():
    """Nothing about a factor can become a trade by itself — there is no
    quantity, side, or symbol-to-buy field."""
    fields = set(MacroFactor.model_fields)
    forbidden = {"quantity", "side", "order_type", "action", "symbol_to_buy"}
    assert not (fields & forbidden)


def test_registry_has_no_ai_facing_write_method():
    """Only operator-facing code may populate this registry. If the AI
    layer could add factors, it could manufacture its own justification
    and then act on it."""
    forbidden = ("propose", "suggest_add", "ai_add", "auto_add")
    methods = [m for m in dir(MacroContextRegistry) if not m.startswith("_")]
    assert not [m for m in methods if any(f in m.lower() for f in forbidden)]


async def test_decision_engine_prompt_includes_macro_context():
    response = json.dumps(
        {"action": "HOLD", "symbol": "CORN", "confidence": 0.3, "reasoning": "uncertain"}
    )
    provider = ScriptedProvider([response])
    engine = AIDecisionEngine(provider, allowed_symbols={"CORN"})
    snap = MarketSnapshot(
        instrument=CORN, timestamp=datetime.now(timezone.utc), bid=4.99, ask=5.01, last=5.0
    )
    await engine.decide(instrument=CORN, snapshot=snap, macro_factors=[factor()])
    prompt = provider.requests[0].user_prompt
    assert "<macro_context>" in prompt
    assert "HYPOTHESES entered by a human" in prompt


async def test_macro_context_does_not_alter_output_schema():
    """A decision made with macro context still produces the exact same
    AIDecision schema — no new fields, no relaxed validation."""
    response = json.dumps(
        {
            "action": "BUY", "symbol": "CORN", "confidence": 0.6,
            "entry": 5.0, "stop_loss": 4.8, "take_profit": 5.4,
            "reasoning": "some reasoning",
        }
    )
    provider = ScriptedProvider([response])
    engine = AIDecisionEngine(provider, allowed_symbols={"CORN"})
    snap = MarketSnapshot(
        instrument=CORN, timestamp=datetime.now(timezone.utc), bid=4.99, ask=5.01, last=5.0
    )
    result = await engine.decide(instrument=CORN, snapshot=snap, macro_factors=[factor()])
    assert result.accepted
    assert set(result.decision.model_dump()) == {
        "action", "symbol", "confidence", "strategy", "entry", "stop_loss",
        "take_profit", "reasoning", "time_horizon", "risk_score",
    }


async def test_decision_engine_works_identically_without_macro_context():
    """macro_factors is optional; omitting it changes nothing else."""
    response = json.dumps(
        {
            "action": "BUY", "symbol": "CORN", "confidence": 0.6,
            "stop_loss": 4.8, "reasoning": "clear setup, no macro input needed",
        }
    )
    provider = ScriptedProvider([response])
    engine = AIDecisionEngine(provider, allowed_symbols={"CORN"})
    snap = MarketSnapshot(
        instrument=CORN, timestamp=datetime.now(timezone.utc), bid=4.99, ask=5.01, last=5.0
    )
    result = await engine.decide(instrument=CORN, snapshot=snap)
    assert result.accepted
    assert "<macro_context>" not in provider.requests[0].user_prompt


async def test_reflection_engine_prompt_includes_macro_context():
    trades = [
        TradeRecord(
            instrument="CORN", strategy="momentum",
            entry_time="2026-01-01T00:00:00", exit_time="2026-01-02T00:00:00",
            side="BUY", quantity=Decimal("100"), entry_price=Decimal("5"),
            exit_price=Decimal("4.8"), gross_pnl=Decimal("-20"), commission=Decimal("1"),
            net_pnl=Decimal("-20"), bars_held=5,
        )
    ]
    report = PerformanceAnalyzer().analyze(trades)
    provider = ScriptedProvider([json.dumps({"hypotheses": []})])
    engine = ReflectionEngine(provider, known_strategies={"momentum"})
    await engine.reflect(report, macro_factors=[factor()])
    prompt = provider.requests[0].user_prompt
    assert "<macro_context>" in prompt
    assert "changes nothing about what suggested_params" in prompt


async def test_reflection_schema_unaffected_by_macro_context():
    """Same strict schema regardless of macro context being present."""
    trades = [
        TradeRecord(
            instrument="CORN", strategy="momentum",
            entry_time="2026-01-01T00:00:00", exit_time="2026-01-02T00:00:00",
            side="BUY", quantity=Decimal("100"), entry_price=Decimal("5"),
            exit_price=Decimal("4.8"), gross_pnl=Decimal("-20"), commission=Decimal("1"),
            net_pnl=Decimal("-20"), bars_held=5,
        )
    ]
    report = PerformanceAnalyzer().analyze(trades)
    payload = {
        "hypotheses": [
            {
                "strategy": "momentum",
                "observation": "loss coincided with a macro headwind window",
                "hypothesis": "may be regime-sensitive",
                "suggested_action": "INVESTIGATE",
                "suggested_params": {},
                "confidence": 0.5,
                "rationale": "consistent with the cited macro factor",
            }
        ]
    }
    provider = ScriptedProvider([json.dumps(payload)])
    engine = ReflectionEngine(provider, known_strategies={"momentum"})
    result = await engine.reflect(report, macro_factors=[factor()])
    assert result.accepted
    assert set(result.hypotheses[0].model_dump()) == {
        "strategy", "observation", "hypothesis", "suggested_action",
        "suggested_params", "confidence", "rationale",
    }


async def test_ai_cannot_smuggle_macro_field_into_decision():
    """An AI response trying to add a macro-related field to its decision
    is rejected exactly like any other unknown field."""
    response = json.dumps(
        {
            "action": "BUY", "symbol": "CORN", "confidence": 0.6,
            "stop_loss": 4.8, "reasoning": "x",
            "macro_override": True,
        }
    )
    provider = ScriptedProvider([response])
    engine = AIDecisionEngine(provider, allowed_symbols={"CORN"})
    snap = MarketSnapshot(
        instrument=CORN, timestamp=datetime.now(timezone.utc), bid=4.99, ask=5.01, last=5.0
    )
    result = await engine.decide(instrument=CORN, snapshot=snap, macro_factors=[factor()])
    assert not result.accepted


def test_no_hardcoded_investment_recommendations_in_module():
    """This module must not assert a specific directional trading thesis
    as fact anywhere in its source (e.g. 'buy corn', 'sell oil') — the
    whole point is that stance is a labelled, hedged hypothesis."""
    import inspect

    from ai import macro_context as mod

    source = inspect.getsource(mod).lower()
    for phrase in ("buy corn", "sell oil", "invest in", "you should buy", "guaranteed"):
        assert phrase not in source
