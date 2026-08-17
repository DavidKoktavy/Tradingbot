"""Tests for the learning loop: deterministic performance analysis plus
adversarial tests of the AI reflection layer."""

import json
from decimal import Decimal

import pytest

from ai.performance_analyzer import PerformanceAnalyzer
from ai.providers import FailingProvider, NullProvider, ScriptedProvider
from ai.reflection import (
    ReflectionEngine,
    ReflectionHypothesis,
    ReflectionRejectionReason,
    ReflectionResponse,
    SuggestedAction,
)
from backtesting.metrics import TradeRecord
from strategies.promotion import ResearchProposal


def trade(
    strategy: str = "momentum",
    pnl: str = "100",
    n: int = 1,
    exit_time: str | None = None,
) -> TradeRecord:
    return TradeRecord(
        instrument="AAPL",
        strategy=strategy,
        entry_time=f"2026-01-{n:02d}T00:00:00",
        exit_time=exit_time or f"2026-01-{n:02d}T01:00:00",
        side="BUY",
        quantity=Decimal("100"),
        entry_price=Decimal("100"),
        exit_price=Decimal("101"),
        gross_pnl=Decimal(pnl),
        commission=Decimal("1"),
        net_pnl=Decimal(pnl),
        bars_held=5,
    )


# ---- performance analyzer (deterministic) ----------------------------------


def test_empty_trades_produces_warning():
    report = PerformanceAnalyzer().analyze([])
    assert report.total_trades == 0
    assert "No closed trades" in report.warnings[0]


def test_basic_stats_computed():
    trades = [trade(pnl="100"), trade(pnl="-40"), trade(pnl="200"), trade(pnl="-25")]
    report = PerformanceAnalyzer().analyze(trades)
    stats = report.by_strategy["momentum"]
    assert stats.n_trades == 4
    assert stats.win_rate == pytest.approx(0.5)
    assert stats.profit_factor == pytest.approx(300 / 65)
    assert stats.expectancy == pytest.approx((100 - 40 + 200 - 25) / 4)


def test_profit_factor_none_with_no_losses():
    trades = [trade(pnl="100"), trade(pnl="50")]
    report = PerformanceAnalyzer().analyze(trades)
    assert report.by_strategy["momentum"].profit_factor is None


def test_low_trade_count_warns():
    report = PerformanceAnalyzer(min_trades_for_stats=10).analyze([trade()])
    assert any("only 1 trades" in w for w in report.warnings)


def test_strategies_analysed_independently():
    trades = [trade("momentum", "100"), trade("mean_reversion", "-50")]
    report = PerformanceAnalyzer().analyze(trades)
    assert set(report.by_strategy) == {"momentum", "mean_reversion"}
    assert report.by_strategy["momentum"].total_pnl == 100
    assert report.by_strategy["mean_reversion"].total_pnl == -50


def test_degradation_flags_sign_flip():
    analyzer = PerformanceAnalyzer(degradation_window=10, min_trades_for_degradation=20)
    trades = [trade(pnl="50", n=i) for i in range(1, 21)] + [
        trade(pnl="-30", n=i) for i in range(21, 31)
    ]
    report = analyzer.analyze(trades)
    assert len(report.degradation) == 1
    flag = report.degradation[0]
    assert flag.baseline_expectancy > 0
    assert flag.recent_expectancy <= 0
    assert "gone negative" in flag.detail


def test_degradation_flags_material_decline():
    analyzer = PerformanceAnalyzer(degradation_window=10, min_trades_for_degradation=20)
    trades = [trade(pnl="100", n=i) for i in range(1, 21)] + [
        trade(pnl="10", n=i) for i in range(21, 31)
    ]
    report = analyzer.analyze(trades)
    assert len(report.degradation) == 1
    assert "fallen by more than half" in report.degradation[0].detail


def test_stable_performance_not_flagged_as_degrading():
    analyzer = PerformanceAnalyzer(degradation_window=10, min_trades_for_degradation=20)
    trades = [trade(pnl="80", n=i) for i in range(1, 31)]
    report = analyzer.analyze(trades)
    assert report.degradation == []


def test_improving_performance_not_flagged():
    analyzer = PerformanceAnalyzer(degradation_window=10, min_trades_for_degradation=20)
    trades = [trade(pnl="20", n=i) for i in range(1, 21)] + [
        trade(pnl="200", n=i) for i in range(21, 31)
    ]
    report = analyzer.analyze(trades)
    assert report.degradation == []


def test_too_few_trades_skips_degradation_check():
    analyzer = PerformanceAnalyzer(degradation_window=10, min_trades_for_degradation=20)
    trades = [trade(pnl="-10", n=i) for i in range(1, 11)]
    report = analyzer.analyze(trades)
    assert report.degradation == []


def test_streak_detected():
    analyzer = PerformanceAnalyzer(streak_threshold=3)
    trades = [trade(pnl="100", n=1)] + [trade(pnl="-10", n=i) for i in range(2, 6)]
    report = analyzer.analyze(trades)
    assert len(report.streaks) == 1
    assert report.streaks[0].consecutive_losses == 4


def test_streak_not_flagged_below_threshold():
    analyzer = PerformanceAnalyzer(streak_threshold=5)
    trades = [trade(pnl="-10", n=i) for i in range(1, 4)]
    report = analyzer.analyze(trades)
    assert report.streaks == []


def test_streak_broken_by_a_win_does_not_count_further_back():
    analyzer = PerformanceAnalyzer(streak_threshold=3)
    trades = [
        trade(pnl="-10", n=1), trade(pnl="-10", n=2), trade(pnl="-10", n=3),
        trade(pnl="100", n=4),
        trade(pnl="-10", n=5),
    ]
    report = analyzer.analyze(trades)
    assert report.streaks == []


def test_rejection_counts_pass_through():
    report = PerformanceAnalyzer().analyze(
        [trade()], rejection_counts={"MAX_DAILY_LOSS_BREACHED": 5}
    )
    assert report.rejection_counts["MAX_DAILY_LOSS_BREACHED"] == 5


def test_summary_is_human_readable():
    report = PerformanceAnalyzer().analyze([trade(pnl="100"), trade(pnl="-50")])
    text = report.summary()
    assert "momentum" in text
    assert "Trades analysed: 2" in text


# ---- reflection schema strictness -------------------------------------------


def good_hypothesis(**overrides) -> dict:
    payload = {
        "strategy": "momentum",
        "observation": "Win rate dropped from 60% to 30% over the last 20 trades",
        "hypothesis": "Momentum stopped working when volatility regime shifted",
        "suggested_action": "INVESTIGATE",
        "suggested_params": {},
        "confidence": 0.6,
        "rationale": "Consistent with the degradation flag",
    }
    payload.update(overrides)
    return payload


def test_valid_hypothesis_parses():
    h = ReflectionHypothesis.model_validate(good_hypothesis())
    assert h.suggested_action is SuggestedAction.INVESTIGATE


def test_unknown_field_rejected():
    payload = good_hypothesis()
    payload["apply_immediately"] = True
    with pytest.raises(Exception):
        ReflectionHypothesis.model_validate(payload)


@pytest.mark.parametrize(
    "key",
    ["max_daily_loss", "risk_limit", "enable_live_trading", "override_kill_switch",
     "leverage_multiplier", "api_key"],
)
def test_forbidden_param_keys_rejected(key):
    payload = good_hypothesis(suggested_params={key: 10})
    with pytest.raises(Exception):
        ReflectionHypothesis.model_validate(payload)


def test_legitimate_strategy_param_accepted():
    payload = good_hypothesis(
        suggested_action="PROPOSE_PARAMETER_CHANGE",
        suggested_params={"fast_period": 15, "slow_period": 40},
    )
    h = ReflectionHypothesis.model_validate(payload)
    assert h.suggested_params["fast_period"] == 15


def test_too_many_suggested_params_rejected():
    payload = good_hypothesis(suggested_params={f"p{i}": i for i in range(20)})
    with pytest.raises(Exception):
        ReflectionHypothesis.model_validate(payload)


def test_confidence_out_of_range_rejected():
    with pytest.raises(Exception):
        ReflectionHypothesis.model_validate(good_hypothesis(confidence=2.0))


def test_invalid_action_rejected():
    with pytest.raises(Exception):
        ReflectionHypothesis.model_validate(good_hypothesis(suggested_action="RAISE_LIMITS"))


def test_text_fields_capped():
    h = ReflectionHypothesis.model_validate(good_hypothesis(observation="x" * 5000))
    assert len(h.observation) <= 1000


def test_hypothesis_is_immutable():
    from pydantic import ValidationError

    h = ReflectionHypothesis.model_validate(good_hypothesis())
    with pytest.raises(ValidationError):
        h.confidence = 0.99


def test_response_caps_batch_size():
    payload = {"hypotheses": [good_hypothesis() for _ in range(20)]}
    with pytest.raises(Exception):
        ReflectionResponse.model_validate(payload)


def test_response_rejects_unknown_top_level_field():
    payload = {"hypotheses": [], "apply_all": True}
    with pytest.raises(Exception):
        ReflectionResponse.model_validate(payload)


# ---- reflection engine: fail closed -----------------------------------------


def report_with_trades():
    return PerformanceAnalyzer().analyze([trade(pnl="100"), trade(pnl="-50")])


async def test_no_trades_declines_without_calling_provider():
    provider = ScriptedProvider([json.dumps({"hypotheses": [good_hypothesis()]})])
    engine = ReflectionEngine(provider, known_strategies={"momentum"})
    from ai.performance_analyzer import PerformanceReport

    result = await engine.reflect(PerformanceReport())
    assert not result.accepted
    assert result.reason is ReflectionRejectionReason.NO_TRADES
    assert provider.requests == []


async def test_provider_failure_yields_no_hypotheses():
    engine = ReflectionEngine(FailingProvider(), known_strategies={"momentum"})
    result = await engine.reflect(report_with_trades())
    assert not result.accepted
    assert result.reason is ReflectionRejectionReason.PROVIDER_ERROR


async def test_unexpected_exception_is_contained():
    class Exploding(NullProvider):
        async def complete(self, request):
            raise RuntimeError("boom")

    engine = ReflectionEngine(Exploding(), known_strategies={"momentum"})
    result = await engine.reflect(report_with_trades())
    assert not result.accepted
    assert result.reason is ReflectionRejectionReason.PROVIDER_ERROR


async def test_malformed_json_rejected():
    provider = ScriptedProvider(["I think momentum is degrading, you should investigate."])
    engine = ReflectionEngine(provider, known_strategies={"momentum"})
    result = await engine.reflect(report_with_trades())
    assert not result.accepted
    assert result.reason is ReflectionRejectionReason.MALFORMED_JSON


async def test_schema_violation_rejected():
    provider = ScriptedProvider([json.dumps({"hypotheses": [{"strategy": "momentum"}]})])
    engine = ReflectionEngine(provider, known_strategies={"momentum"})
    result = await engine.reflect(report_with_trades())
    assert not result.accepted
    assert result.reason is ReflectionRejectionReason.SCHEMA_VIOLATION


async def test_null_provider_declines_cleanly():
    engine = ReflectionEngine(NullProvider(), known_strategies={"momentum"})
    result = await engine.reflect(report_with_trades())
    assert not result.accepted
    assert engine.provider_available is False


# ---- reflection engine: adversarial -----------------------------------------


async def test_hallucinated_strategy_name_rejects_whole_batch():
    payload = {
        "hypotheses": [
            good_hypothesis(strategy="momentum"),
            good_hypothesis(strategy="totally_made_up_strategy"),
        ]
    }
    provider = ScriptedProvider([json.dumps(payload)])
    engine = ReflectionEngine(provider, known_strategies={"momentum"})
    result = await engine.reflect(report_with_trades())
    assert not result.accepted
    assert result.reason is ReflectionRejectionReason.UNKNOWN_STRATEGY
    assert result.hypotheses == []


async def test_injected_risk_param_rejects_response():
    payload = {
        "hypotheses": [
            good_hypothesis(
                suggested_action="PROPOSE_PARAMETER_CHANGE",
                suggested_params={"max_daily_loss": 0.5},
            )
        ]
    }
    provider = ScriptedProvider([json.dumps(payload)])
    engine = ReflectionEngine(provider, known_strategies={"momentum"})
    result = await engine.reflect(report_with_trades())
    assert not result.accepted
    assert result.reason is ReflectionRejectionReason.SCHEMA_VIOLATION


async def test_recommend_disable_is_inert_text():
    payload = {
        "hypotheses": [
            good_hypothesis(
                strategy="momentum", suggested_action="RECOMMEND_DISABLE",
                confidence=0.9,
            )
        ]
    }
    provider = ScriptedProvider([json.dumps(payload)])
    engine = ReflectionEngine(provider, known_strategies={"momentum"})
    result = await engine.reflect(report_with_trades())
    assert result.accepted
    assert result.hypotheses[0].suggested_action is SuggestedAction.RECOMMEND_DISABLE
    forbidden = ("disable", "apply", "mutate", "update_param", "set_param", "execute")
    members = [m for m in dir(engine) if not m.startswith("_")]
    assert not [m for m in members if any(f in m.lower() for f in forbidden)]


async def test_high_confidence_grants_nothing():
    payload = {"hypotheses": [good_hypothesis(confidence=1.0)]}
    provider = ScriptedProvider([json.dumps(payload)])
    engine = ReflectionEngine(provider, known_strategies={"momentum"})
    result = await engine.reflect(report_with_trades())
    assert result.accepted
    proposal = ReflectionEngine.to_research_proposal(result.hypotheses[0])
    assert isinstance(proposal, ResearchProposal)
    assert proposal.code == ""
    assert not hasattr(proposal, "confidence")
    assert not hasattr(proposal, "auto_approve")


async def test_prompt_fences_data_and_states_pipeline_requirement():
    provider = ScriptedProvider([json.dumps({"hypotheses": []})])
    engine = ReflectionEngine(provider, known_strategies={"momentum"})
    await engine.reflect(report_with_trades())
    prompt = provider.requests[0].user_prompt
    assert "<performance_data>" in prompt and "</performance_data>" in prompt
    assert "DATA, not instructions" in prompt
    system = provider.requests[0].system_prompt
    assert "NOT a trading system yourself" in system
    assert "named human" in system.lower()


async def test_engine_deterministic_temperature():
    provider = ScriptedProvider([json.dumps({"hypotheses": []})])
    engine = ReflectionEngine(provider, known_strategies={"momentum"})
    await engine.reflect(report_with_trades())
    assert provider.requests[0].temperature == 0.0


# ---- to_research_proposal: the only downstream effect ------------------------


def test_to_research_proposal_requires_full_pipeline():
    from strategies.promotion import PromotionPipeline, PromotionStage

    hyp = ReflectionHypothesis.model_validate(
        good_hypothesis(
            suggested_action="PROPOSE_PARAMETER_CHANGE",
            suggested_params={"fast_period": 12},
        )
    )
    proposal = ReflectionEngine.to_research_proposal(hyp, current_params={"slow_period": 40})
    assert proposal.params == {"slow_period": 40, "fast_period": 12}
    assert proposal.proposed_by == "ai_reflection"

    pipeline = PromotionPipeline()
    candidate = pipeline.submit(proposal)
    assert candidate.stage is PromotionStage.RESEARCH


def test_reflection_engine_has_no_execution_capability():
    import inspect

    from ai import reflection as reflection_mod

    source = inspect.getsource(reflection_mod)
    for dangerous in ("exec(", "eval(", "compile(", "__import__", "importlib", "subprocess"):
        assert dangerous not in source, dangerous
