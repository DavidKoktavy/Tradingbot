"""
Strategy research and promotion pipeline (spec section 22).

The required progression:

    RESEARCH -> BACKTEST -> VALIDATION -> PAPER -> HUMAN APPROVAL -> LIVE

Design decisions:

- **Nothing auto-promotes, and LIVE requires a human.** `promote()` refuses
  to enter LIVE without an explicit `HumanApproval` carrying an approver
  identity and a recorded rationale. The AI can propose, backtest, and
  report; it cannot approve. There is no code path that promotes on the
  basis of good numbers alone.

- **Each gate has objective criteria checked in code, and a candidate must
  pass every gate in order.** Skipping a stage raises. Good results at one
  stage never excuse a missing earlier stage — that is exactly the
  shortcut that puts an unvalidated strategy in front of real money.

- **The overfitting check is a gate, not advice.** A candidate whose
  out-of-sample performance collapses relative to in-sample is rejected at
  VALIDATION regardless of how good the headline numbers are.

- **AI-generated strategy code is never executed in the trading process.**
  `ResearchProposal` carries code as inert text. Running it requires a
  separate, explicitly-invoked sandbox outside this module, and the
  pipeline records only *results*, never behaviour. Nothing here imports,
  `exec`s, or otherwise evaluates proposed code.

- **Rejections are permanent for that candidate version.** A rejected
  candidate cannot be resubmitted at the same version; it must be revised,
  which produces a new version and a fresh audit record. Otherwise a
  candidate could be retried until noise carried it through a gate.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum

import structlog
from pydantic import BaseModel, Field

from backtesting.metrics import MetricsResult
from backtesting.statistics import DeflationReport
from backtesting.walk_forward import DegradationReport

log = structlog.get_logger(__name__)


class PromotionStage(StrEnum):
    RESEARCH = "RESEARCH"
    BACKTEST = "BACKTEST"
    VALIDATION = "VALIDATION"
    PAPER = "PAPER"
    APPROVED = "APPROVED"  # human sign-off obtained, not yet live
    LIVE = "LIVE"
    REJECTED = "REJECTED"


_NEXT_STAGE: dict[PromotionStage, PromotionStage | None] = {
    PromotionStage.RESEARCH: PromotionStage.BACKTEST,
    PromotionStage.BACKTEST: PromotionStage.VALIDATION,
    PromotionStage.VALIDATION: PromotionStage.PAPER,
    PromotionStage.PAPER: PromotionStage.APPROVED,
    PromotionStage.APPROVED: PromotionStage.LIVE,
    PromotionStage.LIVE: None,
    PromotionStage.REJECTED: None,
}


class PromotionRefused(Exception):
    """Raised when a promotion is not permitted. Never downgraded to a
    warning: a refused promotion that proceeds anyway defeats the entire
    pipeline."""


class GateCriteria(BaseModel):
    """Objective thresholds. Deliberately conservative, and deliberately
    not tunable by the AI — a candidate that can lower its own bar has no
    bar at all."""

    min_trades: int = 30
    min_out_of_sample_trades: int = 20
    max_drawdown: float = 0.25
    min_profit_factor: float = 1.0
    max_sharpe_before_suspicion: float = 3.0
    min_paper_days: int = 20
    min_paper_trades: int = 20
    # NOTE: there is deliberately no configurable DSR threshold here.
    # Enforcement uses DeflationReport.likely_genuine (backtesting/
    # statistics.py), a fixed 95%-confidence-and-30-trades bar, as the
    # single source of truth — a duplicated, independently-tunable
    # threshold here would risk drifting out of sync with it.


class ResearchProposal(BaseModel):
    """An AI-generated hypothesis. `code` is inert text: nothing in this
    module executes it."""

    model_config = {"frozen": True}

    proposal_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    name: str
    version: str = "0.1.0"
    hypothesis: str
    rationale: str = ""
    proposed_by: str = "ai"
    code: str = ""  # NEVER executed here
    params: dict = Field(default_factory=dict)


class HumanApproval(BaseModel):
    """Evidence of human sign-off. All fields required: an approval with
    no named approver is not an approval."""

    approver: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    approved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    reviewed_backtest: bool = False
    reviewed_paper_results: bool = False

    @property
    def is_complete(self) -> bool:
        return (
            bool(self.approver.strip())
            and bool(self.rationale.strip())
            and self.reviewed_backtest
            and self.reviewed_paper_results
        )

    def missing(self) -> list[str]:
        gaps = []
        if not self.approver.strip():
            gaps.append("approver identity")
        if not self.rationale.strip():
            gaps.append("written rationale")
        if not self.reviewed_backtest:
            gaps.append("confirmation that backtest results were reviewed")
        if not self.reviewed_paper_results:
            gaps.append("confirmation that paper results were reviewed")
        return gaps


class StageRecord(BaseModel):
    stage: PromotionStage
    entered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    passed: bool = False
    detail: str = ""
    failures: list[str] = Field(default_factory=list)


class PaperResults(BaseModel):
    days_running: int = 0
    trades: int = 0
    metrics: MetricsResult | None = None


class StrategyCandidate(BaseModel):
    """A strategy working its way toward production."""

    candidate_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    proposal: ResearchProposal
    stage: PromotionStage = PromotionStage.RESEARCH
    history: list[StageRecord] = Field(default_factory=list)

    backtest_metrics: MetricsResult | None = None
    degradation: DegradationReport | None = None
    deflation: DeflationReport | None = None
    paper_results: PaperResults | None = None
    approval: HumanApproval | None = None
    rejection_reason: str | None = None

    model_config = {"arbitrary_types_allowed": True}

    @property
    def is_rejected(self) -> bool:
        return self.stage is PromotionStage.REJECTED

    @property
    def is_live(self) -> bool:
        return self.stage is PromotionStage.LIVE

    def audit_trail(self) -> str:
        lines = [
            f"Candidate {self.candidate_id} — {self.proposal.name} "
            f"v{self.proposal.version} (proposed by {self.proposal.proposed_by})",
            f"Hypothesis: {self.proposal.hypothesis}",
            f"Current stage: {self.stage}",
        ]
        for record in self.history:
            verdict = "PASS" if record.passed else "FAIL"
            lines.append(f"  [{verdict}] {record.stage} at {record.entered_at.isoformat()}")
            if record.detail:
                lines.append(f"         {record.detail}")
            for failure in record.failures:
                lines.append(f"         - {failure}")
        if self.deflation is not None and self.deflation.has_result:
            lines.append(
                f"Deflated Sharpe Ratio: {self.deflation.deflated_sharpe_ratio:.1%} "
                f"(searched {self.deflation.n_trials} parameter combinations)"
            )
        if self.approval is not None:
            lines.append(
                f"Approved by {self.approval.approver}: {self.approval.rationale}"
            )
        if self.rejection_reason:
            lines.append(f"REJECTED: {self.rejection_reason}")
        return "\n".join(lines)


class PromotionPipeline:
    def __init__(self, criteria: GateCriteria | None = None) -> None:
        self._criteria = criteria or GateCriteria()
        self._candidates: dict[str, StrategyCandidate] = {}
        # Version keys of rejected candidates: a rejection is permanent
        # for that version, so it cannot be retried until noise carries it
        # through a gate.
        self._rejected_versions: set[str] = set()

    @property
    def criteria(self) -> GateCriteria:
        return self._criteria

    def candidates(self) -> list[StrategyCandidate]:
        return list(self._candidates.values())

    def get(self, candidate_id: str) -> StrategyCandidate | None:
        return self._candidates.get(candidate_id)

    def live_candidates(self) -> list[StrategyCandidate]:
        return [c for c in self._candidates.values() if c.is_live]

    def submit(self, proposal: ResearchProposal) -> StrategyCandidate:
        key = f"{proposal.name}@{proposal.version}"
        if key in self._rejected_versions:
            raise PromotionRefused(
                f"{key} was previously rejected. Revise the strategy and submit a new "
                "version rather than resubmitting the same one."
            )
        candidate = StrategyCandidate(proposal=proposal)
        candidate.history.append(
            StageRecord(
                stage=PromotionStage.RESEARCH,
                passed=True,
                detail=f"Proposal accepted for evaluation: {proposal.hypothesis[:120]}",
            )
        )
        self._candidates[candidate.candidate_id] = candidate
        log.info(
            "promotion.submitted",
            candidate_id=candidate.candidate_id,
            name=proposal.name,
            version=proposal.version,
            proposed_by=proposal.proposed_by,
        )
        return candidate

    # ---- gates ---------------------------------------------------------------

    def _check_backtest(self, candidate: StrategyCandidate) -> list[str]:
        metrics = candidate.backtest_metrics
        if metrics is None:
            return ["No backtest results attached"]
        c = self._criteria
        failures = []
        if metrics.n_trades < c.min_trades:
            failures.append(
                f"Only {metrics.n_trades} trades; {c.min_trades} required for the "
                "metrics to mean anything"
            )
        if metrics.max_drawdown is not None and metrics.max_drawdown > c.max_drawdown:
            failures.append(
                f"Max drawdown {metrics.max_drawdown:.1%} exceeds {c.max_drawdown:.1%}"
            )
        if metrics.profit_factor is None:
            failures.append("Profit factor not computable")
        elif metrics.profit_factor < c.min_profit_factor:
            failures.append(
                f"Profit factor {metrics.profit_factor:.2f} below {c.min_profit_factor}"
            )
        if metrics.sharpe is not None and metrics.sharpe > c.max_sharpe_before_suspicion:
            failures.append(
                f"Sharpe {metrics.sharpe:.2f} is implausibly high — investigate for "
                "look-ahead bias or unrealistic fills before proceeding"
            )
        return failures

    def _check_validation(self, candidate: StrategyCandidate) -> list[str]:
        report = candidate.degradation
        if report is None:
            return ["No out-of-sample degradation report attached"]
        c = self._criteria
        failures = []
        if report.is_likely_overfit:
            failures.append(
                "Out-of-sample performance does not support the in-sample result "
                "(overfitting signature detected)"
            )
        if report.out_of_sample.n_trades < c.min_out_of_sample_trades:
            failures.append(
                f"Only {report.out_of_sample.n_trades} out-of-sample trades; "
                f"{c.min_out_of_sample_trades} required"
            )
        oos_return = report.out_of_sample.total_return
        if oos_return is not None and oos_return <= 0:
            failures.append(f"Out-of-sample return is {oos_return:.2%}")

        # Deflated Sharpe Ratio: a stricter, statistically explicit check.
        # Optional because it needs the full trial set from a grid search
        # (grid_search_with_trials), which not every candidate will have
        # run. When present, it is not skipped or softened — a failing
        # DSR blocks promotion exactly like any other failed check.
        #
        # Deliberately checks `likely_genuine`, not the raw
        # deflated_sharpe_ratio number directly: likely_genuine also
        # enforces the minimum-trade-count guard (a headline DSR can be
        # near 100% while resting on far too few trades to trust the
        # independence assumption behind it — see
        # backtesting/statistics.py). Comparing the raw number here would
        # silently bypass that guard.
        deflation = candidate.deflation
        if deflation is not None:
            if not deflation.has_result:
                failures.append(
                    "Deflated Sharpe Ratio could not be computed: "
                    + "; ".join(deflation.warnings)
                )
            elif not deflation.likely_genuine:
                dsr_text = (
                    f"{deflation.deflated_sharpe_ratio:.1%}"
                    if deflation.deflated_sharpe_ratio is not None
                    else "n/a"
                )
                failures.append(
                    f"Deflated Sharpe Ratio {dsr_text} from {deflation.n_trades} trades "
                    f"across {deflation.n_trials} searched parameter combinations does not "
                    "clear the bar for statistical significance"
                )
        return failures

    def _check_paper(self, candidate: StrategyCandidate) -> list[str]:
        results = candidate.paper_results
        if results is None:
            return ["No paper-trading results attached"]
        c = self._criteria
        failures = []
        if results.days_running < c.min_paper_days:
            failures.append(
                f"Only {results.days_running} days of paper trading; "
                f"{c.min_paper_days} required"
            )
        if results.trades < c.min_paper_trades:
            failures.append(
                f"Only {results.trades} paper trades; {c.min_paper_trades} required"
            )
        return failures

    def _check_approval(self, candidate: StrategyCandidate) -> list[str]:
        approval = candidate.approval
        if approval is None:
            return ["No human approval recorded"]
        if not approval.is_complete:
            return [f"Approval incomplete: missing {', '.join(approval.missing())}"]
        return []

    # ---- promotion ------------------------------------------------------------

    def promote(
        self,
        candidate_id: str,
        target: PromotionStage,
        *,
        approval: HumanApproval | None = None,
    ) -> StrategyCandidate:
        candidate = self._candidates.get(candidate_id)
        if candidate is None:
            raise PromotionRefused(f"Unknown candidate {candidate_id}")
        if candidate.is_rejected:
            raise PromotionRefused(
                f"Candidate {candidate_id} was rejected: {candidate.rejection_reason}"
            )

        expected = _NEXT_STAGE[candidate.stage]
        if expected is None:
            raise PromotionRefused(f"{candidate.stage} is terminal")
        if target is not expected:
            raise PromotionRefused(
                f"Illegal promotion {candidate.stage} -> {target}. The only legal next "
                f"stage is {expected}. Stages cannot be skipped, however good the "
                "results look."
            )

        if approval is not None:
            candidate.approval = approval

        checks = {
            PromotionStage.BACKTEST: self._check_backtest,
            PromotionStage.VALIDATION: self._check_validation,
            PromotionStage.PAPER: self._check_paper,
            PromotionStage.APPROVED: self._check_approval,
            PromotionStage.LIVE: self._check_approval,
        }
        failures = checks.get(target, lambda _c: [])(candidate)

        record = StageRecord(stage=target, passed=not failures, failures=failures)

        if failures:
            candidate.history.append(record)
            log.warning(
                "promotion.refused",
                candidate_id=candidate_id,
                target=target,
                failures=failures,
            )
            raise PromotionRefused(
                f"Promotion to {target} refused: " + "; ".join(failures)
            )

        if target is PromotionStage.LIVE:
            # Belt and braces: the approval gate already ran, but live
            # promotion is the one place worth checking twice.
            if candidate.approval is None or not candidate.approval.is_complete:
                raise PromotionRefused(
                    "Live promotion requires complete human approval. "
                    "The AI cannot approve its own strategy."
                )
            log.critical(
                "promotion.live",
                candidate_id=candidate_id,
                name=candidate.proposal.name,
                approver=candidate.approval.approver,
            )

        record.detail = f"Promoted from {candidate.stage} to {target}"
        candidate.history.append(record)
        candidate.stage = target
        log.info("promotion.advanced", candidate_id=candidate_id, stage=target)
        return candidate

    def reject(self, candidate_id: str, reason: str) -> StrategyCandidate:
        candidate = self._candidates.get(candidate_id)
        if candidate is None:
            raise PromotionRefused(f"Unknown candidate {candidate_id}")
        candidate.stage = PromotionStage.REJECTED
        candidate.rejection_reason = reason
        candidate.history.append(
            StageRecord(stage=PromotionStage.REJECTED, passed=False, detail=reason)
        )
        self._rejected_versions.add(
            f"{candidate.proposal.name}@{candidate.proposal.version}"
        )
        log.warning("promotion.rejected", candidate_id=candidate_id, reason=reason)
        return candidate

    def attach_backtest(self, candidate_id: str, metrics: MetricsResult) -> None:
        self._require(candidate_id).backtest_metrics = metrics

    def attach_degradation(self, candidate_id: str, report: DegradationReport) -> None:
        self._require(candidate_id).degradation = report

    def attach_deflation(self, candidate_id: str, report: DeflationReport) -> None:
        self._require(candidate_id).deflation = report

    def attach_paper_results(self, candidate_id: str, results: PaperResults) -> None:
        self._require(candidate_id).paper_results = results

    def _require(self, candidate_id: str) -> StrategyCandidate:
        candidate = self._candidates.get(candidate_id)
        if candidate is None:
            raise PromotionRefused(f"Unknown candidate {candidate_id}")
        return candidate
