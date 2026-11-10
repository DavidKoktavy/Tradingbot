"""
Universe selector: scanner proposes, approved list disposes, scoring ranks.

Design decisions:

- **Candidates = scanner hits ∩ approved list**, in scanner order. If the
  scanner is not configured, fails, or returns nothing that is approved,
  the selector falls back to scoring the whole approved list and records
  *why* in `candidate_source`. A scanner outage must degrade selection,
  not stop it — and must never widen it.

- **Held symbols are always kept.** A symbol with an open position stays
  in the universe even if it no longer ranks. Dropping it would leave the
  position with no strategy evaluating exits, and with no mark for the
  risk engine's daily-loss and exposure checks, which then (correctly)
  refuse every trade.

- **Never raises.** A failed history request marks that one candidate
  ineligible with the reason; any other fault yields an empty selection
  plus the held symbols. The caller keeps trading what it already holds.

- **Bounded broker requests.** At most `max_candidates` historical-data
  requests per selection, optionally spaced by `request_spacing_seconds`,
  so a selection run cannot trip IBKR pacing limits and get the session
  throttled.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

import structlog

from data.models import Bar, Instrument
from universe.approved import ApprovedUniverse
from universe.scoring import CandidateScore, ScoringParams, score_candidate

log = structlog.get_logger(__name__)


class HistoricalBarSource(Protocol):
    async def get_historical_bars(
        self, instrument: Instrument, *, duration: str, bar_size: str, end: datetime | None = None
    ) -> list[Bar]: ...


class CandidateScanner(Protocol):
    async def scan(self) -> list[str]: ...


@dataclass
class UniverseSelection:
    selected: list[str] = field(default_factory=list)
    held: list[str] = field(default_factory=list)
    scored: list[CandidateScore] = field(default_factory=list)
    candidate_source: str = ""
    scanner_hits: list[str] = field(default_factory=list)
    scanner_rejected: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def symbols(self) -> list[str]:
        """Everything the loop should track: held first, then picks."""
        out: list[str] = []
        for s in [*self.held, *self.selected]:
            if s not in out:
                out.append(s)
        return out


class UniverseSelector:
    def __init__(
        self,
        approved: ApprovedUniverse,
        bars: HistoricalBarSource,
        *,
        scanner: CandidateScanner | None = None,
        top_n: int = 5,
        max_candidates: int = 60,
        params: ScoringParams | None = None,
        history_duration: str = "1 Y",
        request_spacing_seconds: float = 0.0,
    ) -> None:
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        self._approved = approved
        self._bars = bars
        self._scanner = scanner
        self._top_n = top_n
        self._max_candidates = max_candidates
        self._params = params or ScoringParams()
        self._duration = history_duration
        self._spacing = request_spacing_seconds

    @property
    def approved(self) -> ApprovedUniverse:
        return self._approved

    async def select(self, held: Iterable[str] = ()) -> UniverseSelection:
        selection = UniverseSelection(held=[h.upper() for h in held])
        try:
            candidates = await self._candidates(selection)
            if len(candidates) > self._max_candidates:
                log.warning(
                    "universe.candidates_truncated",
                    candidates=len(candidates),
                    max_candidates=self._max_candidates,
                )
                candidates = candidates[: self._max_candidates]

            hits = set(selection.scanner_hits)
            for i, symbol in enumerate(candidates):
                if i and self._spacing:
                    await asyncio.sleep(self._spacing)
                score = await self._score(symbol)
                score.scanner_hit = symbol in hits
                selection.scored.append(score)

            ranked = sorted(
                (s for s in selection.scored if s.eligible),
                key=lambda s: (-s.score, s.symbol),
            )
            selection.selected = [s.symbol for s in ranked[: self._top_n]]
        except Exception as exc:  # noqa: BLE001 — selection must not crash the bot
            log.exception("universe.selection_failed", error=str(exc))
            selection.selected = []

        log.info(
            "universe.selected",
            source=selection.candidate_source,
            selected=selection.selected,
            held=selection.held,
            scored=len(selection.scored),
            eligible=sum(1 for s in selection.scored if s.eligible),
            scanner_rejected=selection.scanner_rejected[:10],
        )
        return selection

    async def _candidates(self, selection: UniverseSelection) -> list[str]:
        if self._scanner is None:
            selection.candidate_source = "approved_list"
            return list(self._approved.symbols)

        try:
            raw = await self._scanner.scan()
        except Exception as exc:  # noqa: BLE001
            log.error("universe.scanner_failed", error=str(exc))
            selection.candidate_source = "approved_list (scanner failed)"
            return list(self._approved.symbols)

        approved, rejected = self._approved.filter(raw)
        selection.scanner_hits = approved
        selection.scanner_rejected = rejected
        if not approved:
            selection.candidate_source = "approved_list (no approved scanner hits)"
            return list(self._approved.symbols)
        selection.candidate_source = "scanner"
        return approved

    async def _score(self, symbol: str) -> CandidateScore:
        try:
            bars = await self._bars.get_historical_bars(
                Instrument(symbol=symbol), duration=self._duration, bar_size="1 day"
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("universe.history_failed", symbol=symbol, error=str(exc))
            failed = CandidateScore(symbol=symbol)
            failed.reasons.append(f"history request failed: {exc}")
            return failed
        return score_candidate(symbol, bars, self._params)
