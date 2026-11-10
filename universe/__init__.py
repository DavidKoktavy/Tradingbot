"""
Automatic stock selection.

The control loop trades a list of instruments; this package decides what
goes on that list, without an operator typing `--symbols`.

    IBKR scanner (optional)  ->  proposes candidates
    approved list            ->  the hard boundary: nothing else is tradeable
    scoring                  ->  deterministic ranking on daily bars
    selector                 ->  top N, plus every symbol already held
    refresher                ->  applies the selection to the running loop

Selection decides what the bot *looks at*. It never places an order: every
trade on a selected symbol still comes from a strategy signal and still has
to pass the risk engine.
"""

from universe.approved import ApprovedUniverse
from universe.scoring import CandidateScore, ScoringParams, score_candidate
from universe.selector import UniverseSelection, UniverseSelector

__all__ = [
    "ApprovedUniverse",
    "CandidateScore",
    "ScoringParams",
    "UniverseSelection",
    "UniverseSelector",
    "score_candidate",
]
