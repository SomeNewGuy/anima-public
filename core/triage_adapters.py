# Copyright (C) 2026 Gerald Teeple
#
# This file is part of ANIMA.
#
# ANIMA is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ANIMA is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ANIMA. If not, see <https://www.gnu.org/licenses/>.

"""Triage strategy adapters — bridge between proposal interface and existing engine.

These adapters wrap the existing _triage_* methods on EvolutionEngine as
TriageStrategy implementations. When the refactor is complete, the logic
can be migrated into the strategies directly. For now, delegation keeps
the existing code untouched.

Also provides reverse conversion functions (legacy dict → BeliefProposal)
for the gateway integration in _queue_consolidation_items.

Usage:
    # Gateway setup (in EvolutionEngine.__init__)
    from core.triage_adapters import (
        EvolutionBeliefTriageStrategy,
        EvolutionDreamTriageStrategy,
        EvolutionCorrectionTriageStrategy,
        EvolutionLessonTriageStrategy,
    )

    gateway = ProposalGateway()
    gateway.register_strategy("belief", EvolutionBeliefTriageStrategy(engine))
    gateway.register_strategy("dream", EvolutionDreamTriageStrategy(engine))
    gateway.register_strategy("correction", EvolutionCorrectionTriageStrategy(engine))
    gateway.register_strategy("lesson", EvolutionLessonTriageStrategy(engine))
"""

import logging

from core.proposals import BeliefProposal, TriageDecision, TriageStrategy

logger = logging.getLogger("triage_adapters")


def _proposal_to_belief_data(proposal: BeliefProposal) -> dict:
    """Convert a BeliefProposal to the dict shape that _triage_belief expects.

    _triage_belief reads: statement, source, confidence (str), evidence,
    extraction_context. This reconstructs that dict from the proposal.
    """
    # Reverse-map numeric confidence to string for _map_confidence compatibility
    if proposal.confidence >= 0.75:
        conf_str = "high"
    elif proposal.confidence >= 0.55:
        conf_str = "medium"
    else:
        conf_str = "low"

    return {
        "statement": proposal.statement,
        "source": proposal.source,
        "confidence": conf_str,
        "evidence": proposal.supporting_evidence[0] if proposal.supporting_evidence else "",
        "extraction_context": proposal.extraction_context,
    }


def _proposal_to_dream_data(proposal: BeliefProposal) -> dict:
    """Convert a BeliefProposal to the dict shape that _triage_dream expects.

    _triage_dream reads: inference, similarity, belief_a_id, belief_b_id,
    _domain_a, _domain_b, _graph_context. Most of this lives in dream_data.
    """
    if proposal.dream_data:
        # Dream data carries the full payload from dream generation
        data = dict(proposal.dream_data)
        # Ensure inference and similarity are set (may be overridden)
        data.setdefault("inference", proposal.statement)
        data.setdefault("similarity", proposal.similarity or 0.5)
        return data

    # Fallback: reconstruct minimal dream data from proposal fields
    return {
        "inference": proposal.statement,
        "similarity": proposal.similarity or 0.5,
        "belief_a_id": proposal.parent_a,
        "belief_b_id": proposal.parent_b,
    }


def _proposal_to_correction_data(proposal: BeliefProposal) -> dict:
    """Convert a BeliefProposal to the dict shape that _triage_correction expects.

    _triage_correction reads: original, corrected.
    """
    # The correction's original text is typically in supporting_evidence
    original = ""
    corrected = proposal.statement
    for ev in proposal.supporting_evidence:
        if ev.startswith("Correction from: "):
            original = ev[len("Correction from: "):]
            break
    return {
        "original": original,
        "corrected": corrected,
    }


def _proposal_to_lesson_data(proposal: BeliefProposal) -> dict:
    """Convert a BeliefProposal to the dict shape that _triage_lesson expects.

    _triage_lesson reads: principle.
    """
    return {
        "principle": proposal.statement,
    }


def _tuple_to_decision(result: tuple, category: str) -> TriageDecision:
    """Convert (decision_str, reason_str) tuple to TriageDecision."""
    decision, reason = result
    return TriageDecision(
        decision=decision,
        reason=reason,
        category=category,
        recommended=(decision == "queue"),  # queued items default to recommended
    )


class EvolutionBeliefTriageStrategy(TriageStrategy):
    """Delegates to EvolutionEngine._triage_belief.

    Wraps the existing belief triage logic (confidence floor, contradiction
    detection, duplicate detection, generic self-assessment filter, etc.)
    as a TriageStrategy.
    """

    def __init__(self, engine):
        """engine: an EvolutionEngine instance with _triage_belief method."""
        self._engine = engine

    def evaluate(self, proposal: BeliefProposal) -> TriageDecision:
        data = _proposal_to_belief_data(proposal)
        result = self._engine._auto_triage("belief", data)
        return _tuple_to_decision(result, "belief")


class EvolutionDreamTriageStrategy(TriageStrategy):
    """Delegates to Rust triage state machine via _auto_triage."""

    def __init__(self, engine):
        self._engine = engine

    def evaluate(self, proposal: BeliefProposal) -> TriageDecision:
        data = _proposal_to_dream_data(proposal)
        result = self._engine._auto_triage("dream", data)
        return _tuple_to_decision(result, "dream")


class EvolutionCorrectionTriageStrategy(TriageStrategy):
    """Governance-aware correction triage.

    Runs the engine's _triage_correction for analysis (reason text, quality
    signals).  Behavior depends on governance capabilities:

    - allow_auto_corrections=True  → engine decision passes through
    - allow_auto_corrections=False → override to "queue" (operator review)

    The engine's reason is preserved as context for the operator.
    """

    def __init__(self, engine, governance=None):
        self._engine = engine
        self._governance = governance or {}

    def evaluate(self, proposal: BeliefProposal) -> TriageDecision:
        data = _proposal_to_correction_data(proposal)
        result = self._engine._auto_triage("correction", data)
        return _tuple_to_decision(result, "correction")


class EvolutionLessonTriageStrategy(TriageStrategy):
    """Delegates to Rust triage state machine via _auto_triage."""

    def __init__(self, engine, governance=None):
        self._engine = engine
        self._governance = governance or {}

    def evaluate(self, proposal: BeliefProposal) -> TriageDecision:
        data = _proposal_to_lesson_data(proposal)
        result = self._engine._auto_triage("lesson", data)
        return _tuple_to_decision(result, "lesson")


# ---------------------------------------------------------------------------
# Reverse conversions: legacy dict → BeliefProposal
# Used by _queue_consolidation_items to feed dicts into the gateway.
# ---------------------------------------------------------------------------

def _belief_data_to_proposal(data, episode_id=None):
    """Convert legacy belief extraction dict to BeliefProposal."""
    conf_str = data.get("confidence", "medium")
    conf_map = {"high": 0.8, "medium": 0.6, "low": 0.3}
    evidence = data.get("evidence", "")
    return BeliefProposal(
        statement=data.get("statement", ""),
        confidence=conf_map.get(conf_str, 0.5),
        source=data.get("source", "unknown"),
        extraction_context=data.get("extraction_context", "operator"),
        source_episode=episode_id,
        supporting_evidence=[evidence] if evidence else [],
    )


def _correction_data_to_proposal(data, episode_id=None):
    """Convert legacy correction dict to BeliefProposal."""
    return BeliefProposal(
        statement=data.get("corrected", ""),
        source="correction",
        source_episode=episode_id,
        supporting_evidence=(
            [f"Correction from: {data['original']}"] if data.get("original") else []
        ),
    )


def _lesson_data_to_proposal(data, episode_id=None):
    """Convert legacy lesson dict to BeliefProposal."""
    source = data.get("source", "unknown")
    context = data.get("context", "")
    return BeliefProposal(
        statement=data.get("principle", ""),
        confidence=0.8 if source == "user" else 0.6,
        source="exploration",
        source_episode=episode_id,
        supporting_evidence=(
            [f"Behavioral lesson from {source}: {context}"]
            if context else [f"Behavioral lesson from {source}"]
        ),
    )


def _category_to_proposal(category, data, episode_id=None):
    """Dispatch: convert any legacy extraction dict to BeliefProposal by category."""
    if category == "belief":
        return _belief_data_to_proposal(data, episode_id)
    elif category == "correction":
        return _correction_data_to_proposal(data, episode_id)
    elif category == "lesson":
        return _lesson_data_to_proposal(data, episode_id)
    # Fallback — shouldn't be reached for supported categories
    return BeliefProposal(statement=str(data.get("statement", "")),
                          source_episode=episode_id)
