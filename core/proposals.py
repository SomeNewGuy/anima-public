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


"""Belief proposal interface — standardized shape for all belief sources.

Every path that wants to create or modify a belief produces a BeliefProposal.
All proposals flow through submit_proposal() which handles governance
(triage → auto-accept / queue / reject) before anything touches the graph.

Sources: dream synthesis, exploration extraction, corrections, lessons,
         resolutions, CLI auto-formation, external ingestion, batch seeding.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("proposals")


@dataclass
class BeliefProposal:
    """Standard shape for a belief entering the governance pipeline.

    Every source — dreams, extraction, corrections, seeding, external API —
    produces one of these. The governance layer consumes them uniformly.
    """

    # --- Content ---
    statement: str
    confidence: float = 0.5
    topics: Optional[list] = None
    entities: Optional[list] = None
    supporting_evidence: list = field(default_factory=list)

    # --- Provenance ---
    source: str = "unknown"              # "exploration", "correction", "conversation", "ingestion"
    source_type: Optional[str] = None    # "external", "synthesis", "meta", "identity", "anchor"
    source_episode: Optional[str] = None
    generation_type: Optional[str] = None  # "dream", "operator_seeded", "triplet", ...
    generation_cycle: Optional[int] = None
    extraction_context: str = "operator"   # "operator" or "exploration"

    # --- Synthesis lineage ---
    parent_a: Optional[str] = None
    parent_b: Optional[str] = None
    parent_c: Optional[str] = None       # triplet dreams
    abstraction_depth: int = 0

    # --- Governance flags ---
    operator_anchored: int = 0

    # --- Extended provenance (variant-specific, future item 4) ---
    provenance_meta: Optional[dict] = None  # arbitrary key-value pairs

    # --- Internal ---
    proposal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # --- Dream-specific (for triage, not stored on belief) ---
    similarity: Optional[float] = None   # embedding similarity for dream pairs
    dream_data: Optional[dict] = None    # full dream payload for _triage_dream


@dataclass
class TriageDecision:
    """Result of evaluating a proposal through the governance triage."""

    decision: str    # "accept", "reject", "queue"
    reason: str      # human-readable rationale
    category: str    # "belief", "dream", "correction", "lesson", etc.
    recommended: bool = False  # for queued items: recommend acceptance?


class TriageStrategy:
    """Base class for pluggable triage strategies.

    Each source type can have its own triage implementation.
    The interface: accept a proposal, return a TriageDecision.
    """

    def evaluate(self, proposal: BeliefProposal) -> TriageDecision:
        """Evaluate a proposal. Override in subclasses."""
        return TriageDecision(
            decision="queue",
            reason="no triage strategy configured",
            category="belief",
        )


class DefaultTriageStrategy(TriageStrategy):
    """Passthrough — queues everything for human review.

    Used as fallback for source types without a specific triage strategy.
    Safest default: nothing enters the graph without operator approval.
    """

    def evaluate(self, proposal: BeliefProposal) -> TriageDecision:
        return TriageDecision(
            decision="queue",
            reason="default strategy: requires review",
            category=proposal.generation_type or "belief",
            recommended=proposal.confidence >= 0.6,
        )


def _infer_category(proposal: BeliefProposal) -> str:
    """Infer the governance category from proposal fields.

    Maps to existing categories: belief, dream, triplet_dream, correction,
    lesson, reflection. This maintains backward compatibility with the
    current approval_queue.category values and _process_approved_item dispatch.
    """
    if proposal.generation_type == "dream":
        return "dream"
    if proposal.generation_type == "triplet":
        return "triplet_dream"
    if proposal.source == "correction":
        return "correction"
    # Lessons are high-confidence behavioral principles from user feedback
    if proposal.source == "exploration" and proposal.confidence >= 0.8:
        if any("lesson" in e.lower() or "behavioral" in e.lower()
               for e in proposal.supporting_evidence):
            return "lesson"
    return "belief"


class DepthAwareTriageLayer:
    """Depth governance layer for synthesis beliefs (P0 mitigation).

    Applies structural validation to beliefs above the auto-accept depth
    ceiling. Uses LLM classification to check for concrete referents —
    not keyword matching.

    Configurable thresholds per instance:
    - Below auto_ceiling: pass through to normal triage
    - Between auto_ceiling and review_ceiling: must pass structural validation
    - Above review_ceiling: rejected if validation fails
    - Above hard_cap (if set): rejected outright, no validation

    Generator stays dumb, governance gets smarter.
    """

    VALIDATION_PROMPT = (
        "Evaluate this belief statement for concrete, verifiable content.\n\n"
        "IMPORTANT: The system's own name (e.g., ANIMA) does NOT count as a named entity — "
        "it is the subject of the corpus, not a distinguishing referent. Named entities must be "
        "specific components, mechanisms, metrics, or external references that differentiate "
        "this belief from any other belief about the same system.\n\n"
        "Statement: {statement}\n\n"
        "Answer YES or NO for each:\n"
        "1. NAMED_ENTITY: Does it reference a specific named entity "
        "(specific component, mechanism, metric name, person, external reference)? "
        "The system's own name does NOT qualify.\n"
        "2. MEASURABLE: Does it contain a specific measurable quantity "
        "(an actual number, threshold, ratio, percentage — not just 'quantities exist')?\n"
        "3. TESTABLE: Does it describe a concrete intervention or testable action "
        "(something specific that could be toggled, run, or experimentally verified — "
        "not a vague process description)?\n\n"
        "Example that PASSES:\n"
        "\"ANIMA Mini's ~3B governance-trained model operates within the meta-cognitive loop\"\n"
        "→ NAMED_ENTITY: YES (Mini, 3B — specific component with specific parameter)\n"
        "→ MEASURABLE: YES (~3B — specific quantity)\n"
        "→ TESTABLE: YES (run Mini, verify loop operation)\n\n"
        "Example that FAILS:\n"
        "\"ANIMA's self-reinforcing meta-cognitive architecture creates emergent synergy "
        "between governance and synthesis\"\n"
        "→ NAMED_ENTITY: NO (ANIMA alone is the corpus subject, not a distinguishing referent)\n"
        "→ MEASURABLE: NO (no specific quantity or threshold)\n"
        "→ TESTABLE: NO (no intervention described — 'creates emergent synergy' is not verifiable)\n\n"
        "Format: NAMED_ENTITY: YES/NO | MEASURABLE: YES/NO | TESTABLE: YES/NO"
    )

    def __init__(self, config, inference_engine=None):
        extraction_cfg = config.get("extraction", {})
        self.auto_ceiling = extraction_cfg.get("synthesis_depth_auto_ceiling", 3)
        self.review_ceiling = extraction_cfg.get("synthesis_depth_review_ceiling", 5)
        self.hard_cap = extraction_cfg.get("synthesis_depth_hard_cap", None)
        self.inference = inference_engine

    def check(self, proposal: BeliefProposal) -> TriageDecision | None:
        """Check depth governance. Returns TriageDecision if intercepted, None to pass through."""
        depth = proposal.abstraction_depth

        # Strict depth control: reject depth ≥3 when parent is already ≥2.
        # This prevents abstraction towers (depth 4+ chains of recursive synthesis).
        if depth >= 3 and (proposal.parent_a or proposal.parent_b):
            return TriageDecision(
                decision="reject",
                reason=f"depth {depth} >= 3: deep abstraction chain rejected",
                category="depth_governance",
            )

        if depth < self.auto_ceiling:
            return None  # Below ceiling, pass through

        # Hard cap — reject outright, no validation
        if self.hard_cap is not None and depth > self.hard_cap:
            return TriageDecision(
                decision="reject",
                reason=f"depth {depth} > hard cap {self.hard_cap}",
                category="depth_governance",
            )

        # Run structural validation via LLM
        has_concrete = self._validate_structural(proposal.statement)

        if has_concrete:
            return None  # Passes validation, continue to normal triage

        # Fails validation — route depends on depth vs review ceiling
        if depth >= self.review_ceiling:
            return TriageDecision(
                decision="reject",
                reason=(
                    f"depth {depth} >= review ceiling {self.review_ceiling}, "
                    f"failed structural validation (no entity/quantity/testable action)"
                ),
                category="depth_governance",
            )

        # Between auto and review ceiling — queue for review
        return TriageDecision(
            decision="queue",
            reason=(
                f"depth {depth} >= auto ceiling {self.auto_ceiling}, "
                f"failed structural validation (no entity/quantity/testable action)"
            ),
            category="depth_governance",
        )

    def _validate_structural(self, statement: str) -> bool:
        """LLM classification: does this belief contain concrete referents?

        Returns True if at least one of: named entity, measurable quantity,
        or testable action is present.
        """
        if not self.inference:
            # No inference engine — fall back to pass-through
            logger.warning("Depth triage: no inference engine, skipping validation")
            return True

        prompt = self.VALIDATION_PROMPT.format(statement=statement)
        try:
            response = self.inference.generate_with_messages(
                [
                    {"role": "system", "content": "You are a precise classifier. Answer only in the requested format."},
                    {"role": "user", "content": prompt + " /no_think"},
                ],
                max_tokens=64,
                temperature=0.0,
                timeout=30, task="triage",
            )
            resp_upper = response.upper()
            has_entity = "NAMED_ENTITY: YES" in resp_upper
            has_measurable = "MEASURABLE: YES" in resp_upper
            has_testable = "TESTABLE: YES" in resp_upper
            score = sum([has_entity, has_measurable, has_testable])
            return score >= 2  # Requires 2-of-3 concrete criteria

        except Exception as e:
            logger.warning(
                f"Depth validation FAIL-OPEN: {e} — "
                f"belief passed without validation: {statement[:80]}"
            )
            # Fail-open telemetry: log structured error for diagnostics
            try:
                import sqlite3
                import os
                telemetry_path = os.environ.get("ANIMA_DATA_DIR", "data")
                tel_db_path = os.path.join(telemetry_path, "telemetry.db")
                tel_db = sqlite3.connect(tel_db_path)
                tel_db.execute(
                    "CREATE TABLE IF NOT EXISTS validation_failures "
                    "(timestamp TEXT, statement TEXT, error TEXT, "
                    "error_type TEXT, depth INTEGER)"
                )
                from datetime import datetime, timezone as _tz
                tel_db.execute(
                    "INSERT INTO validation_failures "
                    "(timestamp, statement, error, error_type, depth) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        datetime.now(_tz.utc).isoformat(),
                        statement[:500],
                        str(e)[:200],
                        type(e).__name__,
                        0,  # depth not available here
                    ),
                )
                tel_db.commit()
                tel_db.close()
            except Exception:
                pass  # Telemetry must never block triage
            return True  # On failure, don't block — let normal triage decide


class ProposalGateway:
    """Single entry point for all belief proposals into the governance pipeline.

    Routes proposals through triage → accept/reject/queue. The gateway
    replaces direct add_belief() calls, ensuring every path goes through
    governance.

    Usage:
        gateway = ProposalGateway()
        gateway.register_strategy("dream", DreamTriageStrategy(evolution_engine))
        gateway.register_strategy("correction", CorrectionTriageStrategy())

        result = gateway.submit(proposal, on_accept=..., on_queue=..., on_reject=...)

    The on_accept/on_queue/on_reject callbacks decouple the gateway from
    storage details (SemanticMemory, approval_queue, shadow log). This keeps
    the gateway testable without database dependencies.
    """

    def __init__(self, config=None, inference_engine=None):
        self._strategies: dict[str, TriageStrategy] = {}
        self._default_strategy = DefaultTriageStrategy()
        self._governance = (config or {}).get("_governance", {
            "allow_auto_accept": True,
            "allow_auto_corrections": True,
            "allow_auto_lessons": True,
        })
        self._depth_layer = (
            DepthAwareTriageLayer(config, inference_engine)
            if config else None
        )

    def register_strategy(self, category: str, strategy: TriageStrategy):
        """Register a triage strategy for a specific category."""
        self._strategies[category] = strategy

    def submit(self, proposal: BeliefProposal,
               on_accept=None, on_queue=None, on_reject=None) -> TriageDecision:
        """Submit a proposal through the governance pipeline.

        Args:
            proposal: The belief proposal to evaluate.
            on_accept: Callback(proposal, decision) — store the belief.
            on_queue: Callback(proposal, decision) — add to approval queue.
            on_reject: Callback(proposal, decision) — log the rejection.

        Returns:
            The TriageDecision made for this proposal.
        """
        # Depth governance — check before normal triage
        if self._depth_layer:
            depth_decision = self._depth_layer.check(proposal)
            if depth_decision:
                logger.info(
                    f"Proposal {proposal.proposal_id[:8]}: depth governance "
                    f"→ {depth_decision.decision} ({depth_decision.reason})"
                )
                return depth_decision

        category = _infer_category(proposal)
        strategy = self._strategies.get(category, self._default_strategy)

        decision = strategy.evaluate(proposal)

        # Governance override — monarchy blocks auto-accepts
        if (not self._governance.get("allow_auto_accept", True)
                and decision.decision == "accept"):
            decision = TriageDecision(
                decision="queue",
                reason=f"governance:monarchy ({decision.reason})",
                category=decision.category,
                recommended=True,
            )

        logger.info(
            f"Proposal {proposal.proposal_id[:8]} [{category}]: "
            f"{decision.decision} — {decision.reason}"
        )

        if decision.decision == "accept" and on_accept:
            on_accept(proposal, decision)
        elif decision.decision == "queue" and on_queue:
            on_queue(proposal, decision)
        elif decision.decision == "reject" and on_reject:
            on_reject(proposal, decision)

        return decision

    def submit_batch(self, proposals: list[BeliefProposal],
                     on_accept=None, on_queue=None,
                     on_reject=None) -> list[TriageDecision]:
        """Submit a batch of proposals. Returns list of decisions."""
        decisions = []
        for proposal in proposals:
            decision = self.submit(
                proposal,
                on_accept=on_accept,
                on_queue=on_queue,
                on_reject=on_reject,
            )
            decisions.append(decision)

        accepted = sum(1 for d in decisions if d.decision == "accept")
        queued = sum(1 for d in decisions if d.decision == "queue")
        rejected = sum(1 for d in decisions if d.decision == "reject")
        logger.info(
            f"Batch triage: {len(proposals)} proposals — "
            f"{accepted} accepted, {queued} queued, {rejected} rejected"
        )
        return decisions
