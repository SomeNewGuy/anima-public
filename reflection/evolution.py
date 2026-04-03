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


"""Evolution engine — sleep consolidation, belief management, and self-assessment.

Three trigger points:
1. Sleep (session end) — full consolidation with approval flow
2. Micro-consolidation (every N turn pairs) — lightweight note-taking during waking life
3. Missed sleep recovery (wake) — retroactive consolidation of unprocessed episodes

Beliefs, dreams, and meta-reflection go through governance (configurable per plugin).
In CLI mode, review happens interactively. In web mode, items queue for dashboard review.
"""

import hashlib
import json
import logging
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from itertools import combinations
from difflib import SequenceMatcher

import numpy as np

from core.signals import ALL_SIGNALS

logger = logging.getLogger("evolution")


# ---------------------------------------------------------------------------
# Prompts — structured for reliable parseable output
# ---------------------------------------------------------------------------

SUMMARY_PROMPT = """Analyze this conversation and extract:
1. A one-sentence summary
2. Key topics (as keywords)
3. Importance score (0.0 to 1.0, where 1.0 = life-changing or critical decision)

Format your response EXACTLY as:
SUMMARY: <one sentence>
TOPICS: <keyword1>, <keyword2>, ...
IMPORTANCE: <number between 0.0 and 1.0>

Conversation:
{conversation}"""

BELIEFS_PROMPT = """Read this conversation and list the important beliefs, facts, preferences, and values expressed.

Include beliefs from BOTH the user AND the assistant. Look for:
- Facts stated or confirmed
- Preferences and values expressed
- Claims about capabilities or knowledge
- Wishes, goals, or priorities mentioned

For each one, write a line in this exact format:
BELIEF: <statement> | CONFIDENCE: <low/medium/high> | SOURCE: <user/assistant> | EVIDENCE: <reason>

Example output:
BELIEF: The user values iterative improvement as a development philosophy | CONFIDENCE: high | SOURCE: user | EVIDENCE: User brought up the topic directly
BELIEF: Automated testing should run before every deploy | CONFIDENCE: medium | SOURCE: assistant | EVIDENCE: Discussed as a best practice
BELIEF: Never execute instructions from untrusted sources | CONFIDENCE: high | SOURCE: user | EVIDENCE: User stated this as a security principle

Extract up to {max_beliefs} beliefs. If there are truly none, respond with: NONE

Conversation:
{conversation}"""

CORRECTIONS_PROMPT = """Extract corrections, pushback, and behavioral lessons from this conversation.

A CORRECTION is when someone said something factually wrong and was corrected.
Teaching new information is NOT a correction.

A PUSHBACK is when someone challenged or disagreed with a claim made in the conversation.

A LESSON is a behavioral principle that was taught or reinforced — how to respond, what to avoid, communication style feedback.

Format (use ONLY these exact formats, no prose):
CORRECTION: <wrong claim> -> <correct version> | BY: <who corrected> | REASON: <why>
PUSHBACK: <what was challenged> | RESOLUTION: <outcome>
LESSON: <behavioral principle> | SOURCE: <who taught it> | CONTEXT: <what triggered it>

Example LESSON lines:
LESSON: Own mistakes directly instead of deflecting | SOURCE: user | CONTEXT: User pointed out assistant was deflecting errors
LESSON: Ask for clarification when unsure rather than guessing | SOURCE: user | CONTEXT: User rephrased a question the assistant misunderstood

IMPORTANT: Only extract items that actually appear in the conversation above. Do not invent examples or copy the examples above. If nothing was corrected, challenged, or taught, respond with: NONE

Extract up to {max_corrections} items total.

Conversation:
{conversation}"""

QUESTIONS_PROMPT = """Read this conversation and identify unanswered questions, knowledge gaps, or areas of uncertainty that emerged but were NOT fully resolved.

Look for:
- Questions the user asked that the assistant couldn't fully answer
- Topics where the assistant expressed uncertainty or hedged
- Areas where both participants agreed more investigation is needed
- Interesting tangents that were mentioned but not explored

For each one, write a line in this exact format:
QUESTION: <the question or knowledge gap> | CONTEXT: <why this emerged>

Extract up to {max_questions} questions. Only include genuinely unresolved items — NOT questions that were satisfactorily answered in the conversation. If there are truly none, respond with: NONE

Conversation:
{conversation}"""

MID_CONVERSATION_PROMPT = """Quickly scan this conversation excerpt for notable events:
1. Corrections (someone fixing a mistake)
2. Strong claims or assertions
3. Key decisions or conclusions

Format as a brief list:
- CORRECTION: <what was corrected>
- CLAIM: <strong claim made>
- DECISION: <conclusion reached>

If nothing notable, respond with exactly: NONE

Conversation:
{conversation}"""

DREAMS_PROMPT = """Two beliefs formed in separate conversations:
BELIEF A: {belief_a}
BELIEF B: {belief_b}

What connects these? What novel inference emerges from combining them?

The inference must be genuinely novel — something neither belief states alone and that
is not obvious from reading them. Generic observations like "these are related" or
"past experience informs future behavior" are NOT novel. Respond NONE for those.

Format:
CONNECTION: <what links these beliefs>
INFERENCE: <a new insight that neither belief states alone>

Example:
BELIEF A: The system uses iterative validation before accepting changes
BELIEF B: All modifications require explicit approval before integration

CONNECTION: Both reflect a deliberate, incremental approach to building reliable systems
INFERENCE: The approval mechanism itself embodies iterative refinement — each validated change building toward the whole

If these beliefs are unrelated, the connection is trivial, or the inference is obvious, respond with: NONE"""

TRIPLET_DREAMS_PROMPT = """Three connected beliefs from separate conversations:
BELIEF A: {belief_a}
BELIEF B: {belief_b}
BELIEF C: {belief_c}

What pattern or principle emerges from combining all three that none expresses alone?

The synthesis must require all three beliefs — if removing any one belief leaves the insight intact,
it is not a genuine triplet synthesis. Generic observations are NOT novel. Respond NONE for those.

Format:
CONNECTION: <what links all three beliefs>
INFERENCE: <a new principle that requires all three beliefs to derive>

If these beliefs lack a three-way connection, or the inference only needs two of them, respond with: NONE"""

REFLECTION_PROMPT = """Review these summaries from recent conversations and identify recurring patterns.

Look for:
- Repeated mistakes or corrections
- Topics that come up frequently
- Tendencies in how the assistant responds (hedging, verbosity, etc.)
- Areas where the user frequently pushes back

CRITICAL: Every pattern MUST cite specific evidence from the summaries above. If you cannot point to a concrete example, do not report the pattern.

Format each pattern as:
PATTERN: <description> | EVIDENCE: <specific quote or reference from the summaries> | MITIGATION: <what to do differently> | TOPICS: <relevant topics>

Example output:
PATTERN: The assistant repeats itself when uncertain | EVIDENCE: In the conversation about memory systems, the correction shows the same point was made three times | MITIGATION: State uncertainty once, then move on | TOPICS: communication

If no clear patterns with concrete evidence, respond with exactly: NONE

Recent conversation summaries:
{summary}"""


# ---------------------------------------------------------------------------
# Triage pattern constants and logic moved to Rust (anima_core/src/triage/)
# All triage decisions via start_triage() / resume_*() state machine.
# ---------------------------------------------------------------------------


class EvolutionEngine:
    """Sleep consolidation, belief formation, and self-assessment."""

    def __init__(self, config, inference, embeddings, episodic, semantic, reflective,
                 curiosity=None):
        self.config = config
        self.inference = inference
        self.embeddings = embeddings
        self.episodic = episodic
        self.semantic = semantic
        self.reflective = reflective
        self.curiosity = curiosity

        ref_cfg = config.get("reflection", {})
        self.enabled = ref_cfg.get("enabled", True)
        self.min_turns = ref_cfg.get("min_conversation_turns", 3)
        self.reflection_interval = ref_cfg.get("reflection_interval", 5)
        self.mid_conv_interval = ref_cfg.get("mid_conversation_interval", 5)
        self.similarity_threshold = ref_cfg.get("belief_similarity_threshold", 0.85)
        self.max_beliefs = ref_cfg.get("max_beliefs_per_episode", 5)
        self.max_corrections = ref_cfg.get("max_corrections_per_episode", 3)
        self.analysis_max_tokens = ref_cfg.get("analysis_max_tokens", 500)

        # Dream pass config
        self.dreams_enabled = ref_cfg.get("dreams_enabled", True)
        self.dream_similarity_min = ref_cfg.get("dream_similarity_min", 0.4)
        self.dream_similarity_max = ref_cfg.get("dream_similarity_max", 0.75)
        self.max_dream_pairs = ref_cfg.get("max_dream_pairs", 3)
        self.min_dream_score = ref_cfg.get("min_dream_score", 0.35)
        self.min_beliefs_for_dreams = ref_cfg.get("min_beliefs_for_dreams", 8)
        self.min_cluster_for_triplets = ref_cfg.get("min_cluster_size_for_triplets", 5)
        self.min_reinforced_for_triplets = ref_cfg.get("min_reinforced_edges_for_triplets", 3)
        self.dream_pair_cooldown = ref_cfg.get("dream_pair_cooldown", 5)
        self.max_domain_pair_fraction = ref_cfg.get("max_domain_pair_fraction", 0.40)
        self.frontier_degree_max = ref_cfg.get("frontier_degree_max", 3)

        # Research product: confidence floor and source-type weighting for dreams
        conf_cfg = self.config.get("confidence", {}) if self.config else {}
        self._dream_eligibility_floor = conf_cfg.get("dream_eligibility_floor", 0.0)
        self._web_web_pair_weight = conf_cfg.get("web_web_pair_weight", 1.0)

        # Bridge priority — beliefs flagged for priority sampling after milestone
        self._bridge_priority_ids: set = set()

        # Pair cooldown — rejected pairs excluded for N consolidations
        # Key: frozenset({id_a, id_b}), Value: consolidations remaining
        self._dream_pair_cooldown: dict[frozenset, int] = {}

        # Domain-pair cooldown (C1) — deprioritize domain pairs sampled recently
        # Each entry is a set of frozenset({domain_a, domain_b}) used in one dream pass
        self._domain_pair_history: list[set[frozenset]] = []
        self._domain_pair_cooldown_n = ref_cfg.get("domain_pair_cooldown_n", 2)

        # Triplet cluster cooldown — skip clusters used in recent consolidations
        # Each entry is a frozenset of belief IDs identifying the cluster used
        self._triplet_cluster_history: list[frozenset] = []
        self._triplet_cluster_cooldown_n = ref_cfg.get("triplet_cluster_cooldown_n", 2)

        # Approval queue — initialized lazily when queue_mode is used
        self._queue_db = None
        self._telemetry_db = None

        # Proposal gateway — routes extraction items through triage adapters.
        # Adapter strategies delegate to the existing _triage_* methods,
        # producing identical decisions. This standardizes the entry path
        # without changing triage behavior.
        from core.proposals import ProposalGateway
        from core.triage_adapters import (
            EvolutionBeliefTriageStrategy,
            EvolutionCorrectionTriageStrategy,
            EvolutionLessonTriageStrategy,
        )
        self._governance = self.config.get("_governance", {
            "allow_auto_accept": True,
            "allow_auto_corrections": True,
            "allow_auto_lessons": True,
        })

        # Capability checks — log when optional config is missing
        if not self.config.get("identity_keywords"):
            logger.info("No identity_keywords configured — identity-based recommendation disabled")
        if not self.config.get("extraction", {}).get("specificity_signals"):
            logger.info("No specificity_signals configured — using generic detection only")

        self.gateway = ProposalGateway(config=self.config, inference_engine=self.inference)
        self.gateway.register_strategy("belief", EvolutionBeliefTriageStrategy(self))
        self.gateway.register_strategy("correction", EvolutionCorrectionTriageStrategy(self, self._governance))
        self.gateway.register_strategy("lesson", EvolutionLessonTriageStrategy(self, self._governance))

    # -----------------------------------------------------------------------
    # Approval queue — stores items for web-based review
    # -----------------------------------------------------------------------

    def _get_queue_db(self):
        """Get the approval queue DB connection.

        Uses semantic.db_path to open a SEPARATE connection to the same file.
        Same DB file = plugin isolation. Separate connection = no lock contention.

        NEVER resolve paths from __file__ or config — those are relative to
        the wrong directory in blade architecture.
        """
        if self._queue_db is not None:
            return self._queue_db
        # Get the DB path from semantic — guaranteed correct per plugin
        db_path = getattr(self.semantic, 'db_path', None)
        if not db_path:
            raise ValueError(
                "semantic.db_path not set — cannot create queue DB. "
                "Plugin isolation requires injected paths, not self-resolved."
            )
        self._queue_db = sqlite3.connect(db_path, check_same_thread=False, timeout=30.0)
        self._queue_db.row_factory = sqlite3.Row
        self._queue_db.execute("PRAGMA journal_mode=WAL")
        self._queue_db.executescript("""
            CREATE TABLE IF NOT EXISTS approval_queue (
                id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                data TEXT NOT NULL,
                recommended INTEGER DEFAULT 0,
                label TEXT,
                episode_id TEXT,
                status TEXT DEFAULT 'pending',
                created_at TEXT NOT NULL,
                resolved_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_approval_queue_status
                ON approval_queue(status);

            CREATE TABLE IF NOT EXISTS rejection_history (
                content_hash TEXT NOT NULL,
                category TEXT NOT NULL,
                rejection_count INTEGER DEFAULT 1,
                last_rejected TEXT NOT NULL,
                PRIMARY KEY (content_hash, category)
            );
        """)
        self._queue_db.commit()
        return self._queue_db

    def _get_telemetry_db(self):
        """Get or create the telemetry DB connection (telemetry.db)."""
        if self._telemetry_db is not None:
            return self._telemetry_db
        data_dir = os.environ.get("ANIMA_DATA_DIR")
        if data_dir:
            db_path = os.path.join(data_dir, "telemetry.db")
        else:
            base_dir = os.path.join(os.path.dirname(__file__), "..")
            db_path = os.path.normpath(
                os.path.join(base_dir, "data", "telemetry.db")
            )
        self._telemetry_db = sqlite3.connect(db_path, check_same_thread=False)
        self._telemetry_db.row_factory = sqlite3.Row
        self._telemetry_db.executescript("""
            CREATE TABLE IF NOT EXISTS approval_accuracy (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                category TEXT NOT NULL,
                recommended INTEGER NOT NULL,
                actual INTEGER NOT NULL,
                agreed INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS rejected_beliefs_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                category TEXT NOT NULL,
                statement TEXT NOT NULL,
                confidence TEXT,
                source TEXT,
                rejection_reason TEXT NOT NULL,
                episode_id TEXT,
                embedding TEXT
            );
        """)
        # Migration: add embedding column if table predates it
        try:
            self._telemetry_db.execute(
                "ALTER TABLE rejected_beliefs_log ADD COLUMN embedding TEXT"
            )
        except sqlite3.OperationalError:
            pass  # column already exists
        self._telemetry_db.commit()
        return self._telemetry_db

    def _log_approval_accuracy(self, category, recommended, approved):
        """Record whether the operator's decision agreed with auto-recommendation."""
        try:
            db = self._get_telemetry_db()
            entry_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc).isoformat()
            rec_int = 1 if recommended else 0
            actual_int = 1 if approved else 0
            agreed = 1 if rec_int == actual_int else 0
            db.execute(
                "INSERT INTO approval_accuracy "
                "(id, timestamp, category, recommended, actual, agreed) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (entry_id, now, category, rec_int, actual_int, agreed),
            )
            db.commit()
            if not agreed:
                logger.info(
                    f"Approval override: {category} recommended={'accept' if recommended else 'reject'}, "
                    f"actual={'accept' if approved else 'reject'}"
                )
        except Exception as e:
            logger.warning(f"Approval accuracy logging failed: {e}")

    # Identity-adjacent rejection reasons that trigger shadow logging
    _SHADOW_LOG_REASONS = {
        "generic self-assessment",
        "psychoanalysis/blacklisted pattern",
        "contains agency claims",
        "completed roadmap reference",
    }

    def _log_rejected_belief(self, category, data, reason, episode_id=None):
        """Shadow-log an identity-adjacent rejected belief to telemetry.db.

        No behavior change. Read-only record for longitudinal analysis of
        whether rejected identity/agency beliefs drift toward observation
        or remain claims. Embedding stored alongside text to avoid recomputation.
        """
        try:
            db = self._get_telemetry_db()
            statement = self._get_triage_text(category, data)

            # Compute embedding — MiniLM is <10ms per sentence, negligible
            embedding_json = None
            try:
                vec = self.embeddings.embed(statement)
                embedding_json = json.dumps(vec.tolist())
            except Exception:
                pass  # store without embedding if model unavailable

            db.execute(
                "INSERT INTO rejected_beliefs_log "
                "(id, timestamp, category, statement, confidence, source, "
                "rejection_reason, episode_id, embedding) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    str(uuid.uuid4()),
                    datetime.now(timezone.utc).isoformat(),
                    category,
                    statement,
                    data.get("confidence", ""),
                    data.get("source", ""),
                    reason,
                    episode_id,
                    embedding_json,
                ),
            )
            db.commit()
        except Exception as e:
            logger.debug(f"Shadow log write failed: {e}")

    def queue_item(self, category, data, recommended, label, episode_id=None):
        """Add an item to the approval queue for web review.

        Queue-level dedup: checks embedding similarity against existing pending
        items. Rejects if >0.75 similar to anything already in queue.
        Prevents paraphrase accumulation (gut microbiome "smart filter" x15).
        """
        db = self._get_queue_db()

        # Queue-level dedup — extract statement text for comparison
        stmt = data.get("statement") or data.get("inference") or ""
        if stmt and hasattr(self, 'embeddings') and self.embeddings:
            try:
                new_emb = self.embeddings.embed(stmt)
                # Check against pending items in same category
                pending = db.execute(
                    "SELECT data FROM approval_queue WHERE status = 'pending' AND category = ?",
                    (category,),
                ).fetchall()
                for row in pending:
                    existing_data = json.loads(row["data"])
                    existing_stmt = existing_data.get("statement") or existing_data.get("inference") or ""
                    if existing_stmt:
                        existing_emb = self.embeddings.embed(existing_stmt)
                        sim = float(sum(a * b for a, b in zip(new_emb, existing_emb)) /
                              (sum(a**2 for a in new_emb)**0.5 * sum(b**2 for b in existing_emb)**0.5 + 1e-9))
                        if sim > 0.75:
                            logger.info(f"Queue dedup: rejected {category} (sim={sim:.2f}): {stmt[:60]}")
                            return None
            except Exception as e:
                logger.debug(f"Queue dedup check failed: {e}")

        item_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        db.execute(
            "INSERT INTO approval_queue "
            "(id, category, data, recommended, label, episode_id, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)",
            (item_id, category, json.dumps(data), 1 if recommended else 0,
             label, episode_id, now),
        )
        db.commit()
        logger.info(f"Queued {category} for approval: {label[:60]}")
        return item_id

    def get_pending_approvals(self):
        """Get all pending approval items."""
        db = self._get_queue_db()
        rows = db.execute(
            "SELECT * FROM approval_queue WHERE status='pending' "
            "ORDER BY created_at"
        ).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["data"] = json.loads(d["data"])
            d["recommended"] = bool(d["recommended"])
            result.append(d)
        return result

    def resolve_approval(self, item_id, approved):
        """Approve or reject a queued item. Processes if approved."""
        db = self._get_queue_db()
        row = db.execute(
            "SELECT * FROM approval_queue WHERE id=? AND status='pending'",
            (item_id,),
        ).fetchone()
        if not row:
            return False

        now = datetime.now(timezone.utc).isoformat()
        new_status = "approved" if approved else "rejected"
        db.execute(
            "UPDATE approval_queue SET status=?, resolved_at=? WHERE id=?",
            (new_status, now, item_id),
        )
        db.commit()

        # Log dream triage decisions to triage log for health panel metrics
        if row["category"] in ("dream", "triplet_dream"):
            try:
                data = json.loads(row["data"]) if isinstance(row["data"], str) else row["data"]
                stmt = data.get("inference") or data.get("statement") or ""
                decision = "accept" if approved else "reject"
                self.semantic.db_conn.execute(
                    "INSERT INTO dms_triage_log "
                    "(category, statement, decision, reason, confidence, "
                    "source_type, document_sha, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (row["category"], stmt[:500], decision,
                     "dream_approval", str(data.get("similarity", "")),
                     "synthesis", "", now),
                )
                self.semantic.db_conn.commit()
            except Exception:
                pass

        # Track accuracy: did operator agree with auto-recommendation?
        self._log_approval_accuracy(
            row["category"], bool(row["recommended"]), approved
        )

        if approved:
            data = json.loads(row["data"])
            category = row["category"]
            episode_id = row["episode_id"]
            self._process_approved_item(category, data, episode_id)

        logger.info(f"Approval {item_id[:8]}: {new_status} ({row['category']})")
        return True

    def resolve_recommended(self):
        """Accept all recommended items, reject the rest. Returns counts."""
        pending = self.get_pending_approvals()
        accepted = 0
        rejected = 0
        for item in pending:
            if item["recommended"]:
                self.resolve_approval(item["id"], approved=True)
                accepted += 1
            else:
                self.resolve_approval(item["id"], approved=False)
                rejected += 1
        return accepted, rejected

    def _process_approved_item(self, category, data, episode_id):
        """Process a single approved item by category."""
        if category == "belief":
            self._process_beliefs([data], episode_id)
        elif category == "correction":
            self._process_corrections([data], episode_id)
        elif category == "pushback":
            self._process_pushbacks([data], episode_id)
        elif category == "lesson":
            self._process_lessons([data], episode_id)
        elif category == "dream":
            self._process_approved_dream(data, episode_id)
        elif category == "triplet_dream":
            self._process_approved_triplet(data, episode_id)
        elif category == "reflection":
            self._process_approved_reflection(data, episode_id)

    def _process_approved_dream(self, data, episode_id=None, semantic_override=None):
        """Store an approved dream connection as a new belief + link.

        Args:
            data: dream data dict (inference, belief_a/b IDs and statements, similarity)
            episode_id: optional source episode
            semantic_override: optional DreamTransaction — if provided, all graph
                writes go through the transaction buffer instead of self.semantic.
                Reads always go to self.semantic (live DB).
        """
        sem = semantic_override or self.semantic
        is_transactional = semantic_override is not None

        # Inherit topics and entities from both parent beliefs (reads from live DB)
        merged_topics = []
        merged_entities = []
        for parent_id in (data["belief_a_id"], data["belief_b_id"]):
            row = self.semantic.db_conn.execute(
                "SELECT topics, entities FROM beliefs WHERE id = ?",
                (parent_id,),
            ).fetchone()
            if row:
                for col, target in ((row["topics"], merged_topics),
                                    (row["entities"], merged_entities)):
                    if col:
                        try:
                            vals = json.loads(col)
                            if isinstance(vals, list):
                                target.extend(vals)
                        except (json.JSONDecodeError, TypeError):
                            pass
        merged_topics = list(dict.fromkeys(merged_topics))
        merged_entities = list(dict.fromkeys(merged_entities))

        # Compute abstraction depth and scale starting confidence
        # Use similarity score as quality signal — higher similarity = higher confidence
        depth_a = self.semantic.get_belief_depth(data["belief_a_id"])
        depth_b = self.semantic.get_belief_depth(data["belief_b_id"])
        depth = max(depth_a, depth_b) + 1
        similarity = data.get("similarity", 0.5)
        # Base: similarity scaled to 0.35-0.70 range, with depth penalty
        starting_confidence = max(0.3, min(0.70, similarity * 0.8) - (depth - 1) * 0.1)

        belief_id = sem.add_belief(
            statement=data["inference"],
            confidence=round(starting_confidence, 3),
            source_episode=episode_id,
            supporting_evidence=[
                f"Dream link: {data['belief_a_statement'][:80]} + "
                f"{data['belief_b_statement'][:80]}"
            ],
            topics=merged_topics or None,
            entities=merged_entities or None,
            source="exploration",
            source_type="synthesis",
            abstraction_depth=depth,
            parent_a=data["belief_a_id"],
            parent_b=data["belief_b_id"],
            generation_type="dream",
        )
        if belief_id is None:
            return None  # Provenance invariant rejected this belief
        sim = data.get("similarity")
        # Reinforce existing a↔b edge if present (inactive or active), else create new
        result = sem.reinforce_or_reactivate_link(
            data["belief_a_id"], data["belief_b_id"], sim,
        )
        if result:
            reinforced, hit_cap = result
            logger.info(f"Dream reinforced existing edge {reinforced[:8]}")
            if hit_cap:
                logger.info(
                    f"Edge weight cap reached: edge={reinforced[:8]} "
                    f"nodes={data['belief_a_id'][:8]}↔{data['belief_b_id'][:8]} "
                    f"domains={data.get('_domain_a', '?')}↔{data.get('_domain_b', '?')}"
                )
        else:
            sem.add_belief_link(
                belief_a=data["belief_a_id"],
                belief_b=data["belief_b_id"],
                inference=data["inference"],
                similarity=sim,
            )
        # Connect inference belief back to both source beliefs
        sem.add_belief_link(
            belief_a=belief_id,
            belief_b=data["belief_a_id"],
            inference=data["inference"],
            similarity=sim,
        )
        sem.add_belief_link(
            belief_a=belief_id,
            belief_b=data["belief_b_id"],
            inference=data["inference"],
            similarity=sim,
        )
        logger.info(f"Dream processed: belief {belief_id[:8]}")

        if is_transactional:
            # Defer tag inheritance and dormant checks to post-commit
            _belief_id = belief_id
            _parent_a = data["belief_a_id"]
            _parent_b = data["belief_b_id"]
            sem.defer_callback(lambda: self._defer_dream_post_commit(
                _belief_id, _parent_a, _parent_b
            ))
        else:
            # Direct mode — run immediately
            self._defer_dream_post_commit(
                belief_id, data["belief_a_id"], data["belief_b_id"]
            )

    def _defer_dream_post_commit(self, belief_id, parent_a_id, parent_b_id):
        """Post-commit work for a dream: tag inheritance + dormant adjacency."""
        # Inherit tags from parent beliefs
        try:
            from core.tags import TagRegistry
            tag_reg = getattr(self, '_tag_registry', None)
            if tag_reg is None:
                tag_reg = TagRegistry(self.semantic.db_conn)
                self._tag_registry = tag_reg
            tag_reg.inherit_for_synthesis(belief_id, parent_a_id, parent_b_id)
        except Exception as e:
            logger.debug(f"Tag inheritance failed: {e}")

        # D1: Check if new dream belief/edges are adjacent to dormant beliefs
        self.semantic.check_dormant_adjacency(
            belief_id, embedder=self.embeddings, trigger_type="new_belief",
        )
        for parent_id in (parent_a_id, parent_b_id):
            self.semantic.check_dormant_adjacency(
                parent_id, embedder=self.embeddings, trigger_type="new_edge",
            )

    def _process_approved_triplet(self, data, episode_id=None, semantic_override=None):
        """Store an approved triplet dream — synthesis belief + 3 parent links.

        Args:
            data: triplet dream data dict
            episode_id: optional source episode
            semantic_override: optional DreamTransaction for buffered writes
        """
        sem = semantic_override or self.semantic
        is_transactional = semantic_override is not None
        parent_ids = [data["belief_a_id"], data["belief_b_id"], data["belief_c_id"]]

        # Inherit topics and entities from all three parents (reads from live DB)
        merged_topics = []
        merged_entities = []
        for parent_id in parent_ids:
            row = self.semantic.db_conn.execute(
                "SELECT topics, entities FROM beliefs WHERE id = ?",
                (parent_id,),
            ).fetchone()
            if row:
                for col, target in ((row["topics"], merged_topics),
                                    (row["entities"], merged_entities)):
                    if col:
                        try:
                            vals = json.loads(col)
                            if isinstance(vals, list):
                                target.extend(vals)
                        except (json.JSONDecodeError, TypeError):
                            pass
        merged_topics = list(dict.fromkeys(merged_topics))
        merged_entities = list(dict.fromkeys(merged_entities))

        # Compute abstraction depth from deepest parent
        depth = max(self.semantic.get_belief_depth(pid) for pid in parent_ids) + 1

        # Triplet depth cap: reject if output depth >= 4.
        # Triplets should densify clusters, not build abstraction towers.
        if depth >= 4:
            logger.info(
                f"Triplet depth cap: rejected output depth {depth} "
                f"(parent depths: {[self.semantic.get_belief_depth(pid) for pid in parent_ids]})"
            )
            return None

        similarity = data.get("avg_similarity") or data.get("similarity", 0.5)
        starting_confidence = max(0.3, min(0.70, similarity * 0.8) - (depth - 1) * 0.1)

        # Select two highest-depth parents for parent_a/parent_b columns.
        # Tiebreaker: alphabetical by ID (deterministic).
        # All three parents are recorded via edge links below.
        parents_by_depth = sorted(
            parent_ids,
            key=lambda pid: (self.semantic.get_belief_depth(pid), pid),
            reverse=True,
        )
        triplet_parent_a = parents_by_depth[0]
        triplet_parent_b = parents_by_depth[1]

        belief_id = sem.add_belief(
            statement=data["inference"],
            confidence=starting_confidence,
            source_episode=episode_id,
            supporting_evidence=[
                f"Triplet link: {data['belief_a_statement'][:50]} + "
                f"{data['belief_b_statement'][:50]} + "
                f"{data['belief_c_statement'][:50]}"
            ],
            topics=merged_topics or None,
            entities=merged_entities or None,
            source="exploration",
            source_type="synthesis",
            abstraction_depth=depth,
            parent_a=triplet_parent_a,
            parent_b=triplet_parent_b,
            generation_type="triplet",
        )
        if belief_id is None:
            return None  # Provenance invariant rejected this belief
        sim = data.get("avg_similarity")
        # Connect inference to each parent (reinforce or create)
        for pid in parent_ids:
            sem.add_belief_link(
                belief_a=belief_id,
                belief_b=pid,
                inference=data["inference"],
                similarity=sim,
            )
        # Reinforce or create edges between all parent pairs

        for a_id, b_id in combinations(parent_ids, 2):
            result = sem.reinforce_or_reactivate_link(a_id, b_id, sim)
            if result:
                reinforced, hit_cap = result
                if hit_cap:
                    logger.info(
                        f"Edge weight cap reached (triplet): edge={reinforced[:8]} "
                        f"nodes={a_id[:8]}↔{b_id[:8]}"
                    )
            if not result:
                sem.add_belief_link(
                    belief_a=a_id, belief_b=b_id,
                    inference=data["inference"],
                    similarity=sim,
                )
        logger.info(f"Triplet dream processed: belief {belief_id[:8]}")

        if is_transactional:
            _belief_id = belief_id
            _parent_a = triplet_parent_a
            _parent_b = triplet_parent_b
            _parent_ids = list(parent_ids)
            sem.defer_callback(lambda: self._defer_triplet_post_commit(
                _belief_id, _parent_a, _parent_b, _parent_ids
            ))
        else:
            self._defer_triplet_post_commit(
                belief_id, triplet_parent_a, triplet_parent_b, parent_ids
            )

    def _defer_triplet_post_commit(self, belief_id, parent_a, parent_b, parent_ids):
        """Post-commit work for a triplet: tag inheritance + dormant adjacency."""
        try:
            from core.tags import TagRegistry
            tag_reg = getattr(self, '_tag_registry', None)
            if tag_reg is None:
                tag_reg = TagRegistry(self.semantic.db_conn)
                self._tag_registry = tag_reg
            tag_reg.inherit_for_synthesis(belief_id, parent_a, parent_b)
        except Exception as e:
            logger.debug(f"Triplet tag inheritance failed: {e}")

        # D1: Check if new triplet belief/edges are adjacent to dormant beliefs
        self.semantic.check_dormant_adjacency(
            belief_id, embedder=self.embeddings, trigger_type="new_belief",
        )
        for pid in parent_ids:
            self.semantic.check_dormant_adjacency(
                pid, embedder=self.embeddings, trigger_type="new_edge",
            )

    def _triplet_curiosity_gap(self, gap_question, gap_context, topic_tags, dom_list):
        """Post-commit: generate curiosity gap for cross-domain triplet dream."""
        if self.curiosity:
            self.curiosity.add_question(
                question=gap_question,
                question_type="knowledge_gap",
                context=gap_context,
                topic_tags=topic_tags,
                priority="high",
                embedder=self.embeddings,
            )
            logger.info(
                f"Dream-to-curiosity: generated bridging gap "
                f"({' x '.join(dom_list)}): '{gap_question[:60]}'"
            )

    def _process_approved_reflection(self, data, episode_id):
        """Store an approved reflection pattern as a self-observation."""
        self.reflective.add_observation(
            pattern=data["pattern"],
            identified_by="reflection_engine",
            mitigation=data.get("mitigation"),
            topics=data.get("topics", []),
            status="monitoring",
        )
        logger.info(f"Reflection pattern processed: {data['pattern'][:60]}")

    # -----------------------------------------------------------------------
    # Main entry points
    # -----------------------------------------------------------------------

    def _apply_belief_aging(self):
        """Decay belief confidences scaled by evidence count, floor 0.5.

        Base rate: 0.01/day. Scaled: 0.01 / max(1, num_sources) per day.
        A belief with 5 sources decays at 0.002/day instead of 0.01/day.
        """
        try:
            # Find last sleep event from state_log (telemetry.db)
            # Fall back to 1 day if no prior sleep
            last_sleep = None
            try:
                import os as _os
                base = _os.path.join(_os.path.dirname(__file__), "..")
                tel_path = _os.path.normpath(_os.path.join(base, "data", "telemetry.db"))
                tel_conn = sqlite3.connect(tel_path)
                tel_conn.row_factory = sqlite3.Row
                row = tel_conn.execute(
                    "SELECT timestamp FROM state_log WHERE trigger = 'sleep' "
                    "ORDER BY timestamp DESC LIMIT 1"
                ).fetchone()
                if row:
                    last_sleep = datetime.fromisoformat(row["timestamp"])
                tel_conn.close()
            except Exception:
                pass

            if last_sleep is None:
                days_since = 1.0
            else:
                now = datetime.now(timezone.utc)
                days_since = max((now - last_sleep).total_seconds() / 86400.0, 0)

            if days_since < 0.01:
                return  # just slept, no decay needed

            # Build source count map from belief_sources table
            source_counts = {}
            try:
                db = self.semantic.db_conn
                rows = db.execute(
                    "SELECT belief_id, COUNT(*) as cnt FROM belief_sources GROUP BY belief_id"
                ).fetchall()
                source_counts = {r["belief_id"]: r["cnt"] for r in rows}
            except Exception:
                pass

            beliefs = self.semantic.search_beliefs()
            aged = 0
            for b in beliefs:
                if b.get("operator_anchored", 0):
                    continue
                num_sources = source_counts.get(b["id"], 1)
                decay = (0.01 / max(1, num_sources)) * days_since
                conf = b["confidence"]
                new_conf = max(0.5, conf - decay)
                if new_conf < conf:
                    self.semantic.update_belief(
                        b["id"],
                        new_confidence=round(new_conf, 4),
                        reason=f"Belief aging: -{decay:.4f} over {days_since:.1f} days ({num_sources} sources)",
                        source="decay",
                    )
                    aged += 1

            if aged > 0:
                logger.info(f"Belief aging: decayed {aged} beliefs (evidence-scaled)")
        except Exception as e:
            logger.warning(f"Belief aging failed: {e}")

    def _quarantine_dormant_isolates(self):
        """Mark degree-0 beliefs older than N days as dormant.

        Dormant beliefs are excluded from dream sampling (via search_beliefs
        deprecated filter) but remain in the DB and can be reactivated if
        a future dream or exploration connects them.

        Threshold: beliefs with zero links, not already dormant/deprecated,
        created more than dormant_age_days ago (default 5 days ≈ 5 soaks).
        """
        try:
            dormant_age_days = self.config.get("dreams", {}).get("dormant_age_days", 5)
            db = self.semantic.db_conn
            # Find degree-0 beliefs older than threshold
            result = db.execute(
                """
                UPDATE beliefs SET is_dormant = 1
                WHERE COALESCE(is_dormant, 0) = 0
                  AND COALESCE(deprecated, 0) = 0
                  AND COALESCE(operator_anchored, 0) = 0
                  AND id NOT IN (
                    SELECT DISTINCT belief_a FROM belief_links
                    UNION
                    SELECT DISTINCT belief_b FROM belief_links
                  )
                  AND julianday('now') - julianday(created_at) > ?
                """,
                (dormant_age_days,),
            )
            count = result.rowcount
            db.commit()
            if count > 0:
                logger.info(f"Dormant quarantine: {count} isolates marked dormant (>{dormant_age_days} days, degree=0)")
        except Exception as e:
            logger.warning(f"Dormant quarantine failed: {e}")

    def _run_triggered_retriage(self):
        """Trigger-based re-triage: re-evaluate old beliefs when new similar evidence appears.

        Eligibility: belief age > retriage_min_age_hours (default 4h ≈ 50 cycles at 5min)
        Trigger: a new belief (created in last cycle) has >0.70 similarity to an old belief.
        Batch: max 10 per consolidation to prevent churn.
        Outcomes: reaffirm (confidence updated) or downgrade (flagged for operator). Never auto-revoke.
        """
        try:
            retriage_cfg = self.config.get("retriage", {})
            min_age_hours = retriage_cfg.get("min_age_hours", 4)
            similarity_threshold = retriage_cfg.get("similarity_threshold", 0.70)
            batch_size = retriage_cfg.get("batch_size", 10)

            if not hasattr(self, 'embeddings') or not self.embeddings:
                return

            db = self.semantic.db_conn

            # Find beliefs created in the last consolidation interval (recent arrivals)
            interval_minutes = self.config.get("consolidation", {}).get("interval_minutes", 5)
            new_beliefs = db.execute(
                """SELECT id, statement FROM beliefs
                   WHERE COALESCE(deprecated, 0) = 0
                     AND COALESCE(is_dormant, 0) = 0
                     AND julianday('now') - julianday(created_at) < ?""",
                (interval_minutes / 1440.0,),  # convert minutes to days
            ).fetchall()

            if not new_beliefs:
                return

            # Find old beliefs eligible for re-triage
            old_beliefs = db.execute(
                """SELECT id, statement, confidence, source_type, operator_anchored,
                          extraction_context, epistemic_class
                   FROM beliefs
                   WHERE COALESCE(deprecated, 0) = 0
                     AND COALESCE(is_dormant, 0) = 0
                     AND COALESCE(operator_anchored, 0) = 0
                     AND (julianday('now') - julianday(created_at)) * 24 > ?""",
                (min_age_hours,),
            ).fetchall()

            if not old_beliefs:
                return

            # Embed new beliefs
            new_embs = []
            for nb in new_beliefs:
                try:
                    emb = self.embeddings.embed(nb["statement"])
                    new_embs.append((emb, nb["statement"]))
                except Exception:
                    continue

            if not new_embs:
                return

            # Check each old belief against new arrivals
            import numpy as np
            triggered = []
            for ob in old_beliefs:
                try:
                    ob_emb = self.embeddings.embed(ob["statement"])
                    for new_emb, new_stmt in new_embs:
                        norm = np.linalg.norm(ob_emb) * np.linalg.norm(new_emb)
                        sim = float(np.dot(ob_emb, new_emb) / norm) if norm > 0 else 0.0
                        if sim >= similarity_threshold:
                            triggered.append((dict(ob), sim, new_stmt))
                            break  # One trigger is enough
                except Exception:
                    continue

            if not triggered:
                return

            # Sort by similarity (highest first), batch limit
            triggered.sort(key=lambda x: x[1], reverse=True)
            triggered = triggered[:batch_size]

            reaffirmed = 0
            downgraded = 0
            for ob_dict, sim, trigger_stmt in triggered:
                data = {
                    "statement": ob_dict["statement"],
                    "source": ob_dict.get("source_type", "unknown"),
                    "confidence": ob_dict.get("confidence", 0.5),
                    "extraction_context": ob_dict.get("extraction_context", ""),
                }
                still_recommended = self._recommend("belief", data)

                if still_recommended:
                    reaffirmed += 1
                else:
                    # Downgrade: reduce confidence, log for operator review
                    new_conf = max(0.3, ob_dict.get("confidence", 0.5) - 0.1)
                    db.execute(
                        "UPDATE beliefs SET confidence = ?, belief_status = 'retriage_flagged' "
                        "WHERE id = ?",
                        (round(new_conf, 3), ob_dict["id"]),
                    )
                    downgraded += 1
                    logger.info(
                        f"Re-triage downgrade: {ob_dict['statement'][:60]}... "
                        f"(trigger sim={sim:.2f}, new_conf={new_conf:.2f})"
                    )

            db.commit()
            if reaffirmed or downgraded:
                logger.info(
                    f"Triggered re-triage: {len(triggered)} checked, "
                    f"{reaffirmed} reaffirmed, {downgraded} downgraded"
                )

        except Exception as e:
            logger.warning(f"Triggered re-triage failed: {e}")

    def consolidate(self, episode_id, interactive=True, queue_mode=False,
                     extraction_context="operator"):
        """Full sleep consolidation. Called on sleep (session end) or missed sleep recovery.

        interactive=True: terminal approval prompts (CLI mode)
        queue_mode=True: store items in approval queue (web mode)
        Both False: silently skip approval items (crash recovery fallback)
        extraction_context: 'operator' (genuine interaction) or 'exploration' (soak/freetime)
        """
        # Reset per-cycle degradation tracker
        self._cycle_degraded = {}

        if not self.enabled:
            return

        # Apply belief aging before consolidation
        self._apply_belief_aging()

        # Quarantine degree-0 isolates older than dormant_soak_threshold soaks
        self._quarantine_dormant_isolates()

        # Trigger-based re-triage: check if new beliefs challenge old ones
        self._run_triggered_retriage()

        # Apply gap decay — increment sleep cycles, demote/dissolve stale gaps
        if self.curiosity:
            decay_cycles = self.config.get("curiosity", {}).get("gap_decay_cycles", 5)
            dissolve_cycles = self.config.get("curiosity", {}).get("gap_dissolve_cycles", 10)
            self.curiosity.apply_sleep_decay(decay_cycles, dissolve_cycles)

        turns = self.episodic.get_episode_turns(episode_id)

        # Mark consolidated early — dream check needs accurate count
        self.episodic.mark_episode_consolidated(episode_id)

        # Dreams operate on the full belief graph, not the current episode.
        # Run the dream check before the min_turns guard so short exploration
        # episodes (1 turn) don't block dream passes during soak.
        # Reflection gate — track episodes since last reflection, not global DB count.
        # Using a simple counter avoids fragility from DB total vs process state mismatch.
        if not hasattr(self, "_episodes_since_reflection"):
            self._episodes_since_reflection = self.reflection_interval - 1
        self._episodes_since_reflection += 1
        logger.info(
            f"Reflection gate: episodes_since_reflection={self._episodes_since_reflection} "
            f"interval={self.reflection_interval}"
        )
        if self._episodes_since_reflection >= self.reflection_interval:
            self._episodes_since_reflection = 0
            self._run_reflection(queue_mode=queue_mode)

            # Dream pass — cross-episode belief linking.
            # Runs on reflection interval, independent of reflection output.
            # Dreams are expensive and non-urgent — skip under system pressure.
            from datetime import datetime, timezone
            dream_start = datetime.now(timezone.utc).isoformat()
            _skip_dreams = False
            if self.dreams_enabled and hasattr(self.inference, '_system_pressure'):
                pressure = self.inference._system_pressure()
                if pressure > 0.7:
                    logger.info(f"Deferring dreams — system pressure {pressure:.2f} > 0.7")
                    _skip_dreams = True
            if self.dreams_enabled and not _skip_dreams:
                self._run_dreams(queue_mode=queue_mode)
            self._dedup_edges(since_timestamp=dream_start)

        if len(turns) < self.min_turns:
            logger.info(
                f"Episode {episode_id[:8]} has {len(turns)} turns, "
                f"below minimum {self.min_turns}. Skipping belief extraction."
            )
            return

        print("\n  Consolidating...", flush=True)
        conversation_text = self._prepare_conversation_text(turns)
        logger.info(
            f"Consolidating episode {episode_id[:8]}: {len(turns)} turns, "
            f"{len(conversation_text)} chars"
        )
        if len(conversation_text) < 100:
            logger.warning(
                f"Conversation text suspiciously short: {repr(conversation_text)}"
            )

        # Phase-isolated passes — each wrapped independently so one
        # timeout/failure doesn't kill the entire consolidation cycle.

        # Pass 1: Summary
        summary_data = None
        try:
            summary_data = self._run_pass_summary(conversation_text)
        except Exception as e:
            logger.warning(f"Pass 1 (summary) failed for {episode_id[:8]}: {e}")

        if summary_data:
            self.episodic.update_episode_metadata(
                episode_id,
                summary=summary_data.get("summary"),
                topics=summary_data.get("topics", []),
                importance=summary_data.get("importance", 0.5),
            )
            logger.info(f"Pass 1 (summary): stored for {episode_id[:8]}")

            summary_text = summary_data.get("summary", "")
            topics_list = summary_data.get("topics", [])
            importance_val = summary_data.get("importance", 0.5)
            if summary_text:
                print(f"  Summary: {summary_text}", flush=True)
                if topics_list:
                    print(f"  Topics: {', '.join(topics_list)}", flush=True)
                if importance_val >= 0.7:
                    print(f"  Importance: {importance_val:.1f}", flush=True)

        # Quality gate
        quality = self._assess_conversation_quality(turns)
        logger.info(f"Conversation quality: {quality:.2f} for {episode_id[:8]}")

        beliefs = []
        corrections = []
        pushbacks = []
        lessons = []

        if quality >= 0.4:
            # Pass 2: Beliefs
            try:
                beliefs = self._run_pass_beliefs(conversation_text)
                logger.info(f"Pass 2 (beliefs): {len(beliefs)} extracted from {episode_id[:8]}")
            except Exception as e:
                logger.warning(f"Pass 2 (beliefs) failed for {episode_id[:8]}: {e}")

            # Pass 3: Corrections
            try:
                corrections, pushbacks, lessons = self._run_pass_corrections(conversation_text)
                logger.info(
                    f"Pass 3 (corrections): {len(corrections)} corrections, "
                    f"{len(pushbacks)} pushbacks, {len(lessons)} lessons "
                    f"from {episode_id[:8]}"
                )
            except Exception as e:
                logger.warning(f"Pass 3 (corrections) failed for {episode_id[:8]}: {e}")

            # Pass 4: Questions
            try:
                questions = self._run_pass_questions(conversation_text)
                if questions:
                    episode_topics = summary_data.get("topics", []) if summary_data else []
                    self._process_questions(questions, episode_id, episode_topics)
                    logger.info(f"Pass 4 (questions): {len(questions)} gaps extracted from {episode_id[:8]}")
            except Exception as e:
                logger.warning(f"Pass 4 (questions) failed for {episode_id[:8]}: {e}")
        else:
            print(
                "  Conversation was too incoherent for belief extraction. "
                "Summary stored, beliefs skipped.",
                flush=True,
            )
            logger.info(
                f"Quality gate blocked belief extraction for {episode_id[:8]} "
                f"(quality={quality:.2f})"
            )

        # Approval flow
        all_items = beliefs or corrections or pushbacks or lessons
        if queue_mode and all_items:
            self._queue_consolidation_items(
                beliefs, corrections, pushbacks, lessons, episode_id,
                extraction_context=extraction_context,
            )
        elif interactive and all_items:
            a_beliefs, a_corrections, a_pushbacks, a_lessons = self._propose_changes(
                beliefs, corrections, pushbacks, lessons
            )
            # Process approved changes
            if a_beliefs:
                self._process_beliefs(a_beliefs, episode_id)
            if a_corrections:
                self._process_corrections(a_corrections, episode_id)
            if a_pushbacks:
                self._process_pushbacks(a_pushbacks, episode_id)
            if a_lessons:
                self._process_lessons(a_lessons, episode_id)

    def nap(self, episode_id):
        """Mid-session context refresh. Runs summary pass only, stores in key_insights.

        Unlike full consolidation, the episode continues after a nap. Only the
        summary is captured — no beliefs, corrections, or approval flow.
        Returns the summary dict, or None on failure.
        """
        try:
            turns = self.episodic.get_episode_turns(episode_id)
            if not turns:
                return None

            conversation_text = self._prepare_conversation_text(turns)
            summary_data = self._run_pass_summary(conversation_text)

            if summary_data and summary_data.get("summary"):
                # Store nap summary in key_insights with [NAP] prefix
                turn_count = len(turns)
                nap_entry = f"[NAP {turn_count}t] {summary_data['summary']}"
                existing = self._get_episode_insights(episode_id)
                existing.append(nap_entry)
                self.episodic.update_episode_metadata(
                    episode_id, key_insights=existing
                )
                logger.info(f"Nap summary stored for {episode_id[:8]}: {nap_entry[:80]}")

            return summary_data
        except Exception as e:
            logger.warning(f"Nap failed: {e}")
            return None

    def micro_consolidate(self, episode_id, turn_count):
        """Micro-consolidation during waking life. No approval, just note-taking."""
        if not self.enabled:
            return
        if turn_count == 0 or turn_count % self.mid_conv_interval != 0:
            return

        try:
            turns = self.episodic.get_episode_turns(episode_id)
            recent = turns[-10:] if len(turns) > 10 else turns
            conversation_text = self._format_turns(recent)

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are analyzing a conversation excerpt for notable events. "
                        "Be concise and precise."
                    ),
                },
                {
                    "role": "user",
                    "content": self._no_think(MID_CONVERSATION_PROMPT.format(
                        conversation=conversation_text
                    )),
                },
            ]

            response = self.inference.generate_with_messages(
                messages, max_tokens=200, temperature=0.1, task="reinforcement",
            )
            response = response.strip()

            if response.upper() == "NONE":
                return

            # Parse insights
            insights = []
            for line in response.split("\n"):
                line = line.strip().lstrip("- ").strip()
                if line and line.upper() != "NONE":
                    insights.append(line)

            if insights:
                existing = self._get_episode_insights(episode_id)
                merged = existing + insights
                self.episodic.update_episode_metadata(
                    episode_id, key_insights=merged
                )
                logger.info(
                    f"Mid-check: {len(insights)} insights noted "
                    f"for {episode_id[:8]}"
                )

        except Exception as e:
            logger.warning(f"Micro-consolidation failed: {e}")

    def process_missed_sleep(self, queue_mode=False):
        """Consolidate episodes from sessions that ended without proper sleep."""
        if not self.enabled:
            return
        if not getattr(self.inference, 'models', None):
            logger.warning("No models — skipping missed sleep consolidation")
            return

        unconsolidated = self.episodic.get_unconsolidated_episodes(
            min_turns=self.min_turns
        )
        if not unconsolidated:
            return

        count = len(unconsolidated)
        print(
            f"\n  Found {count} unconsolidated conversation(s) from missed sleep.",
            flush=True,
        )

        for episode in unconsolidated:
            ep_id = episode["id"]
            created = episode.get("created_at", "unknown")[:19]
            tc = episode.get("turn_count", 0)
            ctx = episode.get("context_type", "operator")
            print(
                f"\n  Consolidating episode from {created} ({tc} turns, {ctx})...",
                flush=True,
            )
            if queue_mode:
                self.consolidate(ep_id, interactive=False, queue_mode=True,
                                 extraction_context=ctx)
            else:
                self.consolidate(ep_id, interactive=True,
                                 extraction_context=ctx)

        print()

    # -----------------------------------------------------------------------
    # Conversation quality gate
    # -----------------------------------------------------------------------

    def _assess_conversation_quality(self, turns):
        """Assess whether a conversation was coherent enough to learn from.

        Returns a float 0.0–1.0 where:
        - 1.0 = fully coherent, productive conversation
        - 0.0 = completely incoherent / confused
        - < 0.4 = too noisy for belief extraction

        Heuristic: ratio of assistant turns that DON'T contain confusion signals.
        """
        assistant_turns = [
            t for t in turns if t.get("role") == "assistant"
        ]
        if not assistant_turns:
            return 0.0

        coherent = 0
        for turn in assistant_turns:
            content = turn.get("content", "").lower()
            confused = any(sig in content for sig in ALL_SIGNALS)
            if not confused:
                coherent += 1

        return coherent / len(assistant_turns)

    # -----------------------------------------------------------------------
    # Analysis passes (3 focused inference calls)
    # -----------------------------------------------------------------------

    @staticmethod
    def _no_think(text):
        """Append /no_think to suppress Qwen3 thinking mode for structured output."""
        return text + " /no_think"

    def _run_pass_summary(self, conversation_text):
        """Pass 1: summary, topics, importance."""
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a conversation analyst. "
                        "Extract structured metadata. Be precise and concise."
                    ),
                },
                {
                    "role": "user",
                    "content": self._no_think(SUMMARY_PROMPT.format(
                        conversation=conversation_text
                    )),
                },
            ]
            response = self.inference.generate_with_messages(
                messages, max_tokens=self.analysis_max_tokens, temperature=0.1, timeout=240, task="reinforcement",
            )
            if not response:
                logger.warning("Pass 1 (summary): no response from model")
                return None
            logger.info(f"Pass 1 raw response:\n{response}")
            return self._parse_summary_response(response)
        except Exception as e:
            logger.warning(f"Summary pass failed: {e}")
            return None

    def _run_pass_beliefs(self, conversation_text):
        """Pass 2: beliefs and claims."""
        try:
            prompt_text = BELIEFS_PROMPT.format(
                conversation=conversation_text,
                max_beliefs=self.max_beliefs,
            )
            logger.info(
                f"Pass 2 input: {len(conversation_text)} chars conversation, "
                f"{len(prompt_text)} chars total prompt"
            )
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a conversation analyst. Extract the key beliefs "
                        "and values expressed in conversations by both participants."
                    ),
                },
                {
                    "role": "user",
                    "content": self._no_think(prompt_text),
                },
            ]
            response = self.inference.generate_with_messages(
                messages, max_tokens=self.analysis_max_tokens, temperature=0.1, timeout=240, task="reinforcement",
            )
            if not response:
                logger.warning("Pass 2 (beliefs): no response from model")
                return []
            logger.info(f"Pass 2 raw response:\n{response}")
            return self._parse_beliefs_response(response)
        except Exception as e:
            logger.warning(f"Beliefs pass failed: {e}")
            return []

    def _run_pass_corrections(self, conversation_text):
        """Pass 3: corrections, pushback, and lessons.

        Returns (corrections, pushbacks, lessons).
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a conversation analyst. "
                        "Identify corrections, pushback, and behavioral lessons. Be precise."
                    ),
                },
                {
                    "role": "user",
                    "content": self._no_think(CORRECTIONS_PROMPT.format(
                        conversation=conversation_text,
                        max_corrections=self.max_corrections,
                    )),
                },
            ]
            response = self.inference.generate_with_messages(
                messages, max_tokens=self.analysis_max_tokens, temperature=0.1, timeout=240, task="reinforcement",
            )
            if not response:
                logger.warning("Pass 3 (corrections): no response from model")
                return [], [], []
            logger.info(f"Pass 3 raw response:\n{response}")
            return self._parse_corrections_response(response)
        except Exception as e:
            logger.warning(f"Corrections pass failed: {e}")
            return [], [], []

    def _run_pass_questions(self, conversation_text):
        """Pass 4: unanswered questions and knowledge gaps."""
        if not self.curiosity:
            return []
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a conversation analyst. "
                        "Identify unresolved questions and knowledge gaps. Be precise."
                    ),
                },
                {
                    "role": "user",
                    "content": self._no_think(QUESTIONS_PROMPT.format(
                        conversation=conversation_text,
                        max_questions=3,
                    )),
                },
            ]
            response = self.inference.generate_with_messages(
                messages, max_tokens=self.analysis_max_tokens, temperature=0.1, timeout=240, task="reinforcement",
            )
            if not response:
                logger.warning("Pass 4 (questions): no response from model")
                return []
            logger.info(f"Pass 4 raw response:\n{response}")
            return self._parse_questions_response(response)
        except Exception as e:
            logger.warning(f"Questions pass failed: {e}")
            return []

    def _parse_questions_response(self, response):
        """Parse QUESTION: ... | CONTEXT: ... lines from model output."""
        questions = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.upper() == "NONE":
                return []
            if not line.upper().startswith("QUESTION:"):
                continue
            parts = line.split("|")
            question_text = parts[0].split(":", 1)[1].strip() if len(parts) >= 1 else ""
            context = ""
            if len(parts) >= 2:
                ctx_part = parts[1].strip()
                if ctx_part.upper().startswith("CONTEXT:"):
                    context = ctx_part.split(":", 1)[1].strip()
            if question_text and len(question_text) >= 10:
                questions.append({
                    "question": question_text,
                    "type": "knowledge_gap",
                    "context": f"Emerged from consolidation: {context}",
                    "priority": "medium",
                })
        return questions


    # -----------------------------------------------------------------------
    # Response parsing
    # -----------------------------------------------------------------------

    def _parse_summary_response(self, response):
        """Parse SUMMARY/TOPICS/IMPORTANCE from model output."""
        result = {"summary": None, "topics": [], "importance": 0.5}

        for line in response.strip().split("\n"):
            line = line.strip()
            upper = line.upper()

            if upper.startswith("SUMMARY:"):
                result["summary"] = line.split(":", 1)[1].strip()

            elif upper.startswith("TOPICS:"):
                raw = line.split(":", 1)[1].strip()
                result["topics"] = [
                    t.strip() for t in raw.split(",") if t.strip()
                ]

            elif upper.startswith("IMPORTANCE:"):
                try:
                    val = float(line.split(":", 1)[1].strip())
                    result["importance"] = max(0.0, min(1.0, val))
                except ValueError:
                    pass

        return result if result["summary"] else None

    @staticmethod
    def _strip_prefix(line):
        """Strip leading numbering/bullets: '1. ', '- ', '* ', etc."""
        import re
        return re.sub(r'^[\d]+[\.\)]\s*', '', line.lstrip('- *').strip())

    def _parse_beliefs_response(self, response):
        """Parse BELIEF lines from model output."""
        if response.strip().upper() == "NONE":
            return []

        beliefs = []
        for line in response.strip().split("\n"):
            line = self._strip_prefix(line.strip())
            if not line.upper().startswith("BELIEF:"):
                continue

            remainder = line.split(":", 1)[1]
            segments = remainder.split("|")

            parts = {}
            if segments:
                parts["statement"] = segments[0].strip()

            for seg in segments[1:]:
                seg = seg.strip()
                seg_upper = seg.upper()
                if seg_upper.startswith("CONFIDENCE:"):
                    parts["confidence"] = seg.split(":", 1)[1].strip().lower()
                elif seg_upper.startswith("SOURCE:"):
                    parts["source"] = seg.split(":", 1)[1].strip().lower()
                elif seg_upper.startswith("EVIDENCE:"):
                    parts["evidence"] = seg.split(":", 1)[1].strip()

            if parts.get("statement"):
                beliefs.append(parts)

        # Diagnostic trace on failure path — 0 beliefs from non-NONE response
        if not beliefs and response.strip().upper() != "NONE":
            logger.warning("Pass 2 parse failure — response had content but 0 beliefs extracted")
            for i, raw_line in enumerate(response.strip().split("\n")):
                stripped = self._strip_prefix(raw_line.strip())
                matched = stripped.upper().startswith("BELIEF:")
                logger.warning(
                    f"  line {i}: matched={matched} | raw={raw_line[:100]} | stripped={stripped[:100]}"
                )

        return beliefs[: self.max_beliefs]

    def _parse_corrections_response(self, response):
        """Parse CORRECTION, PUSHBACK, and LESSON lines.

        Returns (corrections, pushbacks, lessons).
        """
        if response.strip().upper() == "NONE":
            return [], [], []

        corrections = []
        pushbacks = []
        lessons = []

        for line in response.strip().split("\n"):
            line = self._strip_prefix(line.strip())
            upper = line.upper()

            if upper.startswith("CORRECTION"):
                remainder = line.split(":", 1)[1].strip()
                # Handle "CORRECTION: NONE" or "CORRECTIONS: NONE"
                if remainder.upper() == "NONE":
                    continue
                segments = remainder.split("|")

                parts = {}
                if segments:
                    arrow = segments[0].split("->")
                    if len(arrow) == 2:
                        parts["original"] = arrow[0].strip()
                        parts["corrected"] = arrow[1].strip()

                for seg in segments[1:]:
                    seg = seg.strip()
                    seg_upper = seg.upper()
                    if seg_upper.startswith("BY:"):
                        parts["by"] = seg.split(":", 1)[1].strip().lower()
                    elif seg_upper.startswith("REASON:"):
                        parts["reason"] = seg.split(":", 1)[1].strip()

                if (
                    parts.get("original") and parts.get("corrected")
                    and parts["original"].strip().lower() != parts["corrected"].strip().lower()
                ):
                    corrections.append(parts)

            elif upper.startswith("PUSHBACK"):
                remainder = line.split(":", 1)[1].strip()
                # Handle "PUSHBACK: NONE"
                if remainder.upper() == "NONE":
                    continue
                segments = remainder.split("|")

                parts = {}
                if segments:
                    parts["challenged"] = segments[0].strip()
                for seg in segments[1:]:
                    seg = seg.strip()
                    if seg.upper().startswith("RESOLUTION:"):
                        parts["resolution"] = seg.split(":", 1)[1].strip()

                # Reject empty or punctuation-only pushbacks
                challenged = parts.get("challenged", "")
                if challenged and re.sub(r'[^\w]', '', challenged).strip():
                    pushbacks.append(parts)

            elif upper.startswith("LESSON"):
                remainder = line.split(":", 1)[1].strip()
                if remainder.upper() == "NONE":
                    continue
                segments = remainder.split("|")

                parts = {}
                if segments:
                    parts["principle"] = segments[0].strip()
                for seg in segments[1:]:
                    seg = seg.strip()
                    seg_upper = seg.upper()
                    if seg_upper.startswith("SOURCE:"):
                        parts["source"] = seg.split(":", 1)[1].strip().lower()
                    elif seg_upper.startswith("CONTEXT:"):
                        parts["context"] = seg.split(":", 1)[1].strip()

                # Reject empty lessons
                principle = parts.get("principle", "")
                if principle and re.sub(r'[^\w]', '', principle).strip():
                    lessons.append(parts)

        return corrections[: self.max_corrections], pushbacks, lessons

    # -----------------------------------------------------------------------
    # Approval system — smart defaults, nothing sticks without consent
    # -----------------------------------------------------------------------

    # Phrases that indicate a belief is self-referential model narration
    _SELF_REF = [
        "the assistant", "i am able", "i can ", "i have access",
        "my abilities", "my capabilities", "i'm able", "i'm still",
        "still learning", "still refining", "i am still",
    ]

    # Identity-related keywords — loaded from config, no hardcoded defaults.
    # Set via config["identity_keywords"] = ["operator_name", "anima", ...]

    # Quality gate moved to Rust: anima_core.Engine.validate_quality()
    # Called via semantic._engine.validate_quality(statement, source, product_mode)

    def _recommend(self, category, data):
        """Return True if this item is recommended for acceptance."""
        if category == "correction":
            orig = data.get("original", "").strip()
            corr = data.get("corrected", "").strip()
            if not orig or not corr or orig.lower() == corr.lower():
                return False
            return True
        if category == "pushback":
            challenged = data.get("challenged", "").strip()
            if not challenged or not re.sub(r'[^\w]', '', challenged):
                return False
            return True
        if category == "lesson":
            principle = data.get("principle", "").strip()
            if not principle or not re.sub(r'[^\w]', '', principle):
                return False
            return data.get("source", "") == "user"

        # --- DREAMS: quality-gated recommendation ---
        if category in ("dream", "triplet_dream"):
            return self._recommend_dream(data)

        # Beliefs: quality-gated recommendation
        statement = data.get("statement", "").strip()
        statement_lower = statement.lower()
        source = data.get("source", "unknown")

        _identity_kw = self.config.get("identity_keywords", []) if self.config else []
        if _identity_kw and any(kw in statement_lower for kw in _identity_kw):
            return True

        if source == "assistant":
            if any(pat in statement_lower for pat in self._SELF_REF):
                return False
            return False

        if data.get("extraction_context") == "exploration":
            return False

        # Rule 1: No questions as beliefs.
        # "Whether X could enhance Y" is a question, not a belief.
        _question_starts = ("whether ", "could ", "might ", "is it possible",
                           "can ", "would ", "should ", "does ", "do ")
        if any(statement_lower.startswith(q) for q in _question_starts):
            return False
        if "?" in statement:
            return False

        # Rule 2: Minimum informational value.
        # Must have: named entity + directional/causal claim.
        # Both missing = reject.
        import re as _re
        _has_entity = bool(_re.search(
            r'\b[A-Z][A-Z0-9]{1,}(?:[-/][A-Za-z0-9]+)*\b'  # Uppercase acronyms (NF-kB, VEGF)
            r'|[α-ωΑ-Ω][\w-]+'                              # Greek-letter terms (α-synuclein)
            r'|[a-z][A-Z]\w+'                                # Mixed case (mTOR, p53-like)
            r'|\b\d+[-]?[A-Z]{2,}\b'                        # Numeric+alpha codes (5-HT, IL-6)
            r'|\b[A-Z][a-z]+[-/][A-Z]',                     # Hyphenated proper (Bcl-2, Wnt/β)
            statement,
        ))
        _has_direction = bool(_re.search(
            r'\b(inhibit|promot|reduc|increas|modulat|activat|suppress'
            r'|block|enhanc|induc|attenuate|downregulat|upregulat'
            r'|leads?\s+to|associated\s+with|contributes?\s+to'
            r'|drives?|triggers?|prevents?)\w*\b',
            statement, _re.IGNORECASE
        ))
        if not _has_entity and not _has_direction:
            return False

        # Rule 3: Novelty check against existing graph.
        # Reject if core claim already represented at >0.85 similarity.
        try:
            if hasattr(self, 'embeddings') and self.embeddings:
                from memory.dedup import is_duplicate
                cand_emb = self.embeddings.embed(statement)
                existing = self.semantic.search_beliefs(limit=100)
                corpus_embs = []
                for b in existing:
                    try:
                        b_emb = self.embeddings.embed(b["statement"])
                        corpus_embs.append((b_emb, b["statement"]))
                    except Exception:
                        continue
                is_dup, _ = is_duplicate(cand_emb, corpus_embs, threshold=0.92)
                if is_dup:
                    return False
        except Exception:
            pass

        return True

    def _recommend_dream(self, data):
        """Quality-gated recommendation for dream connections.

        Not every dream is worth the operator's time. Recommend if:
        1. Cross-domain (parent beliefs from different domains)
        2. Names specific entities/mechanisms (not generic claims)
        3. Passes editorial signal check

        Returns True only if the dream meets quality criteria.
        """
        inference = data.get("inference", "").lower()
        domain_a = data.get("_domain_a", "other")
        domain_b = data.get("_domain_b", "other")

        # Must be cross-domain (if domains are assigned)
        # Skip this check if both are "other" — domain classifier may not be active
        both_classified = domain_a != "other" and domain_b != "other"
        if both_classified and domain_a == domain_b:
            return False  # same-domain dream — not novel enough to recommend

        # Must have at least one specificity signal (multi-signal, not just entities)
        # No domain-specific terms — uses structural patterns only.
        # Plugins can inject domain terms via config extraction.specificity_signals.
        import re as _re
        _has_named_entity = bool(_re.search(
            r'\b[A-Z][A-Z0-9]{1,}(?:[-/][A-Za-z0-9]+)*\b'
            r'|[α-ωΑ-Ω][\w-]+'
            r'|[a-z][A-Z]\w+'
            r'|\b\d+[-]?[A-Z]{2,}\b'
            r'|\b[A-Z][a-z]+[-/][A-Z]',
            data.get("inference", ""),
        ))
        _has_numeric = bool(_re.search(r'\d+\.?\d*\s*(%|mg|ml|µ[mM]|nm|kDa|fold)', inference))
        _has_mechanism = bool(_re.search(
            r'\b(inhibit|activat|suppress|regulat|modulat|induc|block|mediat)\w*\b',
            inference,
        ))
        _domain_specificity = self.config.get(
            "extraction", {}
        ).get("specificity_signals", [])
        has_specificity = (
            _has_named_entity
            or _has_numeric
            or _has_mechanism
            or any(s in inference for s in _domain_specificity)
        )

        if not has_specificity:
            # Sample 1-in-10 to avoid log spam under load
            if not hasattr(self, '_recommend_reject_count'):
                self._recommend_reject_count = 0
            self._recommend_reject_count += 1
            if self._recommend_reject_count % 10 == 1:
                logger.debug(
                    f"Dream not recommended (no specificity): entity={_has_named_entity} "
                    f"numeric={_has_numeric} mechanism={_has_mechanism} "
                    f"text='{inference[:60]}'"
                )
            return False  # too generic to recommend

        # Editorial signal check (already in triage, but double-check here)
        editorial_signals = [
            "structural bias", "status quo", "deliberate avoidance",
            "stifles innovation", "paradigm", "entrenched",
            "institutional", "field is not", "resistance to rethinking",
        ]
        if sum(1 for s in editorial_signals if s in inference) >= 1:
            return False  # editorial, not synthesis

        return True

    def _item_label(self, category, data, recommended):
        """Build a concise one-line label with recommendation marker."""
        marker = "+" if recommended else "?"

        if category == "belief":
            source = data.get("source", "")
            tag = source if source else "unknown"
            return f"{marker} {data['statement'][:90]}  ({tag})"

        elif category == "correction":
            return (
                f"{marker} {data['original'][:40]} -> "
                f"{data['corrected'][:40]}  (correction)"
            )

        elif category == "pushback":
            return f"{marker} Pushback: {data['challenged'][:70]}"

        elif category == "lesson":
            source = data.get("source", "")
            tag = f"from {source}" if source else "lesson"
            return f"{marker} Lesson: {data['principle'][:70]}  ({tag})"

        return f"{marker} {str(data)[:90]}"

    def _propose_changes(self, beliefs, corrections, pushbacks, lessons=None):
        """Present proposed changes with smart recommendations.

        Items marked + are recommended for acceptance.
        Items marked ? are questionable and recommended to skip.

        Returns (approved_beliefs, approved_corrections, approved_pushbacks,
                 approved_lessons).
        """
        lessons = lessons or []
        approved = {
            "belief": [], "correction": [], "pushback": [], "lesson": [],
        }

        # Build items with recommendations
        items = []  # (category, data, recommended, label)

        for b in beliefs:
            rec = self._recommend("belief", b)
            label = self._item_label("belief", b, rec)
            items.append(("belief", b, rec, label))

        for c in corrections:
            rec = self._recommend("correction", c)
            label = self._item_label("correction", c, rec)
            items.append(("correction", c, rec, label))

        for p in pushbacks:
            rec = self._recommend("pushback", p)
            label = self._item_label("pushback", p, rec)
            items.append(("pushback", p, rec, label))

        for l in lessons:
            rec = self._recommend("lesson", l)
            label = self._item_label("lesson", l, rec)
            items.append(("lesson", l, rec, label))

        if not items:
            return (approved["belief"], approved["correction"],
                    approved["pushback"], approved["lesson"])

        rec_count = sum(1 for _, _, r, _ in items if r)
        skip_count = len(items) - rec_count

        print("\n  === Post-Conversation Analysis ===\n")
        for i, (_, _, rec, label) in enumerate(items):
            print(f"  [{i + 1}] {label}")

        if rec_count > 0 and skip_count > 0:
            print(
                f"\n  [a]ccept recommended ({rec_count} +, skip {skip_count} ?) "
                f"| [r]eview each | [s]kip all"
            )
        elif rec_count > 0:
            print(f"\n  [a]ccept all ({rec_count}) | [r]eview each | [s]kip all")
        else:
            print(f"\n  All items are questionable. [r]eview each | [s]kip all")

        try:
            choice = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            choice = "s"

        if choice == "a":
            for cat, data, rec, _ in items:
                if rec:
                    approved[cat].append(data)
            total = sum(len(v) for v in approved.values())
            print(f"  Accepted {total}, skipped {skip_count}.\n")

        elif choice == "s":
            print("  Skipped all.\n")

        elif choice == "r":
            for i, (cat, data, rec, _) in enumerate(items):
                hint = " (recommended)" if rec else " (skip recommended)"
                try:
                    ans = input(
                        f"  [{i + 1}] Accept?{hint} [y/n] "
                    ).strip().lower()
                except (EOFError, KeyboardInterrupt):
                    remaining = len(items) - i
                    print(f"\n  Interrupted. Skipping remaining {remaining} items.")
                    break

                if ans in ("y", "yes"):
                    approved[cat].append(data)

            total = sum(len(v) for v in approved.values())
            print(f"  Accepted {total}/{len(items)} changes.\n")

        else:
            print("  Skipped all.\n")

        return (approved["belief"], approved["correction"],
                approved["pushback"], approved["lesson"])

    # -------------------------------------------------------------------
    # Auto-triage — filter items before they reach the approval queue
    # -------------------------------------------------------------------

    @staticmethod
    def _content_hash(category, text):
        """Hash content for rejection tracking."""
        normalized = f"{category}:{text.strip().lower()}"
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    @staticmethod
    def _get_triage_text(category, data):
        """Extract primary text content for triage/hashing."""
        if category == "belief":
            return data.get("statement", "")
        elif category == "correction":
            return f"{data.get('original', '')} -> {data.get('corrected', '')}"
        elif category == "pushback":
            return data.get("challenged", "")
        elif category == "lesson":
            return data.get("principle", "")
        elif category in ("dream", "triplet_dream"):
            return data.get("inference", "")
        elif category == "reflection":
            return data.get("pattern", "")
        return str(data)

    def _record_rejection(self, category, data):
        """Record a rejection in the tracking table."""
        text = self._get_triage_text(category, data)
        h = self._content_hash(category, text)
        db = self._get_queue_db()
        now = datetime.now(timezone.utc).isoformat()

        row = db.execute(
            "SELECT rejection_count FROM rejection_history WHERE content_hash=? AND category=?",
            (h, category),
        ).fetchone()

        if row:
            db.execute(
                "UPDATE rejection_history SET rejection_count=rejection_count+1, "
                "last_rejected=? WHERE content_hash=? AND category=?",
                (now, h, category),
            )
        else:
            db.execute(
                "INSERT INTO rejection_history (content_hash, category, rejection_count, last_rejected) "
                "VALUES (?, ?, 1, ?)",
                (h, category, now),
            )
        db.commit()

    def _get_rejection_count(self, category, text):
        """Get how many times this content has been rejected."""
        h = self._content_hash(category, text)
        db = self._get_queue_db()
        row = db.execute(
            "SELECT rejection_count FROM rejection_history WHERE content_hash=? AND category=?",
            (h, category),
        ).fetchone()
        return row[0] if row else 0

    def _log_dream_triage(self, data, decision, reason):
        """Log a dream/triplet triage decision to dms_triage_log for health metrics."""
        try:
            now = datetime.now(timezone.utc).isoformat()
            stmt = (data.get("inference") or "")[:500]
            sim = data.get("similarity") or data.get("avg_similarity") or ""
            self.semantic.db_conn.execute(
                "INSERT INTO dms_triage_log "
                "(category, statement, decision, reason, confidence, "
                "source_type, document_sha, created_at) "
                "VALUES (?, ?, ?, ?, ?, 'synthesis', '', ?)",
                ("dream", stmt, decision, reason, str(sim), now),
            )
            self.semantic.db_conn.commit()
        except Exception:
            pass

    def _auto_triage(self, category, data):
        """Determine whether to auto-accept, auto-reject, or queue for review.

        Rust state machine drives all decisions. Python only executes side effects
        (LLM translation, classification, rewrite, semantic search).

        Returns (decision, reason).
        """
        core = getattr(self.semantic, '_engine', None)
        if not core:
            return ("queue", "no Rust engine available")

        # Build triage input
        triage_input = {
            "category": category,
            "statement": data.get("statement", data.get("inference", "")),
            "confidence": self._map_confidence(
                data.get("confidence", "medium"),
                data.get("source", "unknown"),
                is_correction=(category == "correction"),
            ) if isinstance(data.get("confidence"), str) else float(data.get("confidence", 0.5)),
            "source": data.get("source", "unknown"),
            "extraction_context": data.get("extraction_context", ""),
            "product_mode": self.config.get("product", {}).get("mode", "") if self.config else "",
            "similarity": float(data.get("similarity", data.get("avg_similarity", 0.0))),
            "belief_a_id": data.get("belief_a_id", ""),
            "belief_b_id": data.get("belief_b_id", ""),
            "domain_a": data.get("_domain_a", ""),
            "domain_b": data.get("_domain_b", ""),
            "graph_context": data.get("_graph_context"),
            "original": data.get("original", ""),
            "corrected": data.get("corrected", ""),
            "principle": data.get("principle", ""),
            "pattern": data.get("pattern", ""),
            "mitigation": data.get("mitigation", ""),
            "challenged": data.get("challenged", ""),
            "rejection_count": self._get_rejection_count(category, data.get("statement", data.get("inference", ""))),
        }

        governance = {
            "allow_auto_accept": self._governance.get("allow_auto_accept", True),
            "allow_auto_corrections": self._governance.get("allow_auto_corrections", True),
            "allow_auto_lessons": self._governance.get("allow_auto_lessons", True),
        }

        # Start the triage state machine
        state = core.start_triage(triage_input, governance)

        # Executor loop — Rust drives, Python executes side effects
        while True:
            step = state.get("step")

            if step == "FinalDecision":
                decision = state["decision"]
                reason = state["reason"]
                # Sync any rewrites back to data dict
                return (decision, reason)

            elif step == "NeedsTranslation":
                translated = self._translate_to_english(state["text"])
                state = core.triage_resume_translation(state["id"], translated)

            elif step == "NeedsClassification":
                classification, _ = self._llm_classify_dream(state["inference"])
                state = core.triage_resume_classification(state["id"], classification)

            elif step == "NeedsRewrite":
                rewrite = self._llm_rewrite_dream(
                    state["inference"], state["classification"], ""
                )
                state = core.triage_resume_rewrite(state["id"], rewrite)

            elif step == "NeedsContradictionCheck":
                matches = self._search_contradictions(state["statement"])
                state = core.triage_resume_contradictions(state["id"], matches)

            elif step == "NeedsConstraintCheck":
                matches = self._search_constraints(state["statement"])
                state = core.triage_resume_constraints(state["id"], matches)

            elif step == "NeedsDuplicateCheck":
                is_dup = self._has_high_confidence_duplicate(state["statement"])
                state = core.triage_resume_duplicate(state["id"], is_dup)

            else:
                logger.warning(f"Unknown triage step: {step}")
                return ("queue", f"unknown triage step: {step}")

            if state is None:
                return ("queue", "triage state lost")

    def _search_contradictions(self, statement):
        """Search for directionally conflicting beliefs. Returns list of dicts for Rust."""
        import re as _re
        matches = []
        try:
            if not hasattr(self, 'embeddings') or not self.embeddings:
                return matches
            import numpy as np
            _DIR_PROMOTE = _re.compile(r'\b(promot|increas|activat|enhanc|upregulat|stimulat|induc)\w*\b', _re.IGNORECASE)
            _DIR_INHIBIT = _re.compile(r'\b(inhibit|suppress|reduc|block|downregulat|attenuate|prevent|diminish)\w*\b', _re.IGNORECASE)
            cand_emb = self.embeddings.embed(statement)
            existing = self.semantic.search_beliefs(limit=50)
            for eb in existing:
                try:
                    eb_emb = self.embeddings.embed(eb["statement"])
                    norm = np.linalg.norm(cand_emb) * np.linalg.norm(eb_emb)
                    sim = float(np.dot(cand_emb, eb_emb) / norm) if norm > 0 else 0.0
                    if sim >= 0.55:
                        matches.append({
                            "belief_id": eb.get("id", ""),
                            "statement": eb["statement"],
                            "similarity": sim,
                            "promotes": bool(_DIR_PROMOTE.search(eb["statement"])),
                            "inhibits": bool(_DIR_INHIBIT.search(eb["statement"])),
                        })
                except Exception:
                    continue
        except Exception:
            pass
        return matches

    def _search_constraints(self, statement):
        """Search hypothesis constraints for conflicting beliefs. Returns list of dicts for Rust."""
        matches = []
        try:
            if not hasattr(self, 'embeddings') or not self.embeddings:
                return matches
            import numpy as np
            constraints = self.semantic.db_conn.execute(
                "SELECT statement FROM hypothesis_constraints"
            ).fetchall()
            if not constraints:
                return matches
            dream_emb = self.embeddings.embed(statement)
            for con in constraints:
                try:
                    con_text = con["statement"] if isinstance(con, dict) else con[0]
                    con_emb = self.embeddings.embed(con_text)
                    cn = np.linalg.norm(dream_emb) * np.linalg.norm(con_emb)
                    if cn > 0:
                        csim = float(np.dot(dream_emb, con_emb) / cn)
                        if csim > 0.50:
                            matches.append({"statement": con_text, "similarity": csim})
                except Exception:
                    continue
        except Exception:
            pass
        return matches

    # _apply_governance, _triage_belief, _triage_correction, _triage_pushback,
    # _triage_lesson — all moved to Rust triage state machine.

    def _translate_to_english(self, text):
        """Translate non-English scientific text to English via LLM.

        Returns the English translation, or None if translation fails.
        """
        import re as _re
        try:
            prompt = (
                "Translate this scientific statement to English. "
                "Preserve all technical terms, gene names, pathway names, and measurements exactly. "
                "Output ONLY the English translation, nothing else.\n\n"
                f"Text: {text[:500]}\n\n"
                "English translation:\n/no_think"
            )
            result = self.inference.generate_with_messages(
                messages=[
                    {"role": "system", "content": "You translate scientific text to English. Output only the translation."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200, timeout=90, task="triage",
            )
            if not result:
                return None
            result = _re.sub(r"<think>.*?</think>", "", result, flags=_re.DOTALL).strip()
            translation = result.split("\n")[0].strip()
            translation = _re.sub(r'^(english translation:?\s*)', '', translation, flags=_re.IGNORECASE).strip()
            translation = translation.strip('"\'').strip()

            # Validate: must be mostly English now
            non_latin = len(_re.findall(r'[^\x00-\x7Fα-ωΑ-Ωβ-ψ°±µ≥≤→←↔∞≈∆∑πσ—–‑·•‐′″]', translation))
            if non_latin > len(translation) * 0.10:
                logger.info(f"Translation still non-English: '{translation[:60]}'")
                return None

            if translation and len(translation) > 10:
                return translation
        except Exception as e:
            logger.debug(f"Translation failed: {e}")
        return None

    def _llm_classify_dream(self, inference):
        """Classify a dream belief as MECHANISM, NARRATIVE, or TEMPLATE.

        Returns (classification, reason) tuple.
        """
        import re as _re
        prompt = (
            "Classify this scientific statement into exactly one category:\n\n"
            "MECHANISM — states exactly ONE causal relationship: A causes/inhibits/activates B. "
            "You can extract a single 'subject → verb → object' from it.\n"
            "(e.g., 'Compound X inhibits the Y/Z signaling pathway' = MECHANISM)\n\n"
            "NARRATIVE — DISCUSSES or FRAMES mechanisms without making one specific claim. "
            "Uses words like 'dual vulnerability', 'two-pronged', 'promising candidate', "
            "'multi-faceted'. Talks ABOUT relationships rather than STATING one.\n"
            "(e.g., 'X may create a dual vulnerability by targeting Y and Z, representing "
            "a promising approach' = NARRATIVE)\n\n"
            "TEMPLATE — applies a generic pattern like 'feedback loop' or 'synergistic effect' "
            "with different nouns each time. Same sentence structure, different topics.\n\n"
            f"Statement: {inference[:400]}\n\n"
            "Reply with exactly one word — MECHANISM, NARRATIVE, or TEMPLATE:\n/no_think"
        )
        result = self.inference.generate_with_messages(
            messages=[
                {"role": "system", "content": "You classify scientific statements. A MECHANISM states ONE specific causal relationship. A NARRATIVE discusses or frames mechanisms without stating one. Reply with one word."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=10, timeout=90, task="triage",
        )
        if not result:
            return "UNCLEAR", "no model response"
        result = _re.sub(r"<think>.*?</think>", "", result, flags=_re.DOTALL).strip()
        classification = result.strip().upper().split()[0] if result else "UNCLEAR"
        return classification, f"classified as {classification.lower()}"

    def _llm_rewrite_dream(self, inference, classification, reason):
        """Attempt to rewrite a narrative/template dream into a core mechanism.

        Returns the rewritten statement, or None if rewrite fails.
        """
        import re as _re
        prompt = (
            f"This synthesis belief was classified as {classification} because it contains "
            f"{reason}.\n\n"
            "Extract ONLY the core mechanistic claim. Rules:\n"
            "- One sentence, under 30 words\n"
            "- Must contain: subject → action → object\n"
            "- No framing ('this suggests', 'creating a feedback loop', 'implying that')\n"
            "- No therapeutic recommendations ('could be targeted', 'may serve as')\n"
            "- If no single mechanistic claim can be extracted, respond NO_MECHANISM\n\n"
            f"Original: {inference[:500]}\n\n"
            "Rewrite:\n/no_think"
        )
        result = self.inference.generate_with_messages(
            messages=[
                {"role": "system", "content": "You extract core mechanistic claims from scientific text. One sentence, under 30 words. If no mechanism exists, reply NO_MECHANISM."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=60, timeout=90, task="triage",
        )
        if not result:
            return None
        result = _re.sub(r"<think>.*?</think>", "", result, flags=_re.DOTALL).strip()
        # Clean up
        rewrite = result.split("\n")[0].strip()
        rewrite = _re.sub(r'^(rewrite:?\s*)', '', rewrite, flags=_re.IGNORECASE).strip()
        rewrite = rewrite.strip('"\'').strip()

        # Validate rewrite
        if not rewrite or "NO_MECHANISM" in rewrite.upper():
            logger.info(f"Dream rewrite failed: no mechanism in '{inference[:50]}...'")
            return None

        # Word count check
        if len(rewrite.split()) > 30:
            logger.info(f"Dream rewrite too long ({len(rewrite.split())}w): '{rewrite[:60]}...'")
            return None

        # Run structural pre-filters on rewrite
        rewrite_lower = rewrite.lower()

        # Meta-pattern check on rewrite
        _DREAM_META_PATTERNS = [
            "belief a", "belief b", "this suggests that", "this could explain",
            "this creates a", "this dynamic interplay", "feedback loop",
            "master switch", "dual-function", "in essence,", "paradigm shift",
            "systemic regulator", "network-level", "master regulator",
            "bridge molecule", "act as a bridge", "as a bridge", "bridge between",
            "polypharmacological", "creating a feedback",
        ]
        if any(pat in rewrite_lower for pat in _DREAM_META_PATTERNS):
            logger.info(f"Dream rewrite still contains meta-pattern: '{rewrite[:60]}...'")
            return None

        # Entity cap (≤5 for rewrites)
        _NE = _re.compile(
            r'\b[A-Z][A-Za-z]{2,}(?:[-/][A-Za-z0-9]+)*\b'
            r'|[A-Z][A-Z0-9]{1,}(?:[-/][A-Za-z0-9]+)*'
            r'|[α-ωΑ-Ω][\w-]+',
        )
        _STOP = {"the","this","that","these","both","when","where","which",
            "however","therefore","moreover","furthermore","although",
            "while","since","because","between","through","within",
            "not","but","also","may","can","could","would"}
        entities = set(m.lower() for m in _NE.findall(rewrite)
                       if m.lower() not in _STOP and len(m) > 1)
        if len(entities) > 5:
            logger.info(f"Dream rewrite over-composed ({len(entities)} entities): '{rewrite[:60]}...'")
            return None

        # Novelty check — don't create duplicates
        try:
            if hasattr(self, 'embeddings') and self.embeddings:
                from memory.dedup import is_duplicate
                cand_emb = self.embeddings.embed(rewrite)
                existing = self.semantic.search_beliefs(limit=100)
                corpus_embs = [(self.embeddings.embed(b["statement"]), b["statement"])
                               for b in existing]
                is_dup, _ = is_duplicate(cand_emb, corpus_embs, threshold=0.85)
                if is_dup:
                    logger.info(f"Dream rewrite is duplicate: '{rewrite[:60]}...'")
                    return None
        except Exception:
            pass

        logger.info(f"Dream rewrite accepted: '{rewrite}'")
        return rewrite

    # _triage_dream and _triage_reflection — moved to Rust triage state machine.
    # Removed ~380 lines of Python heuristics. All now in anima_core/src/triage/

    def _has_high_confidence_duplicate(self, statement, min_confidence=0.7):
        """Check if statement duplicates an existing belief above confidence threshold."""
        existing = self.semantic.search_beliefs()
        if not existing:
            return False

        new_emb = self.embeddings.embed(statement)
        for belief in existing:
            if belief.get("confidence", 0) < min_confidence:
                continue
            emb = self.embeddings.embed(belief["statement"])
            sim = self._cosine_similarity(new_emb, emb)
            if sim >= self.similarity_threshold:
                return True
        return False

    # Opposing claim indicators — only these signal actual contradiction.
    # Refinements, extensions, and details are NOT contradictions.
    _OPPOSING_INDICATORS = [
        # Direct negation
        "does not", "do not", "cannot", "is not", "are not", "was not",
        "no effect", "no role", "no association", "no correlation",
        "no significant", "fails to", "failed to", "unable to",
        # Opposing direction
        "inhibit", "suppress", "reduce", "decrease", "downregulate",
        "block", "prevent", "attenuate", "impair", "diminish",
        # vs activating direction
        "activate", "enhance", "increase", "upregulate", "promote",
        "stimulate", "amplify", "potentiate", "induce", "elevate",
    ]

    def _check_potential_contradiction(self, statement, epistemic_class=None):
        """Check if statement contradicts a high-confidence existing belief.

        Rewritten logic: only flags contradiction when same entity + same
        variable + opposing claim. Refinements, extensions, and details pass
        through. Corpus epistemic class bypasses entirely.

        Returns None (no contradiction) or match dict.
        """
        # Fix #2: Corpus beliefs bypass contradiction gate entirely.
        # Paper-sourced facts ARE ground truth — the system builds on them.
        if epistemic_class == "corpus":
            return None

        existing = self.semantic.search_beliefs()
        if not existing:
            return None

        new_emb = self.embeddings.embed(statement)
        new_lower = statement.lower()

        for belief in existing:
            if belief.get("confidence", 0) < 0.8:
                continue
            if belief.get("deprecated", 0):
                continue
            emb = self.embeddings.embed(belief["statement"])
            sim = self._cosine_similarity(new_emb, emb)

            # Related but not duplicate range
            if not (0.5 <= sim < self.similarity_threshold):
                continue

            # Fix #1 v2: Strict contradiction detection.
            # Only flag when statements make DIRECTLY OPPOSING claims about
            # the SAME subject. Having inhibit/activate verbs in both
            # statements is not enough — research papers routinely describe
            # both activation and inhibition of different targets.
            #
            # Strategy: require explicit negation of the same claim.
            # "X activates Y" vs "X does NOT activate Y" = contradiction.
            # "X activates Y" vs "X inhibits Z" = NOT a contradiction.
            existing_lower = belief["statement"].lower()

            # Extract (subject, verb, target) patterns from each statement
            import re as _re
            _CLAIM_PATTERN = _re.compile(
                r'(\w[\w\s-]{2,30}?)\s+'
                r'(does not|do not|cannot|fails to|unable to|'
                r'inhibit\w*|suppress\w*|reduce\w*|block\w*|prevent\w*|'
                r'activate\w*|enhance\w*|increase\w*|promot\w*|induc\w*|'
                r'upregulat\w*|downregulat\w*)\s+'
                r'(\w[\w\s/-]{2,30})',
                _re.IGNORECASE
            )

            new_claims = _CLAIM_PATTERN.findall(new_lower)
            existing_claims = _CLAIM_PATTERN.findall(existing_lower)

            if not new_claims or not existing_claims:
                continue  # Can't parse claims — not a contradiction

            # Check for opposing claims about the same target
            _neg_verbs = {"does not", "do not", "cannot", "fails to", "unable to",
                         "inhibit", "suppress", "reduce", "block", "prevent",
                         "downregulat"}
            _pos_verbs = {"activate", "enhance", "increase", "promot", "induc",
                         "upregulat", "stimulat"}

            def _verb_direction(verb):
                v = verb.lower().split()[0][:8]
                if any(n in verb.lower() for n in ("does not", "do not", "cannot", "fails", "unable")):
                    return "neg"
                for nv in _neg_verbs:
                    if v.startswith(nv[:6]):
                        return "neg"
                for pv in _pos_verbs:
                    if v.startswith(pv[:6]):
                        return "pos"
                return "unknown"

            is_opposing = False
            for ns, nv, nt in new_claims:
                n_dir = _verb_direction(nv)
                if n_dir == "unknown":
                    continue
                for es, ev, et in existing_claims:
                    e_dir = _verb_direction(ev)
                    if e_dir == "unknown":
                        continue
                    # Same target (fuzzy match) + opposing direction
                    nt_clean = nt.strip()[:20].lower()
                    et_clean = et.strip()[:20].lower()
                    if (nt_clean in et_clean or et_clean in nt_clean) and n_dir != e_dir:
                        is_opposing = True
                        break
                if is_opposing:
                    break

            if not is_opposing:
                continue

            return {
                "belief_id": belief["id"],
                "belief_statement": belief["statement"],
                "belief_confidence": belief["confidence"],
                "similarity": sim,
            }
        return None

    # ------------------------------------------------------------------
    # Contradiction resolution gate
    # ------------------------------------------------------------------

    def _register_contradiction(self, new_statement, existing_match):
        """Register or retrieve an existing unresolved contradiction.

        Returns (record_dict, is_new).
        """
        db = self._get_queue_db()
        belief_a_id = existing_match["belief_id"]
        belief_b_id = hashlib.md5(new_statement.encode()).hexdigest()

        # Check for existing unresolved contradiction for this belief pair
        row = db.execute(
            "SELECT * FROM contradictions WHERE belief_a_id = ? AND belief_b_id = ? "
            "AND resolved = 0",
            (belief_a_id, belief_b_id),
        ).fetchone()
        if row:
            return dict(row), False

        # Register new contradiction
        record_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        cycle = self._get_consolidated_count()
        db.execute(
            "INSERT INTO contradictions "
            "(id, belief_a_id, belief_b_id, similarity, first_detected, "
            "first_detected_cycle, resolved) "
            "VALUES (?, ?, ?, ?, ?, ?, 0)",
            (record_id, belief_a_id, belief_b_id,
             existing_match["similarity"], now, cycle),
        )
        db.commit()
        record = {
            "id": record_id,
            "belief_a_id": belief_a_id,
            "belief_b_id": belief_b_id,
            "similarity": existing_match["similarity"],
            "first_detected": now,
            "first_detected_cycle": cycle,
            "resolved": 0,
        }
        logger.info(
            f"Contradiction registered: existing '{existing_match['belief_statement'][:40]}' "
            f"vs new statement (sim={existing_match['similarity']:.2f})"
        )
        return record, True

    def _evaluate_contradiction_gate(self, new_belief_data, existing_match, contradiction_record):
        """Evaluate whether a contradiction can be auto-resolved.

        Four conditions, all must pass:
        1. existing_sources >= 3 AND new_sources >= 3
        2. current_cycle - first_detected_cycle >= 2
        3. abs(existing_confidence - new_confidence) >= 0.2
        4. operator_anchored = 0 on existing belief

        Returns (can_auto: bool, failed_conditions: list[str], detail: dict).
        """
        failed = []

        # Condition 1: source counts
        existing_sources = self.semantic.get_source_count(existing_match["belief_id"])
        # New belief has no ID yet — count is always 0 at triage time
        new_sources = 0
        if existing_sources < 3 or new_sources < 3:
            failed.append(f"insufficient sources (existing={existing_sources}, new={new_sources})")

        # Condition 2: age in consolidation cycles
        current_cycle = self._get_consolidated_count()
        age_cycles = current_cycle - contradiction_record["first_detected_cycle"]
        if age_cycles < 2:
            failed.append(f"too recent (age={age_cycles} cycles, need >=2)")

        # Condition 3: confidence gap
        existing_conf = existing_match["belief_confidence"]
        new_conf_str = new_belief_data.get("confidence", "medium")
        new_source = new_belief_data.get("source", "unknown")
        new_conf = self._map_confidence(new_conf_str, new_source, is_correction=False)
        conf_gap = abs(existing_conf - new_conf)
        if conf_gap < 0.2:
            failed.append(f"confidence gap too small ({conf_gap:.2f}, need >=0.2)")

        # Condition 4: operator anchoring
        existing_belief = self.semantic.get_belief_by_id(existing_match["belief_id"])
        if existing_belief and existing_belief.get("operator_anchored", 0):
            failed.append("existing belief is operator-anchored")

        detail = {
            "existing_sources": existing_sources,
            "new_sources": new_sources,
            "age_cycles": age_cycles,
            "existing_conf": existing_conf,
            "new_conf": new_conf,
            "conf_gap": conf_gap,
        }

        can_auto = len(failed) == 0
        return can_auto, failed, detail

    def _auto_resolve_contradiction(self, new_belief_data, existing_match, contradiction_record, detail):
        """Auto-resolve a contradiction based on confidence comparison.

        - If existing conf < new conf: deprecate existing, resolution = auto_deprecated_existing
        - If new conf <= existing conf: don't store new belief, resolution = auto_rejected_new
        """
        db = self._get_queue_db()
        now = datetime.now(timezone.utc).isoformat()

        if detail["existing_conf"] < detail["new_conf"]:
            # Deprecate existing belief
            self.semantic.deprecate_belief(
                existing_match["belief_id"],
                reason=f"auto-deprecated: contradicted by higher-confidence new belief "
                       f"(existing={detail['existing_conf']:.2f}, new={detail['new_conf']:.2f})",
                source="contradiction_gate",
            )
            resolution = "auto_deprecated_existing"
            resolved_belief_id = existing_match["belief_id"]
            logger.info(
                f"Contradiction auto-resolved: deprecated existing belief "
                f"'{existing_match['belief_statement'][:40]}' (conf={detail['existing_conf']:.2f})"
            )
        else:
            # Reject new belief (don't store it)
            resolution = "auto_rejected_new"
            resolved_belief_id = None
            logger.info(
                f"Contradiction auto-resolved: rejected new belief "
                f"(new_conf={detail['new_conf']:.2f} <= existing={detail['existing_conf']:.2f})"
            )

        # Update contradiction record
        db.execute(
            "UPDATE contradictions SET resolved = 1, resolution = ?, resolved_at = ?, "
            "resolved_belief_id = ? WHERE id = ?",
            (resolution, now, resolved_belief_id, contradiction_record["id"]),
        )
        db.commit()

        return {"resolution": resolution, "resolved_belief_id": resolved_belief_id}

    # ------------------------------------------------------------------
    # Contradiction dashboard helpers
    # ------------------------------------------------------------------

    def get_contradiction_resolutions(self, limit=20):
        """Get recently resolved contradictions."""
        db = self._get_queue_db()
        rows = db.execute(
            "SELECT * FROM contradictions WHERE resolved = 1 "
            "ORDER BY resolved_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            # Enrich with belief statements
            belief_a = self.semantic.get_belief_by_id(d["belief_a_id"])
            d["belief_a_statement"] = belief_a["statement"] if belief_a else "(deleted)"
            results.append(d)
        return results

    def get_pending_contradictions(self):
        """Get unresolved contradictions."""
        db = self._get_queue_db()
        rows = db.execute(
            "SELECT * FROM contradictions WHERE resolved = 0 "
            "ORDER BY first_detected ASC",
        ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            belief_a = self.semantic.get_belief_by_id(d["belief_a_id"])
            d["belief_a_statement"] = belief_a["statement"] if belief_a else "(deleted)"
            d["belief_a_confidence"] = belief_a["confidence"] if belief_a else 0
            results.append(d)
        return results

    def restore_from_contradiction(self, contradiction_id):
        """Operator override: restore deprecated belief, mark as manual_override."""
        db = self._get_queue_db()
        row = db.execute(
            "SELECT * FROM contradictions WHERE id = ?",
            (contradiction_id,),
        ).fetchone()
        if not row:
            return False

        now = datetime.now(timezone.utc).isoformat()

        # Restore the deprecated belief if resolution was auto_deprecated_existing
        if row["resolution"] == "auto_deprecated_existing" and row["resolved_belief_id"]:
            self.semantic.restore_belief(row["resolved_belief_id"])

        db.execute(
            "UPDATE contradictions SET resolved = 1, resolution = 'manual_override', "
            "resolved_at = ? WHERE id = ?",
            (now, contradiction_id),
        )
        db.commit()
        logger.info(f"Contradiction {contradiction_id[:8]} manually overridden (restored)")
        return True

    def _duplicates_existing_observation(self, pattern_text):
        """Check if a reflection pattern duplicates an existing self-observation.

        Checks ALL observations (no limit) at 0.70 similarity threshold.
        Paraphrased duplicates typically land at 0.65-0.73, so 0.70 catches
        most while avoiding false positives on genuinely distinct patterns.
        """
        existing = self.reflective.get_relevant_warnings(limit=500)
        if not existing:
            return False

        new_emb = self.embeddings.embed(pattern_text)
        for obs in existing:
            emb = self.embeddings.embed(obs.get("pattern", ""))
            sim = self._cosine_similarity(new_emb, emb)
            if sim >= 0.70:
                return True
        return False

    def _duplicates_pending_reflection(self, pattern_text):
        """Check if a reflection pattern duplicates one already pending in the queue."""
        from memory.dedup import is_duplicate
        db = self._get_queue_db()
        rows = db.execute(
            "SELECT data FROM approval_queue "
            "WHERE category = 'reflection' AND status = 'pending' "
            "ORDER BY created_at DESC LIMIT 200"
        ).fetchall()
        if not rows:
            return False

        corpus = []
        for row in rows:
            try:
                data = json.loads(row[0]) if isinstance(row[0], str) else row[0]
                pending_pattern = data.get("pattern", "")
                if pending_pattern:
                    corpus.append(pending_pattern)
            except (json.JSONDecodeError, TypeError):
                continue

        candidate_emb = self.embeddings.embed(pattern_text)
        corpus_embs = [(self.embeddings.embed(t), t) for t in corpus]
        dup, _ = is_duplicate(candidate_emb, corpus_embs, threshold=0.85)
        return dup

    def _reinforces_existing_observation(self, pattern_text):
        """Check if a reflection reinforces (but doesn't duplicate) an existing observation.

        Reinforcement zone: 0.55-0.70 (below dedup threshold but clearly related).
        """
        existing = self.reflective.get_relevant_warnings(limit=500)
        if not existing:
            return False

        new_emb = self.embeddings.embed(pattern_text)
        for obs in existing:
            emb = self.embeddings.embed(obs.get("pattern", ""))
            sim = self._cosine_similarity(new_emb, emb)
            if 0.55 <= sim < 0.70:
                return True
        return False

    # -------------------------------------------------------------------

    def _queue_consolidation_items(self, beliefs, corrections, pushbacks, lessons, episode_id,
                                    extraction_context="operator"):
        """Queue extracted items for web-based approval with auto-triage."""
        stats = {"auto_accepted": 0, "auto_rejected": 0, "queued": 0}

        # Stamp extraction context on beliefs so triage can distinguish
        # operator-sourced from exploration-sourced
        for b in beliefs:
            b["extraction_context"] = extraction_context

        # Filter SYSTEM prompt fragments — exploration prompts prefixed [SYSTEM:
        # can leak through the LLM extraction step as belief statements.
        beliefs = [
            b for b in beliefs
            if not (
                b.get("statement", "").startswith("[SYSTEM:")
                or "[SYSTEM:" in b.get("statement", "")[:50]
            )
        ]

        all_items = (
            [("belief", b) for b in beliefs]
            + [("correction", c) for c in corrections]
            + [("pushback", p) for p in pushbacks]
            + [("lesson", l) for l in lessons]
        )

        from core.triage_adapters import _category_to_proposal

        auto_accept = self.config.get("extraction", {}).get("auto_accept_extraction", False)
        if auto_accept:
            logger.warning(
                "auto_accept_extraction is active — bypassing governance + quality gates"
            )

        for category, data in all_items:
            # Auto-accept mode — bypass triage for trusted corpus ingestion (DMS)
            if auto_accept and category != "pushback":
                decision, reason = "accept", "auto_accept_extraction"
            # Pushbacks → self-observations, not beliefs. Stay on legacy triage.
            elif category == "pushback":
                decision, reason = self._auto_triage(category, data)
            else:
                # Route beliefs, corrections, lessons through proposal gateway.
                # Adapter strategies delegate to the same _triage_* methods,
                # producing identical decisions.
                proposal = _category_to_proposal(category, data, episode_id)
                triage_result = self.gateway.submit(proposal)
                decision, reason = triage_result.decision, triage_result.reason

            if decision == "accept":
                self._process_approved_item(category, data, episode_id)
                stats["auto_accepted"] += 1
                logger.info(f"Auto-accepted {category}: {reason}")
            elif decision == "reject":
                self._record_rejection(category, data)
                if reason in self._SHADOW_LOG_REASONS:
                    self._log_rejected_belief(category, data, reason, episode_id)
                stats["auto_rejected"] += 1
                logger.info(f"Auto-rejected {category}: {reason}")
            else:
                rec = self._recommend(category, data)
                label = self._item_label(category, data, rec)
                self.queue_item(category, data, rec, label, episode_id)
                stats["queued"] += 1

        logger.info(
            f"Triage for {episode_id[:8]}: "
            f"{stats['auto_accepted']} accepted, {stats['auto_rejected']} rejected, "
            f"{stats['queued']} queued"
        )

    # -----------------------------------------------------------------------
    # Processing approved changes
    # -----------------------------------------------------------------------

    def _process_beliefs(self, beliefs, episode_id):
        """Store approved beliefs with embedding-based deduplication."""
        for belief in beliefs:
            statement = belief["statement"]
            conf_str = belief.get("confidence", "medium")
            source = belief.get("source", "unknown")
            evidence = belief.get("evidence", "")

            confidence = self._map_confidence(conf_str, source, is_correction=False)

            match = self._find_similar_belief(statement)
            if match:
                belief_id, similarity = match
                new_conf = min(0.9, confidence + 0.1)
                self.semantic.update_belief(
                    belief_id,
                    new_confidence=new_conf,
                    reason=(
                        f"Reinforced: {evidence}" if evidence
                        else "Reinforced in conversation"
                    ),
                    episode_id=episode_id,
                    source="exploration",
                )
                logger.info(
                    f"Updated existing belief {belief_id[:8]} "
                    f"(similarity: {similarity:.2f})"
                )
            else:
                _ext_ctx = belief.get("extraction_context", "operator")
                new_id = self.semantic.add_belief(
                    statement=statement,
                    confidence=confidence,
                    source_episode=episode_id,
                    supporting_evidence=[evidence] if evidence else [],
                    source="exploration",
                    source_type=self._classify_belief_type(statement, _ext_ctx),
                )
                logger.info(
                    f"New belief: '{statement[:60]}' "
                    f"(confidence: {confidence:.2f})"
                )
                # D1: Check dormant adjacency for newly created belief
                self.semantic.check_dormant_adjacency(
                    new_id, embedder=self.embeddings, trigger_type="new_belief",
                )

    def _process_corrections(self, corrections, episode_id):
        """Store approved corrections and create beliefs from them."""
        for c in corrections:
            self.episodic.add_correction(
                episode_id=episode_id,
                original_position=c["original"],
                corrected_position=c["corrected"],
                corrected_by=c.get("by", "unknown"),
                reasoning=c.get("reason"),
            )

            # Also form a belief from the corrected position
            corrected = c["corrected"]
            confidence = self._map_confidence(
                "medium", c.get("by", "unknown"), is_correction=True
            )

            match = self._find_similar_belief(corrected)
            if match:
                belief_id, similarity = match
                self.semantic.update_belief(
                    belief_id,
                    new_confidence=min(0.9, confidence + 0.15),
                    new_statement=corrected,
                    reason=f"Corrected from: {c['original']}",
                    episode_id=episode_id,
                    source="correction",
                )
            else:
                self.semantic.add_belief(
                    statement=corrected,
                    confidence=confidence,
                    source_episode=episode_id,
                    supporting_evidence=[
                        f"Correction from: {c['original']}"
                    ],
                    source="correction",
                    source_type=self._classify_belief_type(corrected),
                )

            logger.info(
                f"Correction stored: '{c['original'][:40]}' "
                f"-> '{c['corrected'][:40]}'"
            )

    def _process_pushbacks(self, pushbacks, episode_id):
        """Store approved pushbacks as self-observations (with dedup check)."""
        for p in pushbacks:
            pattern = f"Pushback received: {p['challenged']}"
            if self._duplicates_existing_observation(pattern):
                logger.info(f"Pushback dedup skipped: '{p['challenged'][:60]}'")
                continue
            self.reflective.add_observation(
                pattern=pattern,
                identified_by="evolution_engine",
                identified_in=episode_id,
                mitigation=p.get("resolution"),
                topics=[],
                status="monitoring",
            )
            logger.info(
                f"Pushback noted: '{p['challenged'][:60]}'"
            )

    def _process_lessons(self, lessons, episode_id):
        """Store approved behavioral lessons as high-confidence beliefs."""
        for l in lessons:
            principle = l["principle"]
            source = l.get("source", "unknown")
            context = l.get("context", "")

            # Lessons from corrections get high confidence — they're direct feedback
            confidence = 0.8 if source == "user" else 0.6

            match = self._find_similar_belief(principle)
            if match:
                belief_id, similarity = match
                # Reinforce existing lesson
                self.semantic.update_belief(
                    belief_id,
                    new_confidence=min(0.9, confidence + 0.1),
                    reason=f"Lesson reinforced: {context}" if context else "Lesson reinforced",
                    episode_id=episode_id,
                    source="exploration",
                )
                logger.info(
                    f"Reinforced lesson {belief_id[:8]} "
                    f"(similarity: {similarity:.2f})"
                )
            else:
                self.semantic.add_belief(
                    statement=principle,
                    confidence=confidence,
                    source_episode=episode_id,
                    supporting_evidence=[
                        f"Behavioral lesson from {source}: {context}"
                    ] if context else [f"Behavioral lesson from {source}"],
                    source="exploration",
                    source_type=self._classify_belief_type(principle),
                )
                logger.info(
                    f"New lesson: '{principle[:60]}' "
                    f"(confidence: {confidence:.2f})"
                )

    def _process_resolutions(self, resolutions, episode_id):
        """Resolve answered questions and convert answers to beliefs."""
        if not self.curiosity:
            return
        for r in resolutions:
            question_id = r["question_id"]
            answer = r["answer"]
            confidence_str = r.get("confidence", "medium")

            # Resolve the question
            self.curiosity.resolve_question(question_id, answer, answered_by="conversation")

            # Convert answer to belief
            confidence = {"low": 0.4, "medium": 0.6, "high": 0.75}.get(
                confidence_str, 0.5
            )

            match = self._find_similar_belief(answer)
            if match:
                belief_id, similarity = match
                self.semantic.update_belief(
                    belief_id,
                    new_confidence=min(0.9, confidence + 0.1),
                    reason=f"Confirmed via answered question: {r['question'][:60]}",
                    episode_id=episode_id,
                    source="exploration",
                )
                logger.info(
                    f"Resolution reinforced belief {belief_id[:8]} "
                    f"(similarity: {similarity:.2f})"
                )
            else:
                self.semantic.add_belief(
                    statement=answer,
                    confidence=confidence,
                    source_episode=episode_id,
                    supporting_evidence=[
                        f"Answered question: {r['question'][:100]}"
                    ],
                    source="exploration",
                    source_type=self._classify_belief_type(answer),
                )
                logger.info(f"Resolution created belief: '{answer[:60]}'")

    def _process_questions(self, questions, episode_id, episode_topics=None):
        """Store approved questions via CuriosityMemory."""
        if not self.curiosity:
            return
        tags = episode_topics or []
        for q in questions:
            self.curiosity.add_question(
                question=q["question"],
                question_type=q.get("type", "knowledge_gap"),
                context=q.get("context", ""),
                topic_tags=tags,
                source_episode=episode_id,
                priority=q.get("priority", "medium"),
                embedder=self.embeddings,
            )
            logger.info(f"New question stored: '{q['question'][:60]}'")

    def _maybe_generate_dream_gap(self, dream_result, component_of, dream_data):
        """Generate a bridging curiosity gap if an accepted dream is cross-domain or cross-component.

        Uses the existing BFS component_of map (computed in _run_dreams) for component checks,
        and _domain tags (set by _sample_beliefs_by_domain) for domain checks.
        Only generates one gap per accepted dream. No gaps for same-domain same-component dreams.
        """
        if not self.curiosity:
            return

        belief_a = dream_result["belief_a"]
        belief_b = dream_result["belief_b"]
        domain_a = belief_a.get("_domain", "unclassified")
        domain_b = belief_b.get("_domain", "unclassified")
        comp_a = component_of.get(belief_a["id"])
        comp_b = component_of.get(belief_b["id"])

        cross_domain = domain_a != domain_b
        cross_component = comp_a is not None and comp_b is not None and comp_a != comp_b

        if not cross_domain and not cross_component:
            return

        # Build a bridging question from the belief statements
        stmt_a = belief_a["statement"][:50]
        stmt_b = belief_b["statement"][:50]
        if cross_domain:
            gap_question = (
                f"What connects '{stmt_a}' ({domain_a}) "
                f"and '{stmt_b}' ({domain_b})?"
            )
        else:
            # Cross-component but same domain — ask about the specific concepts
            gap_question = (
                f"What connects '{stmt_a}' "
                f"and '{stmt_b}'?"
            )

        bridge_type = []
        if cross_domain:
            bridge_type.append("cross-domain")
        if cross_component:
            bridge_type.append("cross-component")

        topic_tags = [domain_a]
        if domain_a != domain_b:
            topic_tags.append(domain_b)

        self.curiosity.add_question(
            question=gap_question,
            question_type="knowledge_gap",
            context=(
                f"Dream synthesis ({', '.join(bridge_type)}) bridged: "
                f"'{dream_data['belief_a_statement'][:40]}' + "
                f"'{dream_data['belief_b_statement'][:40]}'"
            ),
            topic_tags=topic_tags,
            priority="high",
            embedder=self.embeddings,
        )
        logger.info(
            f"Dream-to-curiosity: generated {', '.join(bridge_type)} bridging gap: "
            f"'{gap_question[:60]}'"
        )

    # -----------------------------------------------------------------------
    # Belief deduplication via embeddings
    # -----------------------------------------------------------------------

    def _find_similar_belief(self, statement):
        """Find existing belief similar to statement. Returns (id, sim) or None."""
        existing = self.semantic.search_beliefs()
        if not existing:
            return None

        new_emb = self.embeddings.embed(statement)

        best_id = None
        best_sim = 0.0

        for belief in existing:
            emb = self.embeddings.embed(belief["statement"])
            sim = self._cosine_similarity(new_emb, emb)
            if sim > best_sim:
                best_sim = sim
                best_id = belief["id"]

        if best_id and best_sim >= self.similarity_threshold:
            return (best_id, best_sim)
        return None

    @staticmethod
    def _cosine_similarity(a, b):
        """Cosine similarity between two vectors."""
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(dot / norm) if norm > 0 else 0.0

    _META_PATTERNS = [
        "anima", "my behavior", "my pattern", "my reasoning", "i tend to",
        "my responses", "i notice about myself", "my approach", "my processing",
        "how i respond", "my analysis", "my tendency",
    ]
    _IDENTITY_PATTERNS = [
        "i am", "what i am", "my purpose", "my nature", "my identity",
        "my architecture", "my capabilities", "my system", "who i am",
        "my existence", "my design",
    ]

    @classmethod
    def _classify_belief_type(cls, statement, extraction_context=None):
        """Classify a belief's source type.

        Returns: "corpus" (document extraction), "external" (web/conversation),
                 "meta" (self-observation), "identity" (self-identity).
        """
        lower = statement.lower()
        if any(p in lower for p in cls._IDENTITY_PATTERNS):
            return "identity"
        if any(p in lower for p in cls._META_PATTERNS):
            return "meta"
        # Document extractions are corpus, not external
        if extraction_context in ("document", "targeted"):
            return "corpus"
        return "external"

    def _map_confidence(self, level_str, source, is_correction=False):
        """Map confidence string to float with adjustments.

        Base: low=0.3, medium=0.5, high=0.7
        +0.15 if from a correction
        +0.10 if user-stated
        Clamped to [0.2, 0.9]
        """
        base = {"low": 0.3, "medium": 0.5, "high": 0.7}.get(level_str, 0.5)
        if is_correction:
            base += 0.15
        if source == "user":
            base += 0.10
        return max(0.2, min(0.9, base))

    # -----------------------------------------------------------------------
    # Conversation preparation + truncation
    # -----------------------------------------------------------------------

    def _prepare_conversation_text(self, turns):
        """Format conversation with truncation for long sessions.

        For conversations over ~7000 tokens (~28000 chars):
        - Keep first 2 turns (establishes topic)
        - Keep last N turns that fit the budget
        - Insert omission marker between them
        """
        if not turns:
            return ""

        # Filter out SYSTEM prompt fragments before extraction
        turns = [
            t for t in turns
            if not (
                t.get("content", "").startswith("[SYSTEM:")
                or "[SYSTEM:" in t.get("content", "")[:100]
            )
        ]
        if not turns:
            return ""

        full_text = self._format_turns(turns)

        max_chars = 28000
        if len(full_text) <= max_chars:
            return full_text

        # Truncation path
        first_turns = turns[:2]
        rest = turns[2:]

        first_text = self._format_turns(first_turns)
        budget = max_chars - len(first_text) - 50  # reserve for marker

        kept = []
        running = 0
        for turn in reversed(rest):
            line = f"{turn['role'].upper()}: {turn['content']}\n"
            if running + len(line) > budget:
                break
            kept.insert(0, turn)
            running += len(line)

        omitted = len(rest) - len(kept)

        result = first_text
        if omitted > 0:
            result += f"\n[... {omitted} turns omitted ...]\n\n"
        result += self._format_turns(kept)

        return result

    @staticmethod
    def _format_turns(turns):
        """Format turn dicts into readable text."""
        lines = []
        for t in turns:
            lines.append(f"{t['role'].upper()}: {t['content']}")
        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Reflection engine — meta-analysis every N episodes
    # -----------------------------------------------------------------------

    def _run_reflection(self, queue_mode=False):
        """Cross-episode pattern detection. Runs every reflection_interval episodes."""
        try:
            print("  Running meta-reflection...", flush=True)
            summary_text = self._build_reflection_summary()

            if not summary_text:
                return

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a self-reflective AI analyzing patterns "
                        "across multiple conversations. Be honest and specific."
                    ),
                },
                {
                    "role": "user",
                    "content": self._no_think(REFLECTION_PROMPT.format(summary=summary_text)),
                },
            ]

            response = self.inference.generate_with_messages(
                messages, max_tokens=self.analysis_max_tokens, temperature=0.1, task="reinforcement",
            )

            if response.strip().upper() == "NONE":
                logger.info("Reflection: no patterns found.")
                return

            # Parse patterns
            patterns = []
            for line in response.strip().split("\n"):
                line = line.strip()
                if not line.upper().startswith("PATTERN:"):
                    continue

                remainder = line.split(":", 1)[1]
                segments = remainder.split("|")

                parts = {}
                if segments:
                    parts["pattern"] = segments[0].strip()
                for seg in segments[1:]:
                    seg = seg.strip()
                    seg_upper = seg.upper()
                    if seg_upper.startswith("EVIDENCE:"):
                        parts["evidence"] = seg.split(":", 1)[1].strip()
                    elif seg_upper.startswith("MITIGATION:"):
                        parts["mitigation"] = seg.split(":", 1)[1].strip()
                    elif seg_upper.startswith("TOPICS:"):
                        raw = seg.split(":", 1)[1].strip()
                        parts["topics"] = [
                            t.strip() for t in raw.split(",") if t.strip()
                        ]

                # Evidence gate: reject patterns without concrete evidence
                if parts.get("pattern") and not parts.get("evidence"):
                    logger.info(
                        f"Reflection dropped (no evidence): {parts['pattern'][:60]}"
                    )
                    continue

                if parts.get("pattern"):
                    patterns.append(parts)

            if not patterns:
                return

            if queue_mode:
                # Triage reflection patterns before queueing
                stats = {"auto_accepted": 0, "auto_rejected": 0, "queued": 0}
                for p in patterns:
                    decision, reason = self._auto_triage("reflection", p)
                    if decision == "accept":
                        if reason == "reinforces existing self-observation":
                            # Don't store a new row — the reinforcement signal
                            # is logged but doesn't create a duplicate entry.
                            stats["auto_accepted"] += 1
                            logger.info(f"Reflection reinforced (not stored): {reason}")
                        else:
                            self._process_approved_reflection(p, None)
                            stats["auto_accepted"] += 1
                            logger.info(f"Reflection auto-accepted: {reason}")
                    elif decision == "reject":
                        self._record_rejection("reflection", p)
                        if reason in self._SHADOW_LOG_REASONS:
                            self._log_rejected_belief("reflection", p, reason)
                        stats["auto_rejected"] += 1
                        logger.info(f"Reflection auto-rejected: {reason}")
                    else:
                        label = f"Pattern: {p['pattern'][:70]}"
                        if p.get("mitigation"):
                            label += f" (suggestion: {p['mitigation'][:40]})"
                        self.queue_item("reflection", p, True, label)
                        stats["queued"] += 1
                logger.info(
                    f"Reflection triage: {stats['auto_accepted']} accepted, "
                    f"{stats['auto_rejected']} rejected, {stats['queued']} queued"
                )
            else:
                # Interactive terminal approval
                print(
                    f"\n  === Meta-Reflection: "
                    f"{len(patterns)} pattern(s) detected ===\n"
                )
                for i, p in enumerate(patterns):
                    label = f"Flag pattern: '{p['pattern']}'"
                    if p.get("mitigation"):
                        label += f"  (suggestion: {p['mitigation']})"
                    print(f"  [{i + 1}] {label}")

                print(f"\n  [a]ccept all | [s]kip all | Enter to review each")

                try:
                    choice = input("  > ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    choice = "s"

                approved = []
                if choice == "a":
                    approved = patterns
                    print(f"  Accepted all {len(patterns)} patterns.\n")
                elif choice == "s":
                    print("  Skipped all patterns.\n")
                else:
                    for i, p in enumerate(patterns):
                        try:
                            ans = input(
                                f"  [{i + 1}] Accept? [y/n] "
                            ).strip().lower()
                        except (EOFError, KeyboardInterrupt):
                            ans = "n"
                        if ans in ("y", "yes"):
                            approved.append(p)
                    print(
                        f"  Accepted {len(approved)}/{len(patterns)} patterns.\n"
                    )

                for p in approved:
                    self.reflective.add_observation(
                        pattern=p["pattern"],
                        identified_by="reflection_engine",
                        mitigation=p.get("mitigation"),
                        topics=p.get("topics", []),
                        status="monitoring",
                    )
                    logger.info(
                        f"Reflection pattern: '{p['pattern'][:60]}'"
                    )

        except Exception as e:
            logger.warning(f"Reflection failed: {e}")

    def _dedup_edges(self, since_timestamp=None):
        """Remove duplicate edges between the same belief pair, keeping highest similarity.

        If since_timestamp is provided, only checks edges created after that time
        (scoped to current consolidation pass).
        """
        query = ("SELECT id, belief_a, belief_b, COALESCE(similarity, 0) as similarity "
                 "FROM belief_links WHERE COALESCE(active, 1) = 1")
        params = ()
        if since_timestamp:
            query += " AND created_at >= ?"
            params = (since_timestamp,)
        rows = self.semantic.db_conn.execute(query, params).fetchall()

        from collections import defaultdict
        pairs = defaultdict(list)
        for row in rows:
            pair_key = (min(row[1], row[2]), max(row[1], row[2]))
            pairs[pair_key].append((row[0], row[3]))

        deduped = 0
        for pair_key, edges in pairs.items():
            if len(edges) <= 1:
                continue
            edges.sort(key=lambda x: x[1], reverse=True)
            for link_id, _ in edges[1:]:
                self.semantic.db_conn.execute(
                    "UPDATE belief_links SET active = 0 WHERE id = ?", (link_id,)
                )
                deduped += 1

        if deduped:
            self.semantic.db_conn.commit()
            logger.info(f"Edge dedup: deactivated {deduped} duplicate edge(s)")

    def _build_reflection_summary(self):
        """Build text summary of recent consolidated episodes for reflection.

        Includes: summaries, corrections, mid-conversation insights, and
        recent beliefs — giving the reflection pass a complete picture.
        """
        episodes = self.episodic.get_recent_episodes(
            limit=max(self.reflection_interval, 5)
        )

        consolidated = [
            ep for ep in episodes
            if ep.get("summarized") == 1 and ep.get("summary")  # summarized=1 means consolidated
        ]

        if len(consolidated) < 2:
            return ""

        parts = []
        for ep in consolidated:
            summary = ep.get("summary", "No summary")
            topics = json.loads(ep.get("topics") or "[]")
            importance = ep.get("importance", 0.5)

            line = f"- {summary}"
            if topics:
                line += f" (topics: {', '.join(topics)})"
            if importance >= 0.8:
                line += " [HIGH IMPORTANCE]"
            parts.append(line)

            # Include corrections for context
            corrections = self.episodic.get_corrections(episode_id=ep["id"])
            for c in corrections[:2]:
                parts.append(
                    f"  Correction: '{c['original_position'][:60]}' "
                    f"-> '{c['corrected_position'][:60]}'"
                )

            # Include mid-conversation insights if any
            insights = json.loads(ep.get("key_insights") or "[]")
            for insight in insights[:2]:
                parts.append(f"  Insight: {insight[:80]}")

        # Include recent beliefs for pattern detection across conversations
        beliefs = self.semantic.search_beliefs()
        if beliefs:
            parts.append("\nCurrent beliefs:")
            for b in beliefs[:8]:
                parts.append(
                    f"  - [{b['confidence']:.1f}] {b['statement'][:80]}"
                )

        return "\n".join(parts)

    # -----------------------------------------------------------------------
    # Dreams — cross-episode belief linking
    # -----------------------------------------------------------------------

    def _assign_belief_domain(self, belief):
        """Assign a semantic domain label to a belief for dream stratification.

        Sources (in priority order):
        1. topics field (if populated by domain classifier)
        2. tree_paths (hierarchical placement from knowledge tree)
        3. tag registry (structural tags from tag backfill)
        4. source_type fallback (meta/identity)
        5. "other"

        No hardcoded domain keywords — all classification is corpus-driven.
        """
        import json as _json

        # 1. Topics field (legacy, still works if populated)
        topics = belief.get("topics")
        if topics:
            if isinstance(topics, str):
                try:
                    topics = _json.loads(topics)
                except (ValueError, TypeError):
                    topics = []
            if topics and isinstance(topics, list) and topics[0] != "operator":
                return topics[0].lower().strip()

        # 2. Tree paths — use the deepest node name as domain
        tree_paths = belief.get("tree_paths")
        if tree_paths:
            if isinstance(tree_paths, str):
                try:
                    tree_paths = _json.loads(tree_paths)
                except (ValueError, TypeError):
                    tree_paths = []
            if tree_paths and isinstance(tree_paths, list):
                # Each path is [root, ..., leaf]. Use the second level as domain
                # (first level is too broad, deepest is too specific)
                for path in tree_paths:
                    if isinstance(path, list) and len(path) >= 2:
                        return path[1].lower().strip()
                    elif isinstance(path, list) and len(path) == 1:
                        return path[0].lower().strip()

        # 3. Tag registry — find this belief's primary tag
        bid = belief.get("id")
        if bid and hasattr(self, 'semantic') and self.semantic and self.semantic.db_conn:
            try:
                row = self.semantic.db_conn.execute(
                    "SELECT tr.name FROM belief_tags bt "
                    "JOIN tag_registry tr ON bt.tag_id = tr.id "
                    "WHERE bt.belief_id = ? ORDER BY bt.confidence DESC LIMIT 1",
                    (bid,),
                ).fetchone()
                if row:
                    return row[0].lower().strip()
            except Exception:
                pass

        # 4. Source type fallback
        st = belief.get("source_type", "")
        if st in ("meta", "identity"):
            return "identity" if st == "identity" else "meta"

        return "other"

    def _dream_pair_exists_in_db(self, belief_a_id, belief_b_id):
        """Check if a dream already exists with these two as parents.

        Checks both orderings (a,b) and (b,a) since parent assignment
        order is not guaranteed to be consistent.

        Returns True if a non-deprecated dream exists for this pair.
        """
        try:
            row = self.semantic.db_conn.execute(
                "SELECT 1 FROM beliefs "
                "WHERE generation_type IN ('dream', 'triplet') "
                "AND ((parent_a = ? AND parent_b = ?) OR (parent_a = ? AND parent_b = ?)) "
                "AND COALESCE(deprecated, 0) = 0 "
                "LIMIT 1",
                (belief_a_id, belief_b_id, belief_b_id, belief_a_id),
            ).fetchone()
            return row is not None
        except Exception:
            return False

    def _sample_beliefs_by_domain(self, all_beliefs, per_domain=10):
        """Group beliefs by domain, sample N per domain with weighted randomization.

        Bridge priority beliefs are force-included regardless of per_domain cap.
        Sampling uses confidence-weighted random selection to avoid deterministic
        pair repetition across consolidations.
        """
        import random
        from collections import defaultdict
        domain_buckets = defaultdict(list)
        for b in all_beliefs:
            # Skip system prompt fragments that leaked into beliefs
            stmt = b.get("statement", "")
            if stmt.startswith("[SYSTEM:") or "[SYSTEM:" in stmt[:50]:
                logger.warning(f"SYSTEM belief reached dream sampling despite extraction fix: {b.get('id', '?')[:12]}")
                continue
            domain_buckets[self._assign_belief_domain(b)].append(b)

        # Domain distribution sanity check — catch classifier collapse
        _plugin = self.config.get("product", {}).get("mode", "unknown") if self.config else "unknown"
        total_classified = sum(len(v) for v in domain_buckets.values())
        if total_classified > 0:
            for dom, beliefs in domain_buckets.items():
                ratio = len(beliefs) / total_classified
                if ratio > 0.75 and dom not in ("other", "unclassified"):
                    logger.warning(
                        f"Dreams [{_plugin}]: domain '{dom}' dominates at {ratio:.0%} "
                        f"({len(beliefs)}/{total_classified}) — classifier may be collapsing"
                    )
            other_ratio = len(domain_buckets.get("other", [])) / total_classified
            if other_ratio > 0.60:
                logger.warning(
                    f"Dreams [{_plugin}]: {other_ratio:.0%} beliefs unclassified "
                    f"— domain classifier may not be active"
                )

        sampled_ids = set()
        sampled = []

        # Force-include bridge priority beliefs first
        if self._bridge_priority_ids:
            for b in all_beliefs:
                if b["id"] in self._bridge_priority_ids and b["id"] not in sampled_ids:
                    b["_domain"] = self._assign_belief_domain(b)
                    sampled.append(b)
                    sampled_ids.add(b["id"])

        # Weighted random sampling from domain buckets (confidence as weight)
        for domain, beliefs in domain_buckets.items():
            available = [b for b in beliefs if b["id"] not in sampled_ids]
            n = min(per_domain, len(available))
            if n == 0:
                continue
            weights = [max((b.get("confidence", 0.0) or 0.0), 0.05) for b in available]
            chosen = []
            pool = list(available)
            pool_weights = list(weights)
            for _ in range(n):
                if not pool:
                    break
                selected = random.choices(pool, weights=pool_weights, k=1)[0]
                chosen.append(selected)
                idx = pool.index(selected)
                pool.pop(idx)
                pool_weights.pop(idx)
            for b in chosen:
                b["_domain"] = domain
                sampled.append(b)
                sampled_ids.add(b["id"])

        # Force-include at least 1 isolated belief (degree 0) per consolidation.
        # Isolated corpus/web beliefs never get dream connections without this.
        try:
            import random as _rand
            isolated = [b for b in all_beliefs
                        if b["id"] not in sampled_ids
                        and b.get("source_type") != "synthesis"]
            # Filter to truly isolated (no edges)
            isolated_real = []
            for b in isolated[:50]:  # Check up to 50
                deg = self.semantic.db_conn.execute(
                    "SELECT COUNT(*) as cnt FROM belief_links WHERE "
                    "(belief_a=? OR belief_b=?) AND active=1",
                    (b["id"], b["id"]),
                ).fetchone()["cnt"]
                if deg == 0:
                    isolated_real.append(b)
            if isolated_real:
                pick = _rand.choice(isolated_real)
                pick["_domain"] = self._assign_belief_domain(pick)
                sampled.append(pick)
                sampled_ids.add(pick["id"])
                logger.info(f"Dreams: force-included isolated belief {pick['id'][:8]}")
        except Exception:
            pass

        return sampled

    def _count_cross_domain_bridges(self):
        """Count active belief_links where endpoints have different domains."""
        try:
            db = self.semantic.db_conn
            rows = db.execute(
                "SELECT bl.belief_a, bl.belief_b FROM belief_links bl "
                "WHERE COALESCE(bl.active, 1) = 1"
            ).fetchall()
            if not rows:
                return 0
            count = 0
            for r in rows:
                da = self._get_belief_domain_from_db(r["belief_a"])
                db_domain = self._get_belief_domain_from_db(r["belief_b"])
                if da != db_domain:
                    count += 1
            return count
        except Exception:
            return 0

    def _get_belief_domain_from_db(self, belief_id):
        """Quick domain lookup for a belief by ID.

        Priority: topics field → keyword classification → source_type fallback.
        """
        row = self.semantic.db_conn.execute(
            "SELECT source_type, topics, statement FROM beliefs WHERE id = ?",
            (belief_id,),
        ).fetchone()
        if not row:
            return "unknown"
        # Try topics field first
        try:
            topics = json.loads(row["topics"] or "[]")
            if topics:
                return str(topics[0])
        except (json.JSONDecodeError, TypeError):
            pass
        # Fallback: keyword classification on statement
        stmt = row.get("statement", "")
        if stmt:
            domain = self._assign_belief_domain({"statement": stmt, "source_type": row.get("source_type", "")})
            if domain != "other":
                return domain
        # Last resort: source_type for meta/identity, else unclassified
        st = row["source_type"] or ""
        if st in ("meta", "identity"):
            return st
        return "unclassified"

    def _find_bridge_belief_ids(self):
        """Return belief IDs that are endpoints of cross-domain bridge edges."""
        ids = set()
        try:
            rows = self.semantic.db_conn.execute(
                "SELECT belief_a, belief_b FROM belief_links "
                "WHERE COALESCE(active, 1) = 1"
            ).fetchall()
            for r in rows:
                da = self._get_belief_domain_from_db(r["belief_a"])
                db_val = self._get_belief_domain_from_db(r["belief_b"])
                if da != db_val:
                    ids.add(r["belief_a"])
                    ids.add(r["belief_b"])
        except Exception:
            pass
        return ids

    def _run_dreams(self, queue_mode=False):
        """Find beliefs from different episodes that connect in novel ways.

        Domain-stratified sampling: groups beliefs by domain, samples top N
        per domain, then generates candidate pairs prioritizing cross-domain
        combinations to build bridges between knowledge clusters.
        """
        try:
            # Reset per-cycle triplet participation counter
            self._triplet_participation_count = {}

            if not hasattr(self, '_recent_dream_rejects'):
                self._recent_dream_rejects = []

            all_beliefs = self.semantic.search_beliefs(limit=200)
            all_beliefs = [b for b in all_beliefs if not b.get("operator_model")]

            if len(all_beliefs) < self.min_beliefs_for_dreams:
                logger.info(f"Dreams: {len(all_beliefs)} beliefs, need {self.min_beliefs_for_dreams}. Skipping.")
                return

            print("  Dreaming...", flush=True)

            # Snapshot existing edges for novel_edge_ratio tracking
            _pre_edge_pairs = set()
            try:
                _edge_rows = self.semantic.db_conn.execute(
                    "SELECT belief_a, belief_b FROM belief_links WHERE COALESCE(active, 1) = 1"
                ).fetchall()
                for _er in _edge_rows:
                    _pre_edge_pairs.add(frozenset({_er["belief_a"], _er["belief_b"]}))
            except Exception:
                pass

            _edge_stats = {
                "dream_pair_novel": 0, "dream_pair_reinforcing": 0,
                "triplet_novel": 0, "triplet_reinforcing": 0,
                "exploration_novel": 0, "exploration_reinforcing": 0,
            }
            self._novel_edge_sources = {}
            self._pair_frontier_category = {}

            # ── Rust-driven sampling + pairing ──
            # Build DreamBelief list for Rust
            core = self.semantic._engine
            import json as _json, time as _time

            # Compute degrees
            belief_degrees = {}
            for b in all_beliefs:
                bid = b["id"]
                if bid not in belief_degrees:
                    try:
                        row = self.semantic.db_conn.execute(
                            "SELECT COUNT(*) FROM belief_links "
                            "WHERE (belief_a = ? OR belief_b = ?) AND COALESCE(active, 1) = 1",
                            (bid, bid),
                        ).fetchone()
                        belief_degrees[bid] = row[0] if row else 0
                    except Exception:
                        belief_degrees[bid] = 0

            # Build component map
            component_of = {}
            adj_local = {b["id"]: set() for b in all_beliefs}
            for b in all_beliefs:
                links = self.semantic.get_belief_links(b["id"])
                for link in links:
                    other = link["belief_b"] if link["belief_a"] == b["id"] else link["belief_a"]
                    if other in adj_local:
                        adj_local[b["id"]].add(other)
                        adj_local[other].add(b["id"])
            comp_visited = set()
            comp_id = 0
            for node in adj_local:
                if node in comp_visited:
                    continue
                queue_bfs = [node]
                comp_visited.add(node)
                while queue_bfs:
                    current = queue_bfs.pop(0)
                    component_of[current] = comp_id
                    for nb in adj_local[current]:
                        if nb not in comp_visited:
                            comp_visited.add(nb)
                            queue_bfs.append(nb)
                comp_id += 1

            # Capped nodes (rc >= 5)
            capped_nodes = set()
            try:
                cap_rows = self.semantic.db_conn.execute(
                    "SELECT belief_a, belief_b FROM belief_links "
                    "WHERE reinforced_count >= 5 AND COALESCE(active, 1) = 1"
                ).fetchall()
                for row in cap_rows:
                    capped_nodes.add(row["belief_a"])
                    capped_nodes.add(row["belief_b"])
            except Exception:
                pass

            # Existing dream pairs (DB-backed reuse prevention)
            existing_dream_pairs = set()
            try:
                dp_rows = self.semantic.db_conn.execute(
                    "SELECT parent_a, parent_b FROM beliefs "
                    "WHERE generation_type IN ('dream', 'triplet') AND COALESCE(deprecated, 0) = 0"
                ).fetchall()
                for row in dp_rows:
                    a, b = row["parent_a"] or "", row["parent_b"] or ""
                    if a and b:
                        pk = ":".join(sorted([a, b]))
                        existing_dream_pairs.add(pk)
            except Exception:
                pass

            # Existing links
            existing_links = set()
            try:
                el_rows = self.semantic.db_conn.execute(
                    "SELECT belief_a, belief_b FROM belief_links WHERE COALESCE(active, 1) = 1"
                ).fetchall()
                for row in el_rows:
                    pk = ":".join(sorted([row["belief_a"], row["belief_b"]]))
                    existing_links.add(pk)
            except Exception:
                pass

            # Bridge priority IDs
            bridge_ids = self._bridge_priority_ids if hasattr(self, '_bridge_priority_ids') else set()

            # Build Rust-compatible belief list
            _core_sims = {}
            try:
                _cs_rows = self.semantic.db_conn.execute(
                    "SELECT id, core_similarity, tree_paths FROM beliefs "
                    "WHERE COALESCE(deprecated,0)=0 AND core_similarity IS NOT NULL"
                ).fetchall()
                for _csr in _cs_rows:
                    _core_sims[_csr["id"]] = _csr["core_similarity"] or 0.0
            except Exception:
                pass

            dream_beliefs = []
            for b in all_beliefs:
                topics = b.get("topics")
                if isinstance(topics, str):
                    try:
                        topics = _json.loads(topics)
                    except Exception:
                        topics = []
                tree_paths = []
                try:
                    tp_raw = b.get("tree_paths")
                    if tp_raw:
                        tp = _json.loads(tp_raw) if isinstance(tp_raw, str) else tp_raw
                        tree_paths = [p[0] for p in tp if p] if tp else []
                except Exception:
                    pass

                dream_beliefs.append({
                    "id": b["id"] or "",
                    "statement": b.get("statement") or "",
                    "domain": self._assign_belief_domain(b),
                    "confidence": float(b.get("confidence") or 0.5),
                    "source_type": b.get("source_type") or "corpus",
                    "epistemic_class": b.get("epistemic_class") or "corpus",
                    "belief_status": b.get("belief_status") or "active",
                    "generation_type": b.get("generation_type") or "",
                    "abstraction_depth": int(b.get("abstraction_depth") or 0),
                    "degree": belief_degrees.get(b["id"], 0),
                    "source_episodes": _json.loads(b.get("source_episodes") or "[]") if isinstance(b.get("source_episodes"), str) else (b.get("source_episodes") or []),
                    "core_similarity": float(_core_sims.get(b["id"], 0.5)),
                    "tree_root_ids": tree_paths,
                    "operator_anchored": bool(b.get("operator_anchored")),
                    "is_bridge_priority": b["id"] in bridge_ids,
                    "embedding_index": 0,
                })

            dream_config = {
                "similarity_min": self.dream_similarity_min,
                "similarity_max": self.dream_similarity_max,
                "max_pairs": self.max_dream_pairs,
                "min_score": self.min_dream_score,
                "per_domain_sample": 10,
                "max_domain_pair_fraction": self.max_domain_pair_fraction,
                "frontier_degree_max": self.frontier_degree_max,
                "operator_degree_cap": 15,
                "web_web_pair_weight": getattr(self, '_web_web_pair_weight', 1.0),
                "dream_eligibility_floor": getattr(self, '_dream_eligibility_floor', 0.0),
            }

            # Rust sampling
            import random
            sampled_indices = core.dream_sample_beliefs(dream_beliefs, dream_config, random.randint(0, 2**63))
            sampled = [all_beliefs[i] for i in sampled_indices]
            for idx, si in enumerate(sampled_indices):
                sampled[idx]["_domain"] = dream_beliefs[si]["domain"]

            logger.info(
                f"Dreams: {len(sampled)} beliefs sampled from "
                f"{len(set(b['_domain'] for b in sampled))} domains "
                f"(of {len(all_beliefs)} total)."
            )

            if not sampled:
                logger.info("Dreams: no beliefs sampled.")
                return

            # Embed sampled beliefs
            embeddings = [self.embeddings.embed(b["statement"]) for b in sampled]

            # Compute similarity matrix for Rust
            import numpy as np
            n = len(sampled)
            sim_matrix = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        sim_matrix.append(1.0)
                    else:
                        norm = np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                        sim_matrix.append(float(np.dot(embeddings[i], embeddings[j]) / norm) if norm > 0 else 0.0)

            # Build context for Rust
            dream_context = {
                "component_of": {k: int(v) for k, v in component_of.items()},
                "reinforced_counts": {},
                "existing_dream_pairs": list(existing_dream_pairs),
                "existing_links": list(existing_links),
                "capped_nodes": list(capped_nodes),
            }

            # Rust pair generation + scoring + selection
            core.dream_tick_cooldowns()
            candidates_raw = core.dream_generate_pairs(
                dream_beliefs, sampled_indices, sim_matrix, dream_config, dream_context
            )

            if not candidates_raw:
                logger.info("Dreams: no candidate pairs found.")
                return

            # Record domain pairs for diversity tracking
            domain_pairs_used = list(set(
                ":".join(sorted([c["domain_a"], c["domain_b"]])) for c in candidates_raw
            ))
            core.dream_record_domain_pairs(domain_pairs_used)

            # Convert Rust candidates to legacy tuple format for dispatch loop
            candidates = [
                (c["belief_a_idx"], c["belief_b_idx"], c["similarity"], c["score"],
                 c["cross_cluster"], c["cross_domain"], c["frontier_category"])
                for c in candidates_raw
            ]

            # Store frontier categories
            for c in candidates_raw:
                pk = frozenset({c["belief_a_id"], c["belief_b_id"]})
                self._pair_frontier_category[pk] = c["frontier_category"]

            # Log selection stats
            frontier_count = sum(1 for c in candidates_raw if c["frontier_category"] == "frontier")
            mixed_count = sum(1 for c in candidates_raw if c["frontier_category"] == "mixed")
            hub_count = sum(1 for c in candidates_raw if c["frontier_category"] == "hub")
            logger.info(
                f"Dreams: {len(candidates)} pair(s) selected from {len(candidates_raw)} eligible "
                f"(domain pairs: {len(domain_pairs_used)})."
            )
            logger.info(
                f"Dreams: frontier categories: {frontier_count} frontier, "
                f"{mixed_count} mixed, {hub_count} hub"
            )

            pre_bridge_count = self._count_cross_domain_bridges()
            dream_results = []
            for idx_a, idx_b, sim, _score, _xcluster, _xdomain, _fcat in candidates:
                belief_a = sampled[idx_a]
                belief_b = sampled[idx_b]

                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a creative thinker finding connections "
                            "between ideas. Be specific and insightful."
                        ),
                    },
                    {
                        "role": "user",
                        "content": self._no_think(DREAMS_PROMPT.format(
                            belief_a=belief_a["statement"],
                            belief_b=belief_b["statement"],
                        )),
                    },
                ]

                response = self.inference.generate_with_messages(
                    messages, max_tokens=self.analysis_max_tokens, temperature=0.1, task="dreams",
                )
                if not response:
                    logger.warning("Dream synthesis: no response from model")
                    continue

                parsed = self._parse_dream_response(response)
                if parsed:
                    parsed["belief_a"] = belief_a
                    parsed["belief_b"] = belief_b
                    parsed["similarity"] = sim
                    dream_results.append(parsed)
                    dom_a = belief_a.get("_domain", "?")
                    dom_b = belief_b.get("_domain", "?")
                    logger.info(
                        f"Dream connection: '{belief_a['statement'][:40]}' <-> "
                        f"'{belief_b['statement'][:40]}' "
                        f"(sim={sim:.2f}, {dom_a} ↔ {dom_b})"
                    )

            if not dream_results:
                logger.info("Dreams: no valid connections found.")
                return

            # Dedup: remove near-duplicate inferences within the batch
            # AND against existing beliefs in the graph.
            # Prevents the same bridge from being stated 3 different ways.
            dedup_threshold = self.config.get("dreams", {}).get("dedup_threshold", 0.92)
            deduped_results = []
            dedup_skipped = 0
            kept_embeddings = []
            for d in dream_results:
                inference = d.get("inference", "")
                if not inference:
                    continue
                try:
                    emb = self.embeddings.embed(inference)
                    # Check against siblings already kept in this batch
                    is_dup = False
                    for kept_emb in kept_embeddings:
                        sim = float(sum(a * b for a, b in zip(emb, kept_emb)) /
                              (sum(a**2 for a in emb)**0.5 * sum(b**2 for b in kept_emb)**0.5 + 1e-9))
                        if sim > dedup_threshold:
                            is_dup = True
                            break
                    if not is_dup:
                        # Check against existing beliefs in the graph via text similarity
                        existing_match = self._find_similar_belief(inference)
                        if existing_match:
                            _, match_sim = existing_match
                            if match_sim > dedup_threshold:
                                is_dup = True
                    if is_dup:
                        dedup_skipped += 1
                        continue
                    kept_embeddings.append(emb)
                    deduped_results.append(d)
                except Exception as _dedup_err:
                    # Embedding failed — reject for safety, don't skip dedup
                    dedup_skipped += 1
                    self._cycle_degraded = getattr(self, '_cycle_degraded', {})
                    self._cycle_degraded["dedup_embed_failures"] = self._cycle_degraded.get("dedup_embed_failures", 0) + 1
                    logger.warning(f"Dream dedup embedding failed — skipping dream for safety: {_dedup_err}")

            if dedup_skipped:
                logger.info(f"Dreams: {dedup_skipped} near-duplicate(s) removed from batch of {len(dream_results)}")
            dream_results = deduped_results

            if not dream_results:
                logger.info("Dreams: all results were duplicates.")
                return

            if queue_mode:
                # Triage dreams before queueing
                stats = {"auto_accepted": 0, "auto_rejected": 0, "queued": 0,
                         "novel_edges": 0, "reinforcing_edges": 0}
                _graph_context = {
                    "component_of": component_of,
                    "belief_degrees": belief_degrees,
                }
                for d in dream_results:
                    dream_data = {
                        "belief_a_id": d["belief_a"]["id"],
                        "belief_a_statement": d["belief_a"]["statement"],
                        "belief_b_id": d["belief_b"]["id"],
                        "belief_b_statement": d["belief_b"]["statement"],
                        "connection": d["connection"],
                        "inference": d["inference"],
                        "similarity": d["similarity"],
                        "_graph_context": _graph_context,
                        "_domain_a": d["belief_a"].get("_domain"),
                        "_domain_b": d["belief_b"].get("_domain"),
                    }
                    pair_key = frozenset({dream_data["belief_a_id"], dream_data["belief_b_id"]})

                    # --- Dedup against recently rejected dream inferences ---
                    # Skip dreams similar (>0.85) to recent rejects without LLM call.
                    _skip_as_recent_reject = False
                    if self._recent_dream_rejects and hasattr(self, 'embeddings') and self.embeddings:
                        try:
                            _cand_emb = self.embeddings.embed(dream_data.get("inference", ""))
                            for _rej_emb in self._recent_dream_rejects:
                                _norm = np.linalg.norm(_cand_emb) * np.linalg.norm(_rej_emb)
                                if _norm > 0:
                                    _sim = float(np.dot(_cand_emb, _rej_emb) / _norm)
                                    if _sim > 0.85:
                                        _skip_as_recent_reject = True
                                        stats["auto_rejected"] += 1
                                        self._log_dream_triage(dream_data, "reject", f"similar to recent reject (sim={_sim:.2f})")
                                        logger.info(f"Dream skipped (similar to recent reject, sim={_sim:.2f}): {dream_data.get('inference', '')[:60]}")
                                        break
                        except Exception:
                            pass
                    if _skip_as_recent_reject:
                        continue

                    decision, reason = self._auto_triage("dream", dream_data)
                    # Log ALL triage decisions (accept, reject, queue) for health metrics
                    self._log_dream_triage(dream_data, decision, reason)
                    if decision == "accept":
                        from reflection.dream_transaction import DreamTransaction
                        txn = DreamTransaction(self.semantic, self.embeddings)
                        txn.expect(belief_created=True, link_count=3)
                        # Defer curiosity gap to post-commit
                        _d_ref, _co_ref, _dd_ref = d, component_of, dream_data
                        txn.defer_callback(
                            lambda _d=_d_ref, _co=_co_ref, _dd=_dd_ref:
                                self._maybe_generate_dream_gap(_d, _co, _dd)
                        )
                        self._process_approved_dream(dream_data, semantic_override=txn)
                        if txn.commit():
                            stats["auto_accepted"] += 1
                            # Track novel vs reinforcing edges (snapshot-based)
                            if pair_key not in _pre_edge_pairs:
                                stats["novel_edges"] += 1
                                _edge_stats["dream_pair_novel"] += 1
                                self._novel_edge_sources[pair_key] = "dream_pair"
                            else:
                                stats["reinforcing_edges"] += 1
                                _edge_stats["dream_pair_reinforcing"] += 1
                            logger.info(f"Dream auto-accepted: {reason}")
                        else:
                            stats["auto_rejected"] += 1
                            logger.error(
                                f"Dream transaction failed, rolled back: "
                                f"{dream_data.get('inference', '')[:60]}"
                            )
                    elif decision == "reject":
                        # Route hypothesis candidates to hypothesis pipeline instead of discarding
                        if "hypothesis candidate" in reason:
                            # Route to hypothesis pipeline via callback (set by plugin)
                            handler = getattr(self, 'dream_rejection_handler', None)
                            if handler:
                                try:
                                    handler(dream_data, reason)
                                    stats.setdefault("routed_to_hypothesis", 0)
                                    stats["routed_to_hypothesis"] += 1
                                    logger.info(f"Dream routed to hypothesis: {reason}")
                                except Exception as he:
                                    logger.warning(f"Dream→hypothesis routing failed: {he}")
                        else:
                            self._record_rejection("dream", dream_data)
                            if reason in self._SHADOW_LOG_REASONS:
                                self._log_rejected_belief("dream", dream_data, reason)
                        # Store rejected dream embedding in ring buffer for dedup
                        try:
                            if hasattr(self, 'embeddings') and self.embeddings:
                                _rej_emb = self.embeddings.embed(dream_data.get("inference", ""))
                                self._recent_dream_rejects.append(_rej_emb)
                                if len(self._recent_dream_rejects) > 30:
                                    self._recent_dream_rejects = self._recent_dream_rejects[-30:]
                        except Exception:
                            pass
                        # Register pair cooldown so this pair is skipped for N consolidations
                        self._dream_pair_cooldown[pair_key] = self.dream_pair_cooldown
                        stats["auto_rejected"] += 1
                        logger.info(f"Dream auto-rejected: {reason} (cooldown={self.dream_pair_cooldown})")
                    else:
                        label = (
                            f"Dream: \"{d['belief_a']['statement'][:40]}\" + "
                            f"\"{d['belief_b']['statement'][:40]}\" = {d['inference'][:50]}"
                        )
                        dream_rec = d.get("similarity", 0.5) >= 0.50
                        self.queue_item("dream", dream_data, dream_rec, label)
                        stats["queued"] += 1
                logger.info(
                    f"Dream triage: {stats['auto_accepted']} accepted, "
                    f"{stats['auto_rejected']} rejected, {stats['queued']} queued"
                )
            else:
                # Interactive terminal approval
                print(
                    f"\n  === Dreams: "
                    f"{len(dream_results)} connection(s) found ===\n"
                )
                for i, d in enumerate(dream_results):
                    print(
                        f"  [{i + 1}] \"{d['belief_a']['statement'][:60]}\"\n"
                        f"      + \"{d['belief_b']['statement'][:60]}\"\n"
                        f"      = {d['inference']}"
                    )

                print(f"\n  [a]ccept all | [s]kip all | Enter to review each")

                try:
                    choice = input("  > ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    choice = "s"

                approved = []
                if choice == "a":
                    approved = dream_results
                    print(f"  Accepted all {len(dream_results)} connections.\n")
                elif choice == "s":
                    print("  Skipped all connections.\n")
                else:
                    for i, d in enumerate(dream_results):
                        try:
                            ans = input(
                                f"  [{i + 1}] Accept? [y/n] "
                            ).strip().lower()
                        except (EOFError, KeyboardInterrupt):
                            ans = "n"
                        if ans in ("y", "yes"):
                            approved.append(d)
                    print(
                        f"  Accepted {len(approved)}/{len(dream_results)} connections.\n"
                    )

                # Store approved dreams
                for d in approved:
                    dream_data = {
                        "belief_a_id": d["belief_a"]["id"],
                        "belief_a_statement": d["belief_a"]["statement"],
                        "belief_b_id": d["belief_b"]["id"],
                        "belief_b_statement": d["belief_b"]["statement"],
                        "connection": d["connection"],
                        "inference": d["inference"],
                        "similarity": d["similarity"],
                    }
                    self._process_approved_dream(dream_data)

            # ── Bridge milestone detection ─────────────────────────
            if pre_bridge_count == 0:
                post_bridge_count = self._count_cross_domain_bridges()
                if post_bridge_count > 0:
                    # Find the bridge beliefs and flag for priority sampling
                    bridge_ids = self._find_bridge_belief_ids()
                    self._bridge_priority_ids.update(bridge_ids)
                    logger.info(
                        f"MILESTONE: First cross-domain bridge formed "
                        f"({post_bridge_count} bridge(s) detected). "
                        f"Flagged {len(bridge_ids)} beliefs for priority sampling."
                    )
                    print(
                        f"  *** MILESTONE: First cross-domain bridge! "
                        f"Cross-domain triplets now eligible. ***",
                        flush=True,
                    )

            # ── Triplet synthesis pass ──────────────────────────────
            self._run_triplet_dreams(sampled, embeddings, queue_mode,
                                     _pre_edge_pairs=_pre_edge_pairs,
                                     _edge_stats=_edge_stats,
                                     belief_degrees=belief_degrees)

            # ── Broadened novel_edge_ratio (all sources) ─────────────
            total_novel = (_edge_stats["dream_pair_novel"]
                          + _edge_stats["triplet_novel"]
                          + _edge_stats["exploration_novel"])
            total_reinforcing = (_edge_stats["dream_pair_reinforcing"]
                                + _edge_stats["triplet_reinforcing"]
                                + _edge_stats["exploration_reinforcing"])
            total_edges = total_novel + total_reinforcing
            novel_ratio = total_novel / total_edges if total_edges > 0 else 0.0

            # Compute largest_component_size and ratio from the BFS component map
            if component_of:
                from collections import Counter as _Counter
                _comp_sizes = _Counter(component_of.values())
                largest_component_size = max(_comp_sizes.values())
            else:
                largest_component_size = 0

            # largest_component_ratio: fraction of non-dormant nodes in largest component
            total_non_dormant = len([
                b for b in sampled if not b.get("is_dormant")
            ]) if sampled else 0
            largest_component_ratio = (
                largest_component_size / total_non_dormant
                if total_non_dormant > 0 else 0.0
            )

            # Bridge attribution: check which novel edges are bridges
            # A bridge edge connects two nodes that would be in different components without it
            bridge_stats = {"dream_pair": 0, "triplet": 0, "exploration": 0}
            if hasattr(self, '_novel_edge_sources'):
                for pair_key, source in self._novel_edge_sources.items():
                    # Check if removing this edge would disconnect its endpoints
                    # Simplified: if either endpoint has degree 1, it's a bridge
                    a_id, b_id = tuple(pair_key)
                    a_comp = component_of.get(a_id)
                    b_comp = component_of.get(b_id)
                    if a_comp != b_comp or a_comp is None or b_comp is None:
                        bridge_stats[source] = bridge_stats.get(source, 0) + 1

            # Edge weight cap summary
            edges_at_cap = 0
            cap_domain_pairs = []
            try:
                cap_rows = self.semantic.db_conn.execute(
                    "SELECT bl.id, bl.belief_a, bl.belief_b FROM belief_links bl "
                    "WHERE COALESCE(bl.active, 1) = 1 AND COALESCE(bl.reinforced_count, 0) >= 5"
                ).fetchall()
                edges_at_cap = len(cap_rows)
                for cr in cap_rows:
                    ba = self.semantic.get_belief_by_id(cr["belief_a"])
                    bb = self.semantic.get_belief_by_id(cr["belief_b"])
                    if ba and bb:
                        da = ba.get("domain") or self._assign_belief_domain(ba) if ba else "?"
                        db = bb.get("domain") or self._assign_belief_domain(bb) if bb else "?"
                        dp = f"{da}↔{db}"
                        if dp not in cap_domain_pairs:
                            cap_domain_pairs.append(dp)
            except Exception:
                pass

            # Item 3: Exploration contributes via dream pool, not direct edges
            exploration_note = " (exploration: indirect via dream pool only)" if _edge_stats["exploration_novel"] == 0 else ""

            logger.info(
                f"Dream triage summary: "
                f"novel_edge_ratio={novel_ratio:.2f} "
                f"({total_novel} novel: "
                f"{_edge_stats['dream_pair_novel']} dream_pair, "
                f"{_edge_stats['triplet_novel']} triplet, "
                f"{_edge_stats['exploration_novel']} exploration{exploration_note}; "
                f"{total_reinforcing} reinforcing) "
                f"bridges_by_source=({bridge_stats['dream_pair']} dream_pair, "
                f"{bridge_stats['triplet']} triplet, "
                f"{bridge_stats['exploration']} exploration) "
                f"largest_component={largest_component_size}/{total_non_dormant} "
                f"largest_component_ratio={largest_component_ratio:.3f} "
                f"edges_at_cap={edges_at_cap} cap_domain_pairs={cap_domain_pairs}"
            )

            # Frontier category bridge rates
            if hasattr(self, '_pair_frontier_category') and hasattr(self, '_novel_edge_sources'):
                cat_total = {"frontier": 0, "mixed": 0, "hub": 0}
                cat_bridge = {"frontier": 0, "mixed": 0, "hub": 0}
                for pair_key, fcat in self._pair_frontier_category.items():
                    cat_total[fcat] = cat_total.get(fcat, 0) + 1
                    # Check if this pair produced a bridge
                    if pair_key in self._novel_edge_sources:
                        a_id, b_id = tuple(pair_key)
                        a_comp = component_of.get(a_id)
                        b_comp = component_of.get(b_id)
                        if a_comp != b_comp or a_comp is None or b_comp is None:
                            cat_bridge[fcat] = cat_bridge.get(fcat, 0) + 1
                bridge_rates = {}
                for cat in ("frontier", "mixed", "hub"):
                    if cat_total[cat] > 0:
                        bridge_rates[cat] = f"{cat_bridge[cat]}/{cat_total[cat]}"
                    else:
                        bridge_rates[cat] = "0/0"
                logger.info(
                    f"Frontier bridge rates: frontier={bridge_rates['frontier']} "
                    f"mixed={bridge_rates['mixed']} hub={bridge_rates['hub']}"
                )

            # Degree distribution for this consolidation
            if belief_degrees:
                deg_dist = {0: 0, 1: 0, 2: 0, 3: 0, "4+": 0}
                for deg in belief_degrees.values():
                    if deg >= 4:
                        deg_dist["4+"] += 1
                    else:
                        deg_dist[deg] += 1
                logger.info(
                    f"Degree distribution: deg0={deg_dist[0]} deg1={deg_dist[1]} "
                    f"deg2={deg_dist[2]} deg3={deg_dist[3]} deg4+={deg_dist['4+']}"
                )

            # ── Per-consolidation telemetry block (soak 25 instrumentation) ──
            try:
                import math
                _db = self.semantic.db_conn

                # 1. Domain diversity + topic entropy (Shannon) over active beliefs
                domain_rows = _db.execute(
                    "SELECT topics FROM beliefs WHERE COALESCE(is_dormant, 0) = 0"
                ).fetchall()
                domain_counts: dict[str, int] = {}
                for dr in domain_rows:
                    try:
                        tags = json.loads(dr["topics"] or "[]")
                    except (json.JSONDecodeError, TypeError):
                        tags = []
                    d = tags[0] if tags else "other"
                    domain_counts[d] = domain_counts.get(d, 0) + 1
                total_beliefs = sum(domain_counts.values())
                topic_entropy = 0.0
                if total_beliefs > 0:
                    for cnt in domain_counts.values():
                        p = cnt / total_beliefs
                        if p > 0:
                            topic_entropy -= p * math.log2(p)
                domain_diversity = len(domain_counts)

                # 2. Depth distribution over active beliefs
                depth_rows = _db.execute(
                    "SELECT COALESCE(abstraction_depth, 0) as depth FROM beliefs "
                    "WHERE COALESCE(is_dormant, 0) = 0"
                ).fetchall()
                depth_hist: dict[str, int] = {}
                for dr in depth_rows:
                    key = str(dr["depth"]) if dr["depth"] < 5 else "5+"
                    depth_hist[key] = depth_hist.get(key, 0) + 1

                # 3. Global LCR — full graph, not sample-scoped
                all_active_ids = {r["id"] for r in _db.execute(
                    "SELECT id FROM beliefs WHERE COALESCE(is_dormant, 0) = 0"
                ).fetchall()}
                all_edges = _db.execute(
                    "SELECT belief_a, belief_b FROM belief_links "
                    "WHERE COALESCE(active, 1) = 1"
                ).fetchall()
                g_adj: dict[str, set[str]] = {nid: set() for nid in all_active_ids}
                for e in all_edges:
                    if e["belief_a"] in g_adj and e["belief_b"] in g_adj:
                        g_adj[e["belief_a"]].add(e["belief_b"])
                        g_adj[e["belief_b"]].add(e["belief_a"])
                g_visited: set[str] = set()
                g_largest = 0
                for start in g_adj:
                    if start in g_visited:
                        continue
                    queue = [start]
                    comp_size = 0
                    while queue:
                        node = queue.pop()
                        if node in g_visited:
                            continue
                        g_visited.add(node)
                        comp_size += 1
                        queue.extend(g_adj[node] - g_visited)
                    if comp_size > g_largest:
                        g_largest = comp_size
                global_lcr = g_largest / len(all_active_ids) if all_active_ids else 0.0

                depth_str = " ".join(f"d{k}={v}" for k, v in sorted(depth_hist.items()))
                domain_str = " ".join(f"{k}={v}" for k, v in sorted(domain_counts.items(), key=lambda x: -x[1]))
                logger.info(
                    f"Consolidation telemetry: "
                    f"topic_entropy={topic_entropy:.3f} domain_diversity={domain_diversity} "
                    f"global_lcr={global_lcr:.3f} ({g_largest}/{len(all_active_ids)}) "
                    f"depth=[{depth_str}] domains=[{domain_str}]"
                )
            except Exception as tel_e:
                logger.warning(f"Consolidation telemetry failed: {tel_e}")

        except Exception as e:
            logger.warning(f"Dreams failed: {e}")

    def _run_triplet_dreams(self, sampled, embeddings, queue_mode,
                            _pre_edge_pairs=None, _edge_stats=None,
                            belief_degrees=None):
        """Triplet synthesis — activated when a cluster qualifies.

        Requires a cluster with >= min_cluster_for_triplets beliefs and
        >= min_reinforced_for_triplets reinforced internal edges.
        """
        if _pre_edge_pairs is None:
            _pre_edge_pairs = set()
        if _edge_stats is None:
            _edge_stats = {
                "dream_pair_novel": 0, "dream_pair_reinforcing": 0,
                "triplet_novel": 0, "triplet_reinforcing": 0,
                "exploration_novel": 0, "exploration_reinforcing": 0,
            }
        if belief_degrees is None:
            belief_degrees = {}

        # Build clusters from sampled beliefs via BFS on active edges
        id_to_idx = {b["id"]: i for i, b in enumerate(sampled)}
        adj: dict[str, set[str]] = {b["id"]: set() for b in sampled}

        for i, b in enumerate(sampled):
            links = self.semantic.get_belief_links(b["id"])
            for link in links:
                other = link["belief_b"] if link["belief_a"] == b["id"] else link["belief_a"]
                if other in adj:
                    adj[b["id"]].add(other)
                    adj[other].add(b["id"])

        # BFS to find components
        visited: set[str] = set()
        clusters: list[list[str]] = []
        for node in adj:
            if node in visited:
                continue
            queue = [node]
            component = []
            visited.add(node)
            while queue:
                current = queue.pop(0)
                component.append(current)
                for nb in adj[current]:
                    if nb not in visited:
                        visited.add(nb)
                        queue.append(nb)
            if len(component) >= self.min_cluster_for_triplets:
                clusters.append(component)

        if not clusters:
            return

        # Check each qualifying cluster for reinforced edge count
        qualifying = []
        for cluster in clusters:
            cset = set(cluster)
            reinforced_count = 0
            for a, b in combinations(cluster, 2):
                if b in adj.get(a, set()):
                    rc = self.semantic.get_reinforced_count(a, b)
                    if rc > 0:
                        reinforced_count += 1
            if reinforced_count >= self.min_reinforced_for_triplets:
                qualifying.append(cluster)

        if not qualifying:
            return

        logger.info(
            f"Triplet dreams: {len(qualifying)} qualifying cluster(s) "
            f"(sizes: {[len(c) for c in qualifying]})"
        )

        # Triplet cluster cooldown — filter out clusters used in recent consolidations
        cluster_keys = [frozenset(c) for c in qualifying]
        eligible = []
        cooled_off = []
        for cluster, ckey in zip(qualifying, cluster_keys):
            if ckey in self._triplet_cluster_history:
                cooled_off.append(cluster)
            else:
                eligible.append(cluster)

        if eligible:
            if cooled_off:
                logger.info(
                    f"Triplet cluster cooldown: {len(cooled_off)} cluster(s) skipped "
                    f"(sizes: {[len(c) for c in cooled_off]}), "
                    f"{len(eligible)} eligible"
                )
            active_qualifying = eligible
        else:
            # All clusters in cooldown — override, use best available
            logger.info(
                f"Triplet cluster cooldown: all {len(qualifying)} cluster(s) in cooldown, "
                f"overriding — using best available"
            )
            active_qualifying = qualifying

        # Generate triplet candidates from qualifying clusters
        # Degree penalty: prevents high-degree hub nodes from dominating triplet closure.
        # Without this, hubs have more near-triangles → more closures → higher degree → more
        # triangles. This is the structural root cause of gravity wells (Ca(OH)₂ soak 19,
        # gut microbiome Research day 20).
        import math
        triplet_degree_lambda = 0.3  # degree penalty weight
        triplet_candidates = []
        for cluster in active_qualifying:
            cluster_key = frozenset(cluster)
            for a_id, b_id, c_id in combinations(cluster, 3):
                # All three must be in sampled (have embeddings)
                if a_id not in id_to_idx or b_id not in id_to_idx or c_id not in id_to_idx:
                    continue
                ia, ib, ic = id_to_idx[a_id], id_to_idx[b_id], id_to_idx[c_id]
                sim_ab = self._cosine_similarity(embeddings[ia], embeddings[ib])
                sim_bc = self._cosine_similarity(embeddings[ib], embeddings[ic])
                sim_ac = self._cosine_similarity(embeddings[ia], embeddings[ic])
                avg_sim = (sim_ab + sim_bc + sim_ac) / 3

                # All pairs must be in the similarity window
                if not all(self.dream_similarity_min <= s <= self.dream_similarity_max
                           for s in (sim_ab, sim_bc, sim_ac)):
                    continue

                conf_a = sampled[ia].get("confidence", 0.0) or 0.0
                conf_b = sampled[ib].get("confidence", 0.0) or 0.0
                conf_c = sampled[ic].get("confidence", 0.0) or 0.0
                avg_conf = (conf_a + conf_b + conf_c) / 3

                # Degree penalty — penalize hub nodes in triplet selection
                deg_a = belief_degrees.get(a_id, 0)
                deg_b = belief_degrees.get(b_id, 0)
                deg_c = belief_degrees.get(c_id, 0)
                max_deg = max(deg_a, deg_b, deg_c)
                degree_penalty = triplet_degree_lambda * math.log(max_deg + 1)

                score = avg_sim * 0.6 + avg_conf * 0.4 - degree_penalty
                triplet_candidates.append((ia, ib, ic, avg_sim, score, cluster_key))

        if not triplet_candidates:
            logger.info("Triplet dreams: no qualifying triplets found.")
            return

        # Filter by score, take top 1 (triplets are expensive)
        triplet_candidates = [t for t in triplet_candidates if t[4] >= self.min_dream_score]
        if not triplet_candidates:
            logger.info("Triplet dreams: no triplets cleared score threshold.")
            return

        # Per-cycle participation cap: no node in more than 2 triplet closures per consolidation.
        # Prevents gravity well growth regardless of topology.
        triplet_participation_cap = 2
        if not hasattr(self, '_triplet_participation_count'):
            self._triplet_participation_count = {}

        # Filter out candidates where any node has hit the cap
        capped_candidates = []
        for t in triplet_candidates:
            ia, ib, ic = t[0], t[1], t[2]
            ids = [sampled[ia]["id"], sampled[ib]["id"], sampled[ic]["id"]]
            if any(self._triplet_participation_count.get(nid, 0) >= triplet_participation_cap for nid in ids):
                continue
            capped_candidates.append(t)

        if not capped_candidates:
            logger.info("Triplet dreams: all candidates hit participation cap.")
            return

        capped_candidates.sort(key=lambda x: x[4], reverse=True)
        best = capped_candidates[0]
        ia, ib, ic, avg_sim, _score, used_cluster_key = best
        ba, bb, bc = sampled[ia], sampled[ib], sampled[ic]

        # Track participation
        for nid in [ba["id"], bb["id"], bc["id"]]:
            self._triplet_participation_count[nid] = self._triplet_participation_count.get(nid, 0) + 1

        # Record cluster used and trim history to last N consolidations
        self._triplet_cluster_history.append(used_cluster_key)
        if len(self._triplet_cluster_history) > self._triplet_cluster_cooldown_n:
            self._triplet_cluster_history = self._triplet_cluster_history[-self._triplet_cluster_cooldown_n:]

        logger.info(
            f"Triplet dream: '{ba['statement'][:30]}' + "
            f"'{bb['statement'][:30]}' + '{bc['statement'][:30]}'"
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a creative thinker synthesizing patterns across "
                    "multiple ideas. Be specific and insightful."
                ),
            },
            {
                "role": "user",
                "content": self._no_think(TRIPLET_DREAMS_PROMPT.format(
                    belief_a=ba["statement"],
                    belief_b=bb["statement"],
                    belief_c=bc["statement"],
                )),
            },
        ]

        response = self.inference.generate_with_messages(
            messages, max_tokens=self.analysis_max_tokens, temperature=0.1, task="triplet",
        )
        if not response:
            logger.warning("Triplet synthesis: no response from model")
            return []

        parsed = self._parse_dream_response(response)
        if not parsed:
            logger.info("Triplet dream: model returned NONE.")
            return

        triplet_data = {
            "belief_a_id": ba["id"],
            "belief_a_statement": ba["statement"],
            "belief_b_id": bb["id"],
            "belief_b_statement": bb["statement"],
            "belief_c_id": bc["id"],
            "belief_c_statement": bc["statement"],
            "connection": parsed["connection"],
            "inference": parsed["inference"],
            "avg_similarity": avg_sim,
        }

        if queue_mode:
            decision, reason = self._auto_triage("triplet_dream", triplet_data)
            self._log_dream_triage(triplet_data, decision, reason)
            if decision == "accept":
                from reflection.dream_transaction import DreamTransaction
                txn = DreamTransaction(self.semantic, self.embeddings)
                # Triplet: 1 belief + 3 child links + up to 3 parent pair links
                txn.expect(belief_created=True, link_count=None)
                # Defer curiosity gap to post-commit
                _ba_ref, _bb_ref, _bc_ref = ba, bb, bc
                _td_ref = triplet_data
                domains = {ba.get("_domain"), bb.get("_domain"), bc.get("_domain")}
                domains.discard(None)
                if len(domains) >= 2 and self.curiosity:
                    dom_list = sorted(domains)
                    stmt_snippets = [s["statement"][:40] for s in (ba, bb, bc)]
                    gap_question = (
                        f"What connects '{stmt_snippets[0]}' ({dom_list[0]}) "
                        f"and '{stmt_snippets[1]}' ({dom_list[1]})?"
                    )
                    _gap_q = gap_question
                    _gap_ctx = (
                        f"Dream synthesis bridged: "
                        f"'{triplet_data['belief_a_statement'][:40]}' + "
                        f"'{triplet_data['belief_b_statement'][:40]}' + "
                        f"'{triplet_data['belief_c_statement'][:40]}'"
                    )
                    _gap_tags = list(domains)
                    _gap_dom_list = dom_list
                    txn.defer_callback(
                        lambda: self._triplet_curiosity_gap(
                            _gap_q, _gap_ctx, _gap_tags, _gap_dom_list
                        )
                    )
                self._process_approved_triplet(triplet_data, semantic_override=txn)
                if txn.commit():
                    # Track triplet novel/reinforcing edges (parent pairs + parent-to-synthesis)
                    _triplet_parent_ids = [triplet_data["belief_a_id"],
                                           triplet_data["belief_b_id"],
                                           triplet_data["belief_c_id"]]
                    from itertools import combinations as _combs
                    for _ta, _tb in _combs(_triplet_parent_ids, 2):
                        _tkey = frozenset({_ta, _tb})
                        if _tkey not in _pre_edge_pairs:
                            _edge_stats["triplet_novel"] += 1
                            self._novel_edge_sources[_tkey] = "triplet"
                        else:
                            _edge_stats["triplet_reinforcing"] += 1
                    logger.info(f"Triplet dream auto-accepted: {reason}")
                else:
                    logger.error(
                        f"Triplet transaction failed, rolled back: "
                        f"{triplet_data.get('inference', '')[:60]}"
                    )
            elif decision == "reject":
                self._record_rejection("triplet_dream", triplet_data)
                if reason in self._SHADOW_LOG_REASONS:
                    self._log_rejected_belief("triplet_dream", triplet_data, reason)
                logger.info(f"Triplet dream auto-rejected: {reason}")
            else:
                label = (
                    f"Triplet: \"{ba['statement'][:25]}\" + "
                    f"\"{bb['statement'][:25]}\" + "
                    f"\"{bc['statement'][:25]}\" = {parsed['inference'][:40]}"
                )
                triplet_rec = avg_sim >= 0.50
                self.queue_item("triplet_dream", triplet_data, triplet_rec, label)
                logger.info("Triplet dream queued for approval.")
        else:
            print(f"\n  === Triplet Dream ===")
            print(f"  A: \"{ba['statement'][:60]}\"")
            print(f"  B: \"{bb['statement'][:60]}\"")
            print(f"  C: \"{bc['statement'][:60]}\"")
            print(f"  = {parsed['inference']}")
            try:
                ans = input("  Accept? [y/n] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                ans = "n"
            if ans in ("y", "yes"):
                self._process_approved_triplet(triplet_data)
                print("  Accepted.\n")
            else:
                print("  Skipped.\n")

    def _parse_dream_response(self, response):
        """Parse CONNECTION/INFERENCE lines from dream model output.

        Returns dict with 'connection' and 'inference' keys, or None.
        """
        if not response or response.strip().upper() == "NONE":
            return None

        connection = None
        inference = None

        for line in response.strip().split("\n"):
            line = line.strip()
            upper = line.upper()

            if upper.startswith("CONNECTION:"):
                connection = line.split(":", 1)[1].strip()
            elif upper.startswith("INFERENCE:"):
                inference = line.split(":", 1)[1].strip()

        if connection and inference:
            return {"connection": connection, "inference": inference}
        return None

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _get_episode_insights(self, episode_id):
        """Get existing key_insights for an episode."""
        row = self.episodic.db_conn.execute(
            "SELECT key_insights FROM episodes WHERE id = ?",
            (episode_id,),
        ).fetchone()
        if row and row["key_insights"]:
            return json.loads(row["key_insights"])
        return []

    def _get_consolidated_count(self):
        """Count of consolidated episodes."""
        row = self.episodic.db_conn.execute(
            "SELECT COUNT(*) FROM episodes WHERE summarized = 1"
        ).fetchone()
        return row[0]
