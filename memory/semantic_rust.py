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

"""Rust-backed semantic memory — thin wrapper around anima_core.Engine.

All core operations (beliefs, links, graph, questions) delegate to the
compiled Rust engine. Python-side code handles: embeddings, domain
classification (plugin-provided), and any product-specific logic.

Usage:
    from memory.semantic_rust import SemanticMemory
"""

import json
import logging
import os
import sqlite3
from contextlib import contextmanager

logger = logging.getLogger("semantic")

import anima_core
logger.info("Rust core (anima_core) loaded — using compiled engine")


class SemanticMemory:
    """Belief store backed by Rust-compiled anima_core.

    API-compatible with the pure Python SemanticMemory. All public methods
    have identical signatures and return types.
    """

    def __init__(self, config):
        self.config = config
        self.db_conn = None
        self._engine = None
        self._suppress_commit = False

        # These are set externally by the CLI/engine after init
        self._embeddings = None
        self._core_embedding = None
        self._research_domain_classifier = None

    @contextmanager
    def transaction(self):
        """Context manager for atomic dream buffer writes.

        Suppresses per-belief commits inside the block, then commits once
        at the end.  Used by the dream engine so that a failed dream
        doesn't leave partial state in the DB.
        """
        self._suppress_commit = False  # reset in case of prior failure
        self._suppress_commit = True
        try:
            yield
        except Exception:
            if self.db_conn:
                self.db_conn.rollback()
            raise
        else:
            if self.db_conn:
                self.db_conn.commit()
        finally:
            self._suppress_commit = False

    def initialize(self, product_mode=None, plugin_name=None,
                   data_dir_override=None, db_path_override=None):
        """Initialize the Rust engine and database connection.

        Must be called before any operations. Uses ANIMA_DATA_DIR env var
        or falls back to config-relative path for database location.
        """
        self._engine = anima_core.Engine(self.config)
        logger.info("Rust engine initialized")

        # Also maintain a Python sqlite3 connection for backward compat
        # (plugins, orchestrators, dashboards read db_conn directly)
        data_dir = os.environ.get("ANIMA_DATA_DIR")
        if data_dir:
            sqlite_path = os.path.normpath(
                os.path.join(data_dir, "sqlite", "persistence.db")
            )
        else:
            base_dir = os.path.join(os.path.dirname(__file__), "..")
            sqlite_path = os.path.normpath(
                os.path.join(base_dir, self.config["memory"]["sqlite_path"])
            )
        os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
        self.db_path = sqlite_path  # exposed for other components (evolution queue, etc.)
        self.db_conn = sqlite3.connect(sqlite_path, check_same_thread=False)
        self.db_conn.row_factory = sqlite3.Row
        self.db_conn.execute("PRAGMA journal_mode=WAL")

        return self

    # ── Belief CRUD ──────────────────────────────────────────────

    def add_belief(self, statement, confidence=0.5, topics=None, entities=None,
                   source_episode=None, supporting_evidence=None, source=None,
                   source_type=None, abstraction_depth=0, parent_a=None,
                   parent_b=None, generation_type=None, generation_cycle=None,
                   operator_anchored=0, belief_status="active", corpus_id=1,
                   evidence_confidence=None, synthesis_confidence=None,
                   source_mode=None, belief_type=None):
        """Add a belief. Returns belief ID or None."""
        topics_list = topics if isinstance(topics, list) else (
            json.loads(topics) if isinstance(topics, str) and topics else [])
        entities_list = entities if isinstance(entities, list) else (
            json.loads(entities) if isinstance(entities, str) and entities else [])

        bid = self._engine.add_belief(
            statement=statement,
            confidence=confidence,
            source_type=source_type or source or "corpus",
            topics=json.dumps(topics_list) if topics_list else None,
            entities=json.dumps(entities_list) if entities_list else None,
            parent_a=parent_a,
            parent_b=parent_b,
            generation_type=generation_type,
            generation_cycle=generation_cycle,
            operator_anchored=bool(operator_anchored),
            corpus_id=corpus_id,
            abstraction_depth=abstraction_depth,
            evidence_confidence=evidence_confidence,
            synthesis_confidence=synthesis_confidence,
            source_mode=source_mode,
            belief_type=belief_type,
        )

        # Compute core_similarity via Python embeddings
        if bid and self._embeddings and self._core_embedding is not None:
            try:
                emb = self._embeddings.embed(statement)
                import numpy as np
                norm_a = np.linalg.norm(emb)
                norm_b = np.linalg.norm(self._core_embedding)
                if norm_a > 0 and norm_b > 0:
                    core_sim = float(np.dot(emb, self._core_embedding) / (norm_a * norm_b))
                    self.db_conn.execute(
                        "UPDATE beliefs SET core_similarity = ? WHERE id = ?",
                        (core_sim, bid))
                    self.db_conn.commit()
            except Exception:
                pass

        return bid

    def get_belief_by_id(self, belief_id):
        """Get a single belief by ID."""
        return self._engine.get_belief(belief_id)

    def search_beliefs(self, topics=None, entities=None, min_confidence=0.0,
                       limit=15, include_deprecated=False):
        """Search beliefs."""
        results = self._engine.search_beliefs(limit)
        if min_confidence > 0:
            results = [r for r in results if r.get("confidence", 0) >= min_confidence]
        return results

    def update_belief(self, belief_id, new_confidence=None, new_statement=None,
                      reason=None, episode_id=None, source=None):
        """Update a belief. Returns history ID."""
        return self._engine.update_belief(belief_id, new_confidence, new_statement, reason)

    def deprecate_belief(self, belief_id, reason=None, source=None):
        """Soft-deprecate a belief."""
        return self._engine.deprecate_belief(belief_id, reason)

    def supersede_belief(self, old_belief_id, new_belief_id, reason=None):
        """Mark a belief as superseded by another."""
        self._engine.supersede_belief(old_belief_id, new_belief_id, reason)

    def restore_belief(self, belief_id, restore_confidence=None):
        """Restore a deprecated belief."""
        return self._engine.restore_belief(belief_id, restore_confidence)

    def get_belief_count(self):
        """Count non-deprecated beliefs."""
        return self._engine.get_belief_count()

    def get_belief_history(self, belief_id):
        """Get change history for a belief."""
        return self._engine.get_belief_history(belief_id)

    def delete_belief(self, belief_id):
        """Hard delete a belief."""
        # Rust handles cascade delete
        # But also do it on the Python connection for compat
        self.db_conn.execute("DELETE FROM belief_history WHERE belief_id = ?", (belief_id,))
        self.db_conn.execute("DELETE FROM belief_links WHERE belief_a = ? OR belief_b = ?", (belief_id, belief_id))
        self.db_conn.execute("DELETE FROM belief_sources WHERE belief_id = ?", (belief_id,))
        self.db_conn.execute("DELETE FROM beliefs WHERE id = ?", (belief_id,))
        self.db_conn.commit()

    # ── Belief Links ─────────────────────────────────────────────

    def add_belief_link(self, belief_a, belief_b, inference, similarity=None,
                        link_type="dream"):
        """Create an edge between two beliefs."""
        return self._engine.add_link(belief_a, belief_b, link_type, inference, similarity)

    def get_belief_links(self, belief_id, include_inactive=False):
        """Get all links for a belief."""
        return self._engine.get_belief_links(belief_id)

    def link_exists(self, belief_a, belief_b):
        """Check if a link exists between two beliefs."""
        return self._engine.link_exists(belief_a, belief_b)

    def reinforce_or_reactivate_link(self, belief_a, belief_b, new_similarity):
        """Reinforce an existing link. Returns (link_id, hit_cap) or None."""
        return self._engine.reinforce_link(belief_a, belief_b, new_similarity)

    def get_reinforced_count(self, belief_a, belief_b):
        """Get reinforcement count for a pair."""
        row = self.db_conn.execute(
            "SELECT reinforced_count FROM belief_links "
            "WHERE ((belief_a = ? AND belief_b = ?) OR (belief_a = ? AND belief_b = ?)) "
            "AND COALESCE(active, 1) = 1",
            (belief_a, belief_b, belief_b, belief_a)
        ).fetchone()
        return row[0] if row else 0

    def get_belief_depth(self, belief_id):
        """Get abstraction depth of a belief."""
        row = self.db_conn.execute(
            "SELECT abstraction_depth FROM beliefs WHERE id = ?", (belief_id,)
        ).fetchone()
        return row[0] if row else 0

    # ── Graph Operations ─────────────────────────────────────────

    def get_graph_stats(self):
        """Get graph statistics."""
        return self._engine.get_graph_stats()

    # ── Embeddings Bridge ────────────────────────────────────────
    # Embeddings stay in Python (sentence-transformers).
    # The Rust core doesn't load models — it receives vectors.

    @staticmethod
    def _cosine_similarity(a, b):
        """Cosine similarity between two vectors."""
        import numpy as np
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm < 1e-9:
            return 0.0
        return float(np.dot(a, b) / norm)

    # ── Dormant adjacency check ──────────────────────────────────

    DORMANT_REACTIVATION_THRESHOLD = 0.75

    def check_dormant_adjacency(self, belief_id, embedder=None,
                                trigger_type="new_belief",
                                threshold=None):
        """Check if a new/updated belief is adjacent to any dormant beliefs.

        If similarity >= threshold, flag the dormant belief for reactivation
        review. Does NOT auto-reactivate.
        """
        if embedder is None:
            return []

        threshold = threshold or self.DORMANT_REACTIVATION_THRESHOLD

        trigger = self.db_conn.execute(
            "SELECT statement FROM beliefs WHERE id = ?", (belief_id,)
        ).fetchone()
        if not trigger:
            return []

        dormant = self.db_conn.execute(
            "SELECT id, statement FROM beliefs WHERE COALESCE(is_dormant, 0) = 1"
        ).fetchall()
        if not dormant:
            return []

        try:
            import numpy as np
            trigger_emb = embedder.embed(trigger["statement"])
        except Exception:
            return []

        flagged = []
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()

        for row in dormant:
            existing = self.db_conn.execute(
                "SELECT 1 FROM dormant_review "
                "WHERE dormant_belief_id = ? AND trigger_belief_id = ? "
                "AND status = 'pending' LIMIT 1",
                (row["id"], belief_id),
            ).fetchone()
            if existing:
                continue

            try:
                dormant_emb = embedder.embed(row["statement"])
                norm = np.linalg.norm(trigger_emb) * np.linalg.norm(dormant_emb)
                sim = float(np.dot(trigger_emb, dormant_emb) / norm) if norm > 0 else 0.0
            except Exception:
                continue

            if sim >= threshold:
                import uuid as _uuid
                review_id = str(_uuid.uuid4())
                self.db_conn.execute(
                    "INSERT INTO dormant_review "
                    "(id, dormant_belief_id, trigger_belief_id, trigger_type, "
                    "similarity, created_at, status) "
                    "VALUES (?, ?, ?, ?, ?, ?, 'pending')",
                    (review_id, row["id"], belief_id, trigger_type, sim, now),
                )
                flagged.append(row["id"])

        if flagged and not self._suppress_commit:
            self.db_conn.commit()
        return flagged

    # ── Lifecycle ────────────────────────────────────────────────

    def close(self):
        """Close database connections."""
        if self._engine:
            self._engine.shutdown()
        if self.db_conn:
            self.db_conn.close()

    def delete_all_beliefs(self):
        """Delete all beliefs — used for clean slate."""
        self.db_conn.execute("DELETE FROM belief_history")
        self.db_conn.execute("DELETE FROM belief_links")
        self.db_conn.execute("DELETE FROM belief_sources")
        self.db_conn.execute("DELETE FROM beliefs")
        self.db_conn.commit()
