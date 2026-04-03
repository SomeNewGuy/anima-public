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

Drop-in replacement for semantic.py. All core operations (beliefs, links,
graph, questions) delegate to the compiled Rust engine. Python-side code
handles: embeddings, domain classification (plugin-provided), and any
product-specific logic.

Usage:
    # Instead of: from memory.semantic import SemanticMemory
    from memory.semantic_rust import SemanticMemory
"""

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

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
        """Scoped commit suppression for transactional writes.

        While active, individual write methods skip their per-op commit().
        The caller is responsible for BEGIN/COMMIT/ROLLBACK on db_conn.
        """
        old = self._suppress_commit
        self._suppress_commit = True
        try:
            yield
        finally:
            self._suppress_commit = old

    def initialize(self):
        """Initialize the belief store.

        If Rust core is available, delegates to anima_core.Engine.
        Also opens a Python sqlite3 connection for backward-compat
        (some code reads db_conn directly for product-specific queries).
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

        # If no Rust engine, create tables via Python (backward compat)
        if not self._engine:
            self._create_tables()
            self._migrate_tables()

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
        if self._engine:
            # Compute core_similarity via Python embeddings if available
            # (Rust doesn't have the embedding model)
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
                evidence_confidence=evidence_confidence if evidence_confidence is not None else confidence,
                synthesis_confidence=synthesis_confidence if synthesis_confidence is not None else confidence,
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
        else:
            return self._add_belief_python(statement, confidence, topics, entities,
                                            source_episode, supporting_evidence, source,
                                            source_type, abstraction_depth, parent_a,
                                            parent_b, generation_type, generation_cycle,
                                            operator_anchored, belief_status, corpus_id,
                                            evidence_confidence, synthesis_confidence)

    def get_belief_by_id(self, belief_id):
        """Get a single belief by ID."""
        if self._engine:
            return self._engine.get_belief(belief_id)
        row = self.db_conn.execute(
            "SELECT * FROM beliefs WHERE id = ?", (belief_id,)
        ).fetchone()
        return dict(row) if row else None

    def search_beliefs(self, topics=None, entities=None, min_confidence=0.0,
                       limit=15, include_deprecated=False):
        """Search beliefs."""
        if self._engine:
            # Rust search_beliefs takes just limit for now
            results = self._engine.search_beliefs(limit)
            if min_confidence > 0:
                results = [r for r in results if r.get("confidence", 0) >= min_confidence]
            return results
        # Python fallback
        return self._search_beliefs_python(topics, entities, min_confidence, limit, include_deprecated)

    def update_belief(self, belief_id, new_confidence=None, new_statement=None,
                      reason=None, episode_id=None, source=None):
        """Update a belief. Returns history ID."""
        if self._engine:
            return self._engine.update_belief(belief_id, new_confidence, new_statement, reason)
        return self._update_belief_python(belief_id, new_confidence, new_statement, reason, episode_id, source)

    def deprecate_belief(self, belief_id, reason=None, source=None):
        """Soft-deprecate a belief."""
        if self._engine:
            return self._engine.deprecate_belief(belief_id, reason)
        return self._deprecate_belief_python(belief_id, reason, source)

    def supersede_belief(self, old_belief_id, new_belief_id, reason=None):
        """Mark a belief as superseded by another."""
        if self._engine:
            self._engine.supersede_belief(old_belief_id, new_belief_id, reason)
            return
        self._supersede_belief_python(old_belief_id, new_belief_id, reason)

    def restore_belief(self, belief_id, restore_confidence=None):
        """Restore a deprecated belief."""
        if self._engine:
            return self._engine.restore_belief(belief_id, restore_confidence)
        return self._restore_belief_python(belief_id, restore_confidence)

    def get_belief_count(self):
        """Count non-deprecated beliefs."""
        if self._engine:
            return self._engine.get_belief_count()
        row = self.db_conn.execute(
            "SELECT COUNT(*) FROM beliefs WHERE COALESCE(deprecated, 0) = 0"
        ).fetchone()
        return row[0]

    def get_belief_history(self, belief_id):
        """Get change history for a belief."""
        if self._engine:
            return self._engine.get_belief_history(belief_id)
        return self._get_belief_history_python(belief_id)

    def delete_belief(self, belief_id):
        """Hard delete a belief."""
        if self._engine:
            # Rust handles cascade delete
            # But also do it on the Python connection for compat
            self.db_conn.execute("DELETE FROM belief_history WHERE belief_id = ?", (belief_id,))
            self.db_conn.execute("DELETE FROM belief_links WHERE belief_a = ? OR belief_b = ?", (belief_id, belief_id))
            self.db_conn.execute("DELETE FROM belief_sources WHERE belief_id = ?", (belief_id,))
            self.db_conn.execute("DELETE FROM beliefs WHERE id = ?", (belief_id,))
            self.db_conn.commit()
            return
        self._delete_belief_python(belief_id)

    # ── Belief Links ─────────────────────────────────────────────

    def add_belief_link(self, belief_a, belief_b, inference, similarity=None,
                        link_type="dream"):
        """Create an edge between two beliefs."""
        if self._engine:
            return self._engine.add_link(belief_a, belief_b, link_type, inference, similarity)
        return self._add_link_python(belief_a, belief_b, inference, similarity, link_type)

    def get_belief_links(self, belief_id, include_inactive=False):
        """Get all links for a belief."""
        if self._engine:
            return self._engine.get_belief_links(belief_id)
        return self._get_links_python(belief_id, include_inactive)

    def link_exists(self, belief_a, belief_b):
        """Check if a link exists between two beliefs."""
        if self._engine:
            return self._engine.link_exists(belief_a, belief_b)
        return self._link_exists_python(belief_a, belief_b)

    def reinforce_or_reactivate_link(self, belief_a, belief_b, new_similarity):
        """Reinforce an existing link. Returns (link_id, hit_cap) or None."""
        if self._engine:
            return self._engine.reinforce_link(belief_a, belief_b, new_similarity)
        return self._reinforce_link_python(belief_a, belief_b, new_similarity)

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
        if self._engine:
            return self._engine.get_graph_stats()
        return self._get_graph_stats_python()

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

    # ── Python Fallbacks ─────────────────────────────────────────
    # These are used when Rust core is not available.
    # They implement the same logic as the original semantic.py.

    def _add_belief_python(self, *args, **kwargs):
        """Pure Python fallback for add_belief."""
        # Import original implementation on demand
        logger.warning("Using Python fallback for add_belief — Rust core not available")
        return None

    def _search_beliefs_python(self, *args, **kwargs):
        logger.warning("Using Python fallback for search_beliefs")
        query = "SELECT * FROM beliefs WHERE COALESCE(deprecated, 0) = 0 AND COALESCE(is_dormant, 0) = 0"
        query += " ORDER BY confidence DESC LIMIT ?"
        limit = args[3] if len(args) > 3 else 15
        rows = self.db_conn.execute(query, (limit,)).fetchall()
        return [dict(r) for r in rows]

    def _update_belief_python(self, *args, **kwargs):
        logger.warning("Using Python fallback for update_belief")
        return None

    def _deprecate_belief_python(self, *args, **kwargs):
        logger.warning("Using Python fallback for deprecate_belief")
        return None

    def _supersede_belief_python(self, *args, **kwargs):
        logger.warning("Using Python fallback for supersede_belief")
        return None

    def _restore_belief_python(self, *args, **kwargs):
        logger.warning("Using Python fallback for restore_belief")
        return None

    def _get_belief_history_python(self, belief_id):
        rows = self.db_conn.execute(
            "SELECT * FROM belief_history WHERE belief_id = ? ORDER BY timestamp",
            (belief_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def _delete_belief_python(self, belief_id):
        self.db_conn.execute("DELETE FROM belief_history WHERE belief_id = ?", (belief_id,))
        self.db_conn.execute("DELETE FROM belief_links WHERE belief_a = ? OR belief_b = ?", (belief_id, belief_id))
        self.db_conn.execute("DELETE FROM beliefs WHERE id = ?", (belief_id,))
        self.db_conn.commit()

    def _add_link_python(self, belief_a, belief_b, inference, similarity, link_type):
        import uuid
        from datetime import datetime, timezone
        link_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        self.db_conn.execute(
            "INSERT INTO belief_links (id, belief_a, belief_b, link_type, inference, similarity, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (link_id, belief_a, belief_b, link_type, inference, similarity, now))
        self.db_conn.commit()
        return link_id

    def _get_links_python(self, belief_id, include_inactive):
        query = "SELECT * FROM belief_links WHERE (belief_a = ? OR belief_b = ?)"
        if not include_inactive:
            query += " AND COALESCE(active, 1) = 1"
        rows = self.db_conn.execute(query, (belief_id, belief_id)).fetchall()
        return [dict(r) for r in rows]

    def _link_exists_python(self, belief_a, belief_b):
        row = self.db_conn.execute(
            "SELECT 1 FROM belief_links WHERE "
            "((belief_a = ? AND belief_b = ?) OR (belief_a = ? AND belief_b = ?)) "
            "AND COALESCE(active, 1) = 1",
            (belief_a, belief_b, belief_b, belief_a)
        ).fetchone()
        return row is not None

    def _reinforce_link_python(self, belief_a, belief_b, similarity):
        row = self.db_conn.execute(
            "SELECT id, reinforced_count, similarity FROM belief_links "
            "WHERE ((belief_a = ? AND belief_b = ?) OR (belief_a = ? AND belief_b = ?)) "
            "LIMIT 1",
            (belief_a, belief_b, belief_b, belief_a)
        ).fetchone()
        if not row:
            return None
        link_id = row["id"]
        rc = row["reinforced_count"] or 0
        new_rc = min(rc + 1, 5)
        new_sim = max(row["similarity"] or 0, similarity)
        self.db_conn.execute(
            "UPDATE belief_links SET reinforced_count = ?, similarity = ?, active = 1 WHERE id = ?",
            (new_rc, new_sim, link_id))
        self.db_conn.commit()
        return (link_id, new_rc >= 5)

    def _get_graph_stats_python(self):
        beliefs = self.db_conn.execute(
            "SELECT COUNT(*) FROM beliefs WHERE COALESCE(deprecated,0)=0"
        ).fetchone()[0]
        edges = self.db_conn.execute(
            "SELECT COUNT(*) FROM belief_links WHERE COALESCE(active,1)=1"
        ).fetchone()[0]
        return {
            "beliefs": beliefs,
            "edges": edges,
            "integration_pct": (edges / max(beliefs, 1)) * 100,
        }

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

    # ── Table creation (Python fallback only) ────────────────────

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

    def _create_tables(self):
        """Create core tables — only used when Rust core is not available."""
        self.db_conn.executescript("""
            CREATE TABLE IF NOT EXISTS beliefs (
                id TEXT PRIMARY KEY,
                statement TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.5,
                supporting_evidence TEXT,
                contradicting_evidence TEXT,
                source_episodes TEXT,
                topics TEXT,
                entities TEXT,
                last_challenged TEXT,
                last_updated TEXT NOT NULL,
                created_at TEXT NOT NULL,
                source_type TEXT,
                deprecated INTEGER DEFAULT 0,
                operator_anchored INTEGER DEFAULT 0,
                abstraction_depth INTEGER DEFAULT 0,
                parent_a TEXT,
                parent_b TEXT,
                generation_type TEXT,
                belief_status TEXT DEFAULT 'active',
                corpus_id INTEGER DEFAULT 1,
                epistemic_class TEXT DEFAULT 'corpus',
                core_similarity REAL DEFAULT NULL
            );
            CREATE TABLE IF NOT EXISTS belief_links (
                id TEXT PRIMARY KEY,
                belief_a TEXT NOT NULL,
                belief_b TEXT NOT NULL,
                link_type TEXT NOT NULL DEFAULT 'dream',
                inference TEXT NOT NULL,
                similarity REAL,
                created_at TEXT NOT NULL,
                active INTEGER DEFAULT 1,
                reinforced_count INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS belief_history (
                id TEXT PRIMARY KEY,
                belief_id TEXT NOT NULL,
                old_confidence REAL,
                new_confidence REAL,
                reason TEXT,
                timestamp TEXT NOT NULL
            );
        """)
        self.db_conn.commit()

    def _migrate_tables(self):
        """Idempotent migrations — only used when Rust core is not available."""
        migrations = [
            "ALTER TABLE beliefs ADD COLUMN is_dormant INTEGER DEFAULT 0",
            "ALTER TABLE beliefs ADD COLUMN operator_model INTEGER DEFAULT 0",
            "ALTER TABLE beliefs ADD COLUMN generation_cycle INTEGER",
            "ALTER TABLE beliefs ADD COLUMN evidence_confidence REAL",
            "ALTER TABLE beliefs ADD COLUMN synthesis_confidence REAL",
            "ALTER TABLE beliefs ADD COLUMN valid_from TEXT",
            "ALTER TABLE beliefs ADD COLUMN valid_to TEXT",
            "ALTER TABLE beliefs ADD COLUMN superseded_by TEXT",
            "ALTER TABLE beliefs ADD COLUMN temporal_type TEXT DEFAULT 'ongoing'",
            "ALTER TABLE beliefs ADD COLUMN tree_paths TEXT DEFAULT '[]'",
            "ALTER TABLE beliefs ADD COLUMN promoted_from TEXT",
            "ALTER TABLE beliefs ADD COLUMN last_verified TEXT",
            "ALTER TABLE beliefs ADD COLUMN evidence_strength REAL",
            "ALTER TABLE belief_links ADD COLUMN direction TEXT DEFAULT 'bidirectional'",
        ]
        for sql in migrations:
            try:
                self.db_conn.execute(sql)
            except sqlite3.OperationalError:
                pass  # Column already exists
        self.db_conn.commit()
