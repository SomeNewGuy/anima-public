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

"""Tag system for ANIMA knowledge organization.

Controlled vocabulary with supervised growth. Tags come from three sources:
- Tree nodes: structural tags derived from knowledge tree placement
- Document entities: concept tags propagated from NER extraction
- LLM: new tags created when 5+ beliefs share an unregistered entity

Tags are orthogonal to tree position — tree provides structural tags,
entities provide concept tags, evidence_type provides methodological tags.

Core engine feature — ships with every product.
"""

import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger("core.tags")

# Entity types that become tags (from document_entities)
TAGGABLE_ENTITY_TYPES = {"topic", "identifier", "code"}

# Max tags per belief — prevents explosion on synthesis inheritance
MAX_TAGS_PER_BELIEF = 8

# Threshold for promoting an entity to a registered tag
ENTITY_CLUSTER_THRESHOLD = 5

# Embedding similarity threshold for tag merging (synonyms)
TAG_MERGE_THRESHOLD = 0.85

# Cycles with 0 beliefs before pruning a tag
TAG_PRUNE_CYCLES = 20


class TagRegistry:
    """Manages tag lifecycle: creation, assignment, inheritance, merging."""

    def __init__(self, db, embedding_engine=None):
        self.db = db
        self.embeddings = embedding_engine

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def ensure_schema(self):
        """Create tag tables if not exists."""
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS tag_registry (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                category TEXT NOT NULL,
                source TEXT NOT NULL,
                belief_count INTEGER DEFAULT 0,
                created_at TEXT,
                last_used_at TEXT,
                inactive_cycles INTEGER DEFAULT 0
            )
        """)
        # category: structural / entity / evidence_type / custom
        # source: tree / entity / llm / operator

        self.db.execute("""
            CREATE TABLE IF NOT EXISTS belief_tags (
                belief_id TEXT NOT NULL,
                tag_id TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                source TEXT NOT NULL,
                created_at TEXT,
                PRIMARY KEY (belief_id, tag_id),
                FOREIGN KEY (belief_id) REFERENCES beliefs(id),
                FOREIGN KEY (tag_id) REFERENCES tag_registry(id)
            )
        """)

        # Index for tag-based retrieval
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_belief_tags_tag ON belief_tags(tag_id)"
        )
        self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_belief_tags_belief ON belief_tags(belief_id)"
        )

        self.db.commit()

    # ------------------------------------------------------------------
    # Tag registration
    # ------------------------------------------------------------------

    def _normalize_tag_id(self, name):
        """Convert a tag name to a snake_case ID."""
        import re
        # Lowercase, replace non-alphanumeric with underscore, collapse multiples
        tid = re.sub(r'[^a-z0-9]+', '_', name.lower().strip()).strip('_')
        # Cap length
        return tid[:64] if tid else "unknown"

    def register_tag(self, name, category, source):
        """Register a tag in the registry. Returns tag_id. Idempotent."""
        tag_id = self._normalize_tag_id(name)
        now = datetime.now(timezone.utc).isoformat()

        existing = self.db.execute(
            "SELECT id FROM tag_registry WHERE id = ?", (tag_id,)
        ).fetchone()

        if existing:
            return tag_id

        self.db.execute(
            "INSERT OR IGNORE INTO tag_registry "
            "(id, name, category, source, belief_count, created_at, last_used_at, inactive_cycles) "
            "VALUES (?, ?, ?, ?, 0, ?, ?, 0)",
            (tag_id, name, category, source, now, now),
        )
        self.db.commit()
        logger.debug(f"Registered tag: {tag_id} ({category}/{source})")
        return tag_id

    def _ensure_tree_tags_registered(self, tree_nodes):
        """Register all Layer 2+ tree nodes as structural tags.

        tree_nodes: list of dicts with id, name, layer from knowledge_tree.
        """
        count = 0
        for node in tree_nodes:
            if node["layer"] >= 2:  # Layer 2+ become tags
                self.register_tag(node["name"], "structural", "tree")
                count += 1
        if count:
            logger.info(f"Registered {count} structural tags from tree nodes")

    # ------------------------------------------------------------------
    # Tag assignment
    # ------------------------------------------------------------------

    def assign_tag(self, belief_id, tag_id, confidence=1.0, source="tree"):
        """Assign a tag to a belief. Idempotent. Respects MAX_TAGS_PER_BELIEF."""
        # Check current tag count
        current = self.db.execute(
            "SELECT COUNT(*) FROM belief_tags WHERE belief_id = ?", (belief_id,)
        ).fetchone()[0]

        if current >= MAX_TAGS_PER_BELIEF:
            return False

        now = datetime.now(timezone.utc).isoformat()
        self.db.execute(
            "INSERT OR IGNORE INTO belief_tags (belief_id, tag_id, confidence, source, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (belief_id, tag_id, confidence, source, now),
        )

        # Update belief_count on registry
        self.db.execute(
            "UPDATE tag_registry SET belief_count = belief_count + 1, last_used_at = ? "
            "WHERE id = ? AND NOT EXISTS "
            "(SELECT 1 FROM belief_tags WHERE belief_id = ? AND tag_id = ? AND created_at < ?)",
            (now, tag_id, belief_id, tag_id, now),
        )
        return True

    def _update_belief_counts(self):
        """Recompute belief_count for all tags from junction table."""
        self.db.execute("""
            UPDATE tag_registry SET belief_count = (
                SELECT COUNT(*) FROM belief_tags WHERE tag_id = tag_registry.id
            )
        """)
        self.db.commit()

    # ------------------------------------------------------------------
    # Derive tags from tree placement
    # ------------------------------------------------------------------

    def derive_from_tree(self, belief_id, tree_paths):
        """Derive structural tags from a belief's tree paths.

        tree_paths: list of lists, e.g. [["biological","immunology","checkpoint_response"]]
        Skips Layer 0 (root) and Layer 1 (too broad). Tags from Layer 2+ nodes.
        """
        if not tree_paths:
            return []

        assigned = []
        for path in tree_paths:
            for i, node_id in enumerate(path):
                # Skip root (layer 0) and Layer 1 — too broad for useful tags
                # Path index 0 is typically Layer 1, index 1+ is Layer 2+
                if i < 1:
                    continue
                tag_id = self._normalize_tag_id(node_id)
                # Register if not exists (tree node names are clean)
                self.register_tag(node_id.replace("_", " ").title(), "structural", "tree")
                if self.assign_tag(belief_id, tag_id, confidence=1.0, source="tree"):
                    assigned.append(tag_id)

        self.db.commit()
        return assigned

    # ------------------------------------------------------------------
    # Derive tags from document entities
    # ------------------------------------------------------------------

    def derive_from_entities(self, belief_id, document_sha=None):
        """Propagate document entities to belief tags.

        Joins belief → belief_sources_documents → document_entities.
        Only propagates taggable entity types (topic, identifier, code).
        """
        if document_sha:
            shas = [document_sha]
        else:
            # Look up source documents for this belief
            rows = self.db.execute(
                "SELECT document_sha FROM belief_sources_documents WHERE belief_id = ?",
                (belief_id,),
            ).fetchall()
            shas = [r["document_sha"] for r in rows]

        if not shas:
            return []

        assigned = []
        for sha in shas:
            entities = self.db.execute(
                "SELECT entity_type, entity_value FROM document_entities "
                "WHERE document_sha = ? AND entity_type IN (?, ?, ?)",
                (sha, *TAGGABLE_ENTITY_TYPES),
            ).fetchall()

            for ent in entities:
                name = ent["entity_value"].strip()
                if len(name) < 2 or len(name) > 80:
                    continue

                tag_id = self._normalize_tag_id(name)
                if not tag_id:
                    continue

                # Only register if it already exists OR meets cluster threshold
                existing = self.db.execute(
                    "SELECT id FROM tag_registry WHERE id = ?", (tag_id,)
                ).fetchone()

                if existing:
                    if self.assign_tag(belief_id, tag_id, confidence=0.8, source="entity"):
                        assigned.append(tag_id)
                # Not registered yet — count how many beliefs share this entity
                # via document_entities. If >= threshold, register it.
                else:
                    count = self.db.execute(
                        "SELECT COUNT(DISTINCT bsd.belief_id) FROM belief_sources_documents bsd "
                        "JOIN document_entities de ON de.document_sha = bsd.document_sha "
                        "WHERE de.entity_value = ?",
                        (ent["entity_value"],),
                    ).fetchone()[0]

                    if count >= ENTITY_CLUSTER_THRESHOLD:
                        self.register_tag(name, "entity", "entity")
                        if self.assign_tag(belief_id, tag_id, confidence=0.8, source="entity"):
                            assigned.append(tag_id)

        self.db.commit()
        return assigned

    # ------------------------------------------------------------------
    # Inheritance for synthesis beliefs
    # ------------------------------------------------------------------

    def inherit_for_synthesis(self, belief_id, parent_a_id, parent_b_id):
        """Inherit tags from parent beliefs for a synthesis (dream) belief.

        Strategy: shared tags first (conceptual overlap that triggered the dream),
        then fill remaining slots by belief_count descending (established tags win).
        Capped at MAX_TAGS_PER_BELIEF.
        """
        # Get parent tags
        tags_a = set()
        tags_b = set()

        for pid, tag_set in [(parent_a_id, tags_a), (parent_b_id, tags_b)]:
            if not pid:
                continue
            rows = self.db.execute(
                "SELECT tag_id, confidence FROM belief_tags WHERE belief_id = ?",
                (pid,),
            ).fetchall()
            for r in rows:
                tag_set.add(r["tag_id"])

        if not tags_a and not tags_b:
            return []

        # Shared tags first — this is why these beliefs were dreamed together
        shared = tags_a & tags_b
        only_a = tags_a - tags_b
        only_b = tags_b - tags_a

        # Rank non-shared by belief_count (established tags win)
        remaining = list(only_a | only_b)
        if remaining:
            counts = {}
            for tid in remaining:
                row = self.db.execute(
                    "SELECT belief_count FROM tag_registry WHERE id = ?", (tid,)
                ).fetchone()
                counts[tid] = row["belief_count"] if row else 0
            remaining.sort(key=lambda t: counts.get(t, 0), reverse=True)

        # Assemble: shared first, then ranked remaining, capped
        ordered = list(shared) + remaining
        assigned = []
        for tag_id in ordered[:MAX_TAGS_PER_BELIEF]:
            if self.assign_tag(belief_id, tag_id, confidence=0.9, source="inherited"):
                assigned.append(tag_id)

        self.db.commit()
        return assigned

    # ------------------------------------------------------------------
    # Tag queries
    # ------------------------------------------------------------------

    def get_belief_tags(self, belief_id):
        """Get all tags for a belief."""
        rows = self.db.execute(
            "SELECT bt.tag_id, bt.confidence, bt.source, tr.name, tr.category "
            "FROM belief_tags bt JOIN tag_registry tr ON bt.tag_id = tr.id "
            "WHERE bt.belief_id = ?",
            (belief_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_beliefs_by_tag(self, tag_id, limit=100):
        """Get all belief IDs with a given tag."""
        rows = self.db.execute(
            "SELECT belief_id, confidence FROM belief_tags WHERE tag_id = ? LIMIT ?",
            (tag_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_top_tags(self, limit=20):
        """Get top tags by belief count."""
        rows = self.db.execute(
            "SELECT id, name, category, source, belief_count "
            "FROM tag_registry WHERE belief_count > 0 "
            "ORDER BY belief_count DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_tag_overlap(self, query_tags, belief_id):
        """Count how many query-relevant tags a belief has. For retrieval boost."""
        if not query_tags:
            return 0
        placeholders = ",".join("?" * len(query_tags))
        row = self.db.execute(
            f"SELECT COUNT(*) FROM belief_tags "
            f"WHERE belief_id = ? AND tag_id IN ({placeholders})",
            (belief_id, *query_tags),
        ).fetchone()
        return row[0] if row else 0

    def find_tags_for_query(self, query_text, limit=10):
        """Find tags relevant to a query by embedding similarity.

        Returns list of (tag_id, similarity) for retrieval boosting.
        """
        if not self.embeddings:
            return []

        import numpy as np

        query_emb = self.embeddings.embed(query_text)
        tags = self.db.execute(
            "SELECT id, name FROM tag_registry WHERE belief_count > 0"
        ).fetchall()

        scored = []
        for tag in tags:
            tag_emb = self.embeddings.embed(tag["name"])
            norm = np.linalg.norm(query_emb) * np.linalg.norm(tag_emb)
            sim = float(np.dot(query_emb, tag_emb) / norm) if norm > 0 else 0.0
            if sim >= 0.40:
                scored.append((tag["id"], sim))

        scored.sort(key=lambda x: -x[1])
        return scored[:limit]

    # ------------------------------------------------------------------
    # Tag co-occurrence analysis (for curiosity gap generation)
    # ------------------------------------------------------------------

    def find_co_occurrences(self, min_shared=3, limit=20):
        """Find tag pairs that co-occur in beliefs but lack a direct linking belief.

        Returns pairs sorted by co-occurrence count descending.
        Used to generate curiosity gaps — "these concepts appear together
        but nothing directly connects them."
        """
        rows = self.db.execute("""
            SELECT bt1.tag_id AS tag_a, bt2.tag_id AS tag_b,
                   COUNT(DISTINCT bt1.belief_id) AS shared_count
            FROM belief_tags bt1
            JOIN belief_tags bt2 ON bt1.belief_id = bt2.belief_id
                AND bt1.tag_id < bt2.tag_id
            GROUP BY bt1.tag_id, bt2.tag_id
            HAVING shared_count >= ?
            ORDER BY shared_count DESC
            LIMIT ?
        """, (min_shared, limit)).fetchall()

        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Backfill existing beliefs
    # ------------------------------------------------------------------

    def backfill(self):
        """Backfill tags for all existing beliefs.

        1. Register structural tags from tree nodes
        2. Derive tree tags for beliefs with tree_paths
        3. Derive entity tags for beliefs with document provenance
        """
        logger.info("Starting tag backfill...")

        # Step 1: Register tree node tags
        tree_nodes = self.db.execute(
            "SELECT id, name, layer FROM knowledge_tree WHERE layer >= 2"
        ).fetchall()
        for node in tree_nodes:
            self.register_tag(
                node["name"], "structural", "tree"
            )
        logger.info(f"Registered {len(tree_nodes)} structural tags from tree")

        # Step 2: Tree-derived tags
        beliefs_with_tree = self.db.execute(
            "SELECT id, tree_paths FROM beliefs "
            "WHERE tree_paths IS NOT NULL AND tree_paths != '[]' "
            "AND COALESCE(deprecated, 0) = 0"
        ).fetchall()

        tree_tagged = 0
        for b in beliefs_with_tree:
            try:
                paths = json.loads(b["tree_paths"])
                if paths:
                    tags = self.derive_from_tree(b["id"], paths)
                    if tags:
                        tree_tagged += 1
            except (json.JSONDecodeError, TypeError):
                continue

        logger.info(f"Tree-tagged {tree_tagged} beliefs")

        # Step 3: Entity-derived tags
        beliefs_with_docs = self.db.execute(
            "SELECT DISTINCT belief_id FROM belief_sources_documents"
        ).fetchall()

        entity_tagged = 0
        for b in beliefs_with_docs:
            tags = self.derive_from_entities(b["belief_id"])
            if tags:
                entity_tagged += 1

        logger.info(f"Entity-tagged {entity_tagged} beliefs")

        # Recompute counts
        self._update_belief_counts()
        self.db.commit()

        total_tags = self.db.execute("SELECT COUNT(*) FROM tag_registry").fetchone()[0]
        total_assignments = self.db.execute("SELECT COUNT(*) FROM belief_tags").fetchone()[0]
        logger.info(
            f"Tag backfill complete: {total_tags} tags in registry, "
            f"{total_assignments} total assignments"
        )

    # ------------------------------------------------------------------
    # Tag maintenance
    # ------------------------------------------------------------------

    def merge_similar_tags(self):
        """Merge tags with embedding similarity > TAG_MERGE_THRESHOLD.

        E.g. "immune_checkpoint" and "checkpoint_therapy" → one tag.
        Keeps the tag with more beliefs, reassigns the other's beliefs.
        """
        if not self.embeddings:
            return 0

        import numpy as np

        tags = self.db.execute(
            "SELECT id, name, belief_count FROM tag_registry WHERE belief_count > 0"
        ).fetchall()

        if len(tags) < 2:
            return 0

        # Compute embeddings
        tag_embs = {}
        for t in tags:
            tag_embs[t["id"]] = (t["name"], t["belief_count"], self.embeddings.embed(t["name"]))

        merged = 0
        removed = set()

        tag_ids = list(tag_embs.keys())
        for i in range(len(tag_ids)):
            if tag_ids[i] in removed:
                continue
            for j in range(i + 1, len(tag_ids)):
                if tag_ids[j] in removed:
                    continue

                tid_a, tid_b = tag_ids[i], tag_ids[j]
                emb_a = tag_embs[tid_a][2]
                emb_b = tag_embs[tid_b][2]
                norm = np.linalg.norm(emb_a) * np.linalg.norm(emb_b)
                sim = float(np.dot(emb_a, emb_b) / norm) if norm > 0 else 0.0

                if sim >= TAG_MERGE_THRESHOLD:
                    # Keep the one with more beliefs
                    count_a = tag_embs[tid_a][1]
                    count_b = tag_embs[tid_b][1]
                    keep, drop = (tid_a, tid_b) if count_a >= count_b else (tid_b, tid_a)

                    # Reassign beliefs from drop → keep
                    self.db.execute(
                        "UPDATE OR IGNORE belief_tags SET tag_id = ? WHERE tag_id = ?",
                        (keep, drop),
                    )
                    # Remove any duplicates created by the reassignment
                    self.db.execute(
                        "DELETE FROM belief_tags WHERE tag_id = ?", (drop,)
                    )
                    self.db.execute(
                        "DELETE FROM tag_registry WHERE id = ?", (drop,)
                    )
                    removed.add(drop)
                    merged += 1
                    logger.info(
                        f"Merged tag '{tag_embs[drop][0]}' into '{tag_embs[keep][0]}' "
                        f"(sim={sim:.3f})"
                    )

        if merged:
            self._update_belief_counts()
            self.db.commit()
            logger.info(f"Merged {merged} duplicate tags")

        return merged

    def prune_dead_tags(self):
        """Remove tags with 0 beliefs for TAG_PRUNE_CYCLES+ cycles."""
        pruned = self.db.execute(
            "DELETE FROM tag_registry WHERE belief_count = 0 AND inactive_cycles >= ?",
            (TAG_PRUNE_CYCLES,),
        ).rowcount

        # Increment inactive cycle counter for 0-belief tags
        self.db.execute(
            "UPDATE tag_registry SET inactive_cycles = inactive_cycles + 1 "
            "WHERE belief_count = 0"
        )
        # Reset counter for tags with beliefs
        self.db.execute(
            "UPDATE tag_registry SET inactive_cycles = 0 WHERE belief_count > 0"
        )
        self.db.commit()

        if pruned:
            logger.info(f"Pruned {pruned} dead tags")
        return pruned
