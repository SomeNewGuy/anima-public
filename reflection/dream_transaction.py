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

"""Transactional dream buffer — captures graph writes in memory, commits atomically.

All graph-modifying operations from dream processing (add_belief, add_belief_link,
reinforce_or_reactivate_link, check_dormant_adjacency) are buffered. Diagnostic writes
(dms_triage_log, rejection_history, rejected_beliefs_log) bypass the buffer.

Usage:
    txn = DreamTransaction(semantic, embeddings)
    txn.expect(belief_created=True, link_count=3)
    belief_id = txn.add_belief(statement=..., ...)
    txn.add_belief_link(belief_a=belief_id, ...)

    ok, errors = txn.validate()
    if ok:
        txn.commit()
    else:
        txn.rollback()
"""

import logging
from collections import namedtuple

logger = logging.getLogger("dream_transaction")

_PROVISIONAL_PREFIX = "__prov_"

DreamOp = namedtuple("DreamOp", ["op_type", "args", "kwargs", "provisional_id"])


class DreamTransaction:
    """Write buffer for dream graph operations with atomic commit.

    Provides the same write API as SemanticMemory for graph-modifying ops.
    Reads delegate to the real semantic store. Writes are buffered and
    replayed atomically on commit().

    Provisional IDs (prefixed __prov_) are returned from add_belief() and
    resolved to real IDs during commit. They must NEVER escape the transaction.
    """

    def __init__(self, semantic, embeddings=None):
        self.semantic = semantic
        self.embeddings = embeddings
        self._ops = []
        self._provisional_ids = {}  # provisional_id -> real_id (populated on commit)
        self._next_provisional = 0
        self._deferred_callbacks = []

        # Expected op validation
        self._expected_belief_created = False
        self._expected_link_count = None
        self._actual_belief_count = 0
        self._actual_link_count = 0

    # ------------------------------------------------------------------
    # Expectation declaration
    # ------------------------------------------------------------------

    def expect(self, belief_created=False, link_count=None):
        """Declare expected structure for this dream's transaction.

        Called by the dream processing code to declare what a valid
        transaction should contain. validate() checks actual ops against
        these expectations.

        Args:
            belief_created: True if a new synthesis belief should be created
            link_count: expected number of belief links (edges)
        """
        self._expected_belief_created = belief_created
        self._expected_link_count = link_count

    # ------------------------------------------------------------------
    # Deferred callbacks
    # ------------------------------------------------------------------

    def defer_callback(self, fn):
        """Register a function to run after successful commit.

        Callbacks execute in registration order. Failures are logged but
        do not roll back the committed transaction. Use for non-critical
        side effects (curiosity gaps, tag inheritance).
        """
        self._deferred_callbacks.append(fn)

    # ------------------------------------------------------------------
    # Write API (buffered)
    # ------------------------------------------------------------------

    def add_belief(self, **kwargs):
        """Buffer a belief creation. Returns a provisional ID."""
        prov_id = f"{_PROVISIONAL_PREFIX}{self._next_provisional:04d}__"
        self._next_provisional += 1
        self._ops.append(DreamOp("add_belief", (), kwargs, prov_id))
        self._actual_belief_count += 1
        return prov_id

    def add_belief_link(self, belief_a, belief_b, inference, similarity=None,
                        link_type="dream"):
        """Buffer a belief link creation. Accepts provisional IDs."""
        kwargs = {
            "belief_a": belief_a,
            "belief_b": belief_b,
            "inference": inference,
            "similarity": similarity,
            "link_type": link_type,
        }
        self._ops.append(DreamOp("add_belief_link", (), kwargs, None))
        self._actual_link_count += 1

    def reinforce_or_reactivate_link(self, belief_a, belief_b, new_similarity):
        """Buffer a link reinforcement. Accepts provisional IDs."""
        kwargs = {
            "belief_a": belief_a,
            "belief_b": belief_b,
            "new_similarity": new_similarity,
        }
        self._ops.append(DreamOp("reinforce_or_reactivate_link", (), kwargs, None))

    def check_dormant_adjacency(self, belief_id, embedder=None,
                                trigger_type="new_belief", threshold=None):
        """Buffer a dormant adjacency check. Deferred to commit time."""
        kwargs = {
            "belief_id": belief_id,
            "embedder": embedder,
            "trigger_type": trigger_type,
            "threshold": threshold,
        }
        self._ops.append(DreamOp("check_dormant_adjacency", (), kwargs, None))

    # ------------------------------------------------------------------
    # Read-through API (delegates to real semantic store)
    # ------------------------------------------------------------------

    @property
    def db_conn(self):
        """Read-through to real DB connection for queries during dream processing."""
        return self.semantic.db_conn

    def get_belief_depth(self, belief_id):
        """Delegate depth lookup to real semantic store."""
        return self.semantic.get_belief_depth(belief_id)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self):
        """Pre-commit consistency checks.

        Returns (ok: bool, errors: list[str]).
        Checks:
        1. No unresolved provisional IDs in link ops
        2. Expected ops match actual ops
        3. All provisional IDs created by add_belief ops
        """
        errors = []

        # Collect all provisional IDs created by add_belief ops
        created_provisionals = {
            op.provisional_id for op in self._ops if op.op_type == "add_belief"
        }

        # Check all provisional IDs in link ops reference created beliefs
        for op in self._ops:
            if op.op_type in ("add_belief_link", "reinforce_or_reactivate_link",
                              "check_dormant_adjacency"):
                for key in ("belief_a", "belief_b", "belief_id"):
                    val = op.kwargs.get(key)
                    if val and str(val).startswith(_PROVISIONAL_PREFIX):
                        if val not in created_provisionals:
                            errors.append(
                                f"Unresolved provisional ID '{val}' in "
                                f"{op.op_type}.{key}"
                            )

        # Expected op validation
        if self._expected_belief_created and self._actual_belief_count == 0:
            errors.append("Expected belief creation but none buffered")

        if (self._expected_link_count is not None
                and self._actual_link_count != self._expected_link_count):
            errors.append(
                f"Expected {self._expected_link_count} links but "
                f"{self._actual_link_count} buffered"
            )

        return (len(errors) == 0, errors)

    # ------------------------------------------------------------------
    # Commit / Rollback
    # ------------------------------------------------------------------

    def _resolve_id(self, val):
        """Resolve a provisional ID to a real ID, or return as-is."""
        if val and str(val).startswith(_PROVISIONAL_PREFIX):
            real = self._provisional_ids.get(val)
            if real is None:
                raise ValueError(f"Provisional ID '{val}' not yet resolved")
            return real
        return val

    def commit(self):
        """Replay all buffered ops atomically against the real semantic store.

        Returns True on success, False on validation failure or DB error.
        On failure, all graph writes are rolled back.
        """
        ok, errors = self.validate()
        if not ok:
            logger.error(f"Dream transaction validation failed: {errors}")
            return False

        with self.semantic.transaction():
            self.semantic.db_conn.execute("BEGIN IMMEDIATE")
            try:
                for op in self._ops:
                    self._replay_op(op)
                self.semantic.db_conn.commit()
            except Exception as e:
                self.semantic.db_conn.rollback()
                logger.error(f"Dream transaction rolled back: {e}")
                return False

        # Post-commit: deferred callbacks (graph is consistent)
        for cb in self._deferred_callbacks:
            try:
                cb()
            except Exception as e:
                logger.warning(f"Deferred dream callback failed: {e}")

        return True

    def _replay_op(self, op):
        """Execute a single buffered op against the real semantic store."""
        if op.op_type == "add_belief":
            real_id = self.semantic.add_belief(**op.kwargs)
            if real_id is None:
                raise ValueError(
                    f"add_belief returned None for provisional {op.provisional_id}: "
                    f"{op.kwargs.get('statement', '')[:60]}"
                )
            self._provisional_ids[op.provisional_id] = real_id

        elif op.op_type == "add_belief_link":
            kwargs = dict(op.kwargs)
            kwargs["belief_a"] = self._resolve_id(kwargs["belief_a"])
            kwargs["belief_b"] = self._resolve_id(kwargs["belief_b"])
            self.semantic.add_belief_link(**kwargs)

        elif op.op_type == "reinforce_or_reactivate_link":
            kwargs = dict(op.kwargs)
            kwargs["belief_a"] = self._resolve_id(kwargs["belief_a"])
            kwargs["belief_b"] = self._resolve_id(kwargs["belief_b"])
            self.semantic.reinforce_or_reactivate_link(**kwargs)

        elif op.op_type == "check_dormant_adjacency":
            kwargs = dict(op.kwargs)
            kwargs["belief_id"] = self._resolve_id(kwargs["belief_id"])
            self.semantic.check_dormant_adjacency(**kwargs)

    def rollback(self):
        """Discard all buffered ops and provisional IDs."""
        self._ops.clear()
        self._provisional_ids.clear()
        self._deferred_callbacks.clear()
        self._actual_belief_count = 0
        self._actual_link_count = 0
        self._next_provisional = 0
