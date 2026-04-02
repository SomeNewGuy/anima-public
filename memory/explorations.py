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

"""Exploration memory — stores autonomous exploration findings from free time.

Explorations are stored as accepted by default. ANIMA surfaces findings
organically in conversation rather than through an approval queue.
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone

logger = __import__("logging").getLogger("explorations")

VALID_STATUSES = {"preliminary", "accepted", "rejected"}


class ExplorationMemory:
    def __init__(self, config):
        self.config = config
        self.db_conn = None

    def initialize(self):
        # Use ANIMA_DATA_DIR if set, otherwise fall back to repo-relative path
        _data_dir = os.environ.get("ANIMA_DATA_DIR")
        base_dir = os.path.join(os.path.dirname(__file__), "..")
        if _data_dir:
            sqlite_path = os.path.normpath(
                os.path.join(_data_dir, "sqlite", "persistence.db")
            )
        else:
            sqlite_path = os.path.normpath(
                os.path.join(base_dir, self.config["memory"]["sqlite_path"])
            )
        os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
        self.db_conn = sqlite3.connect(sqlite_path, check_same_thread=False)
        self.db_conn.row_factory = sqlite3.Row
        self.db_conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()
        self._migrate_tables()
        return self

    def _create_tables(self):
        self.db_conn.executescript("""
            CREATE TABLE IF NOT EXISTS explorations (
                id TEXT PRIMARY KEY,
                trigger_type TEXT NOT NULL,
                trigger_id TEXT NOT NULL,
                trigger_text TEXT NOT NULL,
                topic_key TEXT,
                revisit_count INTEGER DEFAULT 0,
                queries TEXT,
                raw_results TEXT,
                findings TEXT,
                reflection TEXT,
                new_questions TEXT,
                search_used INTEGER DEFAULT 0,
                internal_confidence REAL,
                status TEXT DEFAULT 'preliminary',
                episode_id TEXT,
                created_at TEXT NOT NULL,
                reviewed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS exploration_links (
                id TEXT PRIMARY KEY,
                exploration_id TEXT NOT NULL,
                belief_id TEXT NOT NULL,
                link_type TEXT NOT NULL,
                strength REAL NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_explorations_status
                ON explorations(status);
            CREATE INDEX IF NOT EXISTS idx_explorations_topic_key
                ON explorations(topic_key);
            CREATE INDEX IF NOT EXISTS idx_exploration_links_exploration
                ON exploration_links(exploration_id);
            CREATE INDEX IF NOT EXISTS idx_exploration_links_belief
                ON exploration_links(belief_id);
        """)
        self.db_conn.commit()

    def _migrate_tables(self):
        """Add columns to existing tables (idempotent via try/except)."""
        migrations = [
            "ALTER TABLE explorations ADD COLUMN domain TEXT",
            "ALTER TABLE explorations ADD COLUMN priority_ai TEXT DEFAULT 'normal'",
            "ALTER TABLE explorations ADD COLUMN priority_operator TEXT",
        ]
        for sql in migrations:
            try:
                self.db_conn.execute(sql)
                self.db_conn.commit()
            except sqlite3.OperationalError:
                pass  # column already exists

    @staticmethod
    def _make_topic_key(text):
        """Normalize topic text to a key for matching revisits."""
        return text.strip().lower()

    def add_exploration(self, trigger_type, trigger_id, trigger_text,
                        queries, raw_results, findings, reflection,
                        new_questions, search_used=False,
                        internal_confidence=None, episode_id=None,
                        status="accepted", domain=None, priority_ai="normal"):
        """Store a new exploration. Returns exploration ID."""
        exploration_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        topic_key = self._make_topic_key(trigger_text)

        # Check for prior explorations on same topic
        row = self.db_conn.execute(
            "SELECT MAX(revisit_count) FROM explorations WHERE topic_key = ?",
            (topic_key,),
        ).fetchone()
        revisit_count = (row[0] or 0) + 1 if row and row[0] is not None else 0

        self.db_conn.execute(
            "INSERT INTO explorations "
            "(id, trigger_type, trigger_id, trigger_text, topic_key, "
            "revisit_count, queries, raw_results, findings, reflection, "
            "new_questions, search_used, internal_confidence, status, "
            "episode_id, created_at, domain, priority_ai) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                exploration_id, trigger_type, trigger_id, trigger_text,
                topic_key, revisit_count,
                json.dumps(queries or []),
                json.dumps(raw_results or []),
                findings, reflection,
                json.dumps(new_questions or []),
                1 if search_used else 0,
                internal_confidence,
                status,
                episode_id, now, domain, priority_ai,
            ),
        )
        self.db_conn.commit()

        logger.info(
            f"Exploration stored: {exploration_id[:8]} "
            f"topic_key='{topic_key[:40]}' revisit={revisit_count} "
            f"search_used={search_used} priority={priority_ai}"
        )
        return exploration_id

    def add_exploration_link(self, exploration_id, belief_id, link_type, strength):
        """Store a link between an exploration and a belief."""
        link_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        self.db_conn.execute(
            "INSERT INTO exploration_links "
            "(id, exploration_id, belief_id, link_type, strength, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (link_id, exploration_id, belief_id, link_type, strength, now),
        )
        self.db_conn.commit()
        return link_id

    def get_exploration_links(self, exploration_id):
        """Get all belief links for an exploration."""
        rows = self.db_conn.execute(
            "SELECT * FROM exploration_links WHERE exploration_id = ? "
            "ORDER BY strength DESC",
            (exploration_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_belief_explorations(self, belief_id):
        """Get all explorations linked to a belief."""
        rows = self.db_conn.execute(
            "SELECT el.*, e.trigger_text, e.findings "
            "FROM exploration_links el "
            "JOIN explorations e ON el.exploration_id = e.id "
            "WHERE el.belief_id = ? ORDER BY el.strength DESC",
            (belief_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def set_operator_priority(self, exploration_id, priority):
        """Set operator priority on an exploration."""
        if priority not in ("normal", "important", "critical"):
            raise ValueError(f"Invalid priority: {priority}")
        self.db_conn.execute(
            "UPDATE explorations SET priority_operator = ? WHERE id = ?",
            (priority, exploration_id),
        )
        self.db_conn.commit()

    def get_pending(self, limit=10):
        """Get explorations awaiting review."""
        rows = self.db_conn.execute(
            "SELECT * FROM explorations WHERE status = 'preliminary' "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_exploration(self, exploration_id):
        """Get a single exploration by ID."""
        row = self.db_conn.execute(
            "SELECT * FROM explorations WHERE id = ?",
            (exploration_id,),
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def review_exploration(self, exploration_id, status):
        """Accept or reject an exploration."""
        if status not in ("accepted", "rejected"):
            raise ValueError(f"Invalid review status: {status}")
        now = datetime.now(timezone.utc).isoformat()
        self.db_conn.execute(
            "UPDATE explorations SET status = ?, reviewed_at = ? WHERE id = ?",
            (status, now, exploration_id),
        )
        self.db_conn.commit()
        logger.info(f"Exploration {exploration_id[:8]} {status}")

    def get_accepted(self, limit=20):
        """Get accepted explorations."""
        rows = self.db_conn.execute(
            "SELECT * FROM explorations WHERE status = 'accepted' "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_recent_topic_keys(self, limit=10):
        """Get distinct recent exploration topic keys (for dedup/recency penalty)."""
        rows = self.db_conn.execute(
            "SELECT DISTINCT topic_key FROM explorations "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [r["topic_key"] for r in rows]

    def get_all_topic_keys(self):
        """Get all distinct topic keys ever explored (for novel topic dedup)."""
        rows = self.db_conn.execute(
            "SELECT DISTINCT topic_key FROM explorations"
        ).fetchall()
        return set(r["topic_key"] for r in rows)

    def get_exploration_count(self, status=None):
        """Count explorations, optionally filtered by status."""
        if status:
            row = self.db_conn.execute(
                "SELECT COUNT(*) FROM explorations WHERE status = ?",
                (status,),
            ).fetchone()
        else:
            row = self.db_conn.execute(
                "SELECT COUNT(*) FROM explorations"
            ).fetchone()
        return row[0]

    def close(self):
        if self.db_conn:
            self.db_conn.close()
            self.db_conn = None

    @staticmethod
    def _row_to_dict(row):
        """Convert sqlite3.Row to dict with parsed JSON fields."""
        d = dict(row)
        for field in ("queries", "raw_results", "new_questions"):
            val = d.get(field)
            if val:
                try:
                    d[field] = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    d[field] = []
            else:
                d[field] = []
        d["search_used"] = bool(d.get("search_used", 0))
        return d
