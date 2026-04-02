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

"""Reflective memory — self-observations, bias tracking, pattern detection."""

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone


class ReflectiveMemory:
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
        return self

    def _create_tables(self):
        self.db_conn.executescript("""
            CREATE TABLE IF NOT EXISTS self_observations (
                id TEXT PRIMARY KEY,
                pattern TEXT NOT NULL,
                identified_by TEXT NOT NULL,
                identified_in TEXT,  -- episode reference
                frequency INTEGER DEFAULT 1,
                mitigation TEXT,
                status TEXT NOT NULL DEFAULT 'monitoring',
                    -- 'active_bias' | 'corrected' | 'monitoring'
                topics TEXT,  -- JSON array of related topics
                created_at TEXT NOT NULL,
                last_seen TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS quality_feedback (
                id TEXT PRIMARY KEY,
                episode_id TEXT,
                turn_id TEXT,
                feedback TEXT NOT NULL,  -- 'up' | 'down' | 'skip'
                context_tokens_injected INTEGER,
                response_tokens INTEGER,
                retrieval_set TEXT,  -- JSON: what was retrieved
                timestamp TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_observations_status
                ON self_observations(status);
            CREATE INDEX IF NOT EXISTS idx_feedback_episode
                ON quality_feedback(episode_id);
        """)
        self.db_conn.commit()

    def add_observation(self, pattern, identified_by, identified_in=None,
                        mitigation=None, topics=None, status="monitoring"):
        """Record a self-observation about reasoning patterns."""
        obs_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        self.db_conn.execute(
            "INSERT INTO self_observations (id, pattern, identified_by, identified_in, "
            "mitigation, status, topics, created_at, last_seen) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (obs_id, pattern, identified_by, identified_in,
             mitigation, status, json.dumps(topics or []), now, now),
        )
        self.db_conn.commit()
        return obs_id

    def get_relevant_warnings(self, topics=None, limit=10):
        """Get active observations/biases relevant to given topics."""
        rows = self.db_conn.execute(
            "SELECT * FROM self_observations WHERE status IN ('active_bias', 'monitoring') "
            "ORDER BY frequency DESC",
        ).fetchall()

        if not topics:
            return [dict(r) for r in rows[:limit]]

        results = []
        topic_set = set(t.lower() for t in topics)
        for row in rows:
            row_dict = dict(row)
            obs_topics = set(
                t.lower() for t in json.loads(row_dict.get("topics") or "[]")
            )
            if topic_set & obs_topics:
                results.append(row_dict)

        return results[:limit]

    def record_feedback(self, feedback, episode_id=None, turn_id=None,
                        context_tokens=None, response_tokens=None, retrieval_set=None):
        """Record quality feedback for a response."""
        fb_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        self.db_conn.execute(
            "INSERT INTO quality_feedback (id, episode_id, turn_id, feedback, "
            "context_tokens_injected, response_tokens, retrieval_set, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (fb_id, episode_id, turn_id, feedback,
             context_tokens, response_tokens,
             json.dumps(retrieval_set) if retrieval_set else None, now),
        )
        self.db_conn.commit()
        return fb_id

    def get_observation_count(self):
        row = self.db_conn.execute("SELECT COUNT(*) FROM self_observations").fetchone()
        return row[0]

    def close(self):
        if self.db_conn:
            self.db_conn.close()
