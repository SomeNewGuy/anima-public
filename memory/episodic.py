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

"""Episodic memory — conversation storage and retrieval via ChromaDB + SQLite."""

import os
import json
import sqlite3
import uuid
from datetime import datetime, timezone

import chromadb


class EpisodicMemory:
    def __init__(self, config):
        self.config = config
        self.chroma_client = None
        self.collection = None
        self.db_conn = None

    def initialize(self):
        # Use ANIMA_DATA_DIR if set, otherwise fall back to repo-relative path
        data_dir = os.environ.get("ANIMA_DATA_DIR")
        if data_dir:
            chroma_dir = os.path.normpath(os.path.join(data_dir, "chroma"))
        else:
            base_dir = os.path.join(os.path.dirname(__file__), "..")
            chroma_dir = os.path.normpath(
                os.path.join(base_dir, self.config["memory"]["chroma_persist_dir"])
            )
        os.makedirs(chroma_dir, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = self.chroma_client.get_or_create_collection(
            name="episodes",
            metadata={"hnsw:space": "cosine"},
        )

        # SQLite for structured metadata
        if data_dir:
            sqlite_path = os.path.normpath(
                os.path.join(data_dir, "sqlite", "persistence.db")
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
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                participants TEXT NOT NULL,  -- JSON array
                summary TEXT,
                key_insights TEXT,  -- JSON array
                topics TEXT,  -- JSON array
                entities TEXT,  -- JSON array
                turn_count INTEGER DEFAULT 0,
                importance REAL DEFAULT 0.5,
                summarized INTEGER DEFAULT 0,  -- 0=unconsolidated, 1=consolidated
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS turns (
                id TEXT PRIMARY KEY,
                episode_id TEXT NOT NULL,
                role TEXT NOT NULL,  -- 'user' or 'assistant'
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                turn_index INTEGER NOT NULL,
                entities TEXT,  -- JSON array
                FOREIGN KEY (episode_id) REFERENCES episodes(id)
            );

            CREATE TABLE IF NOT EXISTS corrections (
                id TEXT PRIMARY KEY,
                episode_id TEXT NOT NULL,
                turn_id TEXT,
                original_position TEXT NOT NULL,
                corrected_position TEXT NOT NULL,
                corrected_by TEXT NOT NULL,
                reasoning TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (episode_id) REFERENCES episodes(id)
            );

            CREATE INDEX IF NOT EXISTS idx_turns_episode ON turns(episode_id);
            CREATE INDEX IF NOT EXISTS idx_corrections_episode ON corrections(episode_id);
            CREATE INDEX IF NOT EXISTS idx_episodes_timestamp ON episodes(timestamp);
        """)
        self.db_conn.commit()
        self._migrate_tables()

    def _migrate_tables(self):
        """Add columns that may not exist in older databases."""
        cols = {
            r[1]
            for r in self.db_conn.execute("PRAGMA table_info(episodes)").fetchall()
        }
        if "context_type" not in cols:
            self.db_conn.execute(
                "ALTER TABLE episodes ADD COLUMN context_type TEXT DEFAULT 'operator'"
            )
            self.db_conn.commit()

    def create_episode(self, participants=None, context_type="operator"):
        """Start a new conversation episode.

        context_type: 'operator' (genuine interaction) or 'exploration' (soak/autonomous)
        """
        episode_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        participants = participants or ["jerry"]

        self.db_conn.execute(
            "INSERT INTO episodes (id, timestamp, participants, created_at, context_type) "
            "VALUES (?, ?, ?, ?, ?)",
            (episode_id, now, json.dumps(participants), now, context_type),
        )
        self.db_conn.commit()
        return episode_id

    def add_turn(self, episode_id, role, content, entities=None):
        """Add a conversation turn to an episode."""
        turn_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # Get next turn index
        row = self.db_conn.execute(
            "SELECT COALESCE(MAX(turn_index), -1) + 1 FROM turns WHERE episode_id = ?",
            (episode_id,),
        ).fetchone()
        turn_index = row[0]

        self.db_conn.execute(
            "INSERT INTO turns (id, episode_id, role, content, timestamp, turn_index, entities) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (turn_id, episode_id, role, content, now, turn_index, json.dumps(entities or [])),
        )

        # Update episode turn count
        self.db_conn.execute(
            "UPDATE episodes SET turn_count = turn_count + 1 WHERE id = ?",
            (episode_id,),
        )
        self.db_conn.commit()

        # Index in ChromaDB for vector search
        self.collection.add(
            ids=[turn_id],
            documents=[content],
            metadatas=[{
                "episode_id": episode_id,
                "role": role,
                "turn_index": turn_index,
                "timestamp": now,
            }],
        )

        return turn_id

    def search_similar(self, query_text, n_results=10):
        """Search for similar turns using vector similarity."""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
        )
        return results

    def get_episode_turns(self, episode_id):
        """Get all turns for an episode in order."""
        rows = self.db_conn.execute(
            "SELECT * FROM turns WHERE episode_id = ? ORDER BY turn_index",
            (episode_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_recent_episodes(self, limit=10):
        """Get the most recent episodes."""
        rows = self.db_conn.execute(
            "SELECT * FROM episodes ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def update_episode_metadata(self, episode_id, summary=None, key_insights=None,
                                 topics=None, entities=None, importance=None):
        """Update episode metadata after consolidation."""
        updates = []
        params = []
        if summary is not None:
            updates.append("summary = ?")
            params.append(summary)
        if key_insights is not None:
            updates.append("key_insights = ?")
            params.append(json.dumps(key_insights))
        if topics is not None:
            updates.append("topics = ?")
            params.append(json.dumps(topics))
        if entities is not None:
            updates.append("entities = ?")
            params.append(json.dumps(entities))
        if importance is not None:
            updates.append("importance = ?")
            params.append(importance)

        if updates:
            params.append(episode_id)
            self.db_conn.execute(
                f"UPDATE episodes SET {', '.join(updates)} WHERE id = ?",
                params,
            )
            self.db_conn.commit()

    def get_corrections(self, episode_id=None):
        """Get corrections, optionally filtered by episode."""
        if episode_id:
            rows = self.db_conn.execute(
                "SELECT * FROM corrections WHERE episode_id = ? ORDER BY timestamp",
                (episode_id,),
            ).fetchall()
        else:
            rows = self.db_conn.execute(
                "SELECT * FROM corrections ORDER BY timestamp DESC",
            ).fetchall()
        return [dict(r) for r in rows]

    def add_correction(self, episode_id, original_position, corrected_position,
                       corrected_by, reasoning=None, turn_id=None):
        """Record a correction made during conversation."""
        correction_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        self.db_conn.execute(
            "INSERT INTO corrections (id, episode_id, turn_id, original_position, "
            "corrected_position, corrected_by, reasoning, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (correction_id, episode_id, turn_id, original_position,
             corrected_position, corrected_by, reasoning, now),
        )
        self.db_conn.commit()
        return correction_id

    def get_unconsolidated_episodes(self, min_turns=3):
        """Get episodes that haven't been consolidated yet, with enough turns to be worth it."""
        rows = self.db_conn.execute(
            "SELECT * FROM episodes WHERE summarized = 0 AND turn_count >= ? "
            "ORDER BY created_at",
            (min_turns,),
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_episode_consolidated(self, episode_id):
        """Mark an episode as consolidated (summarized = 1)."""
        self.db_conn.execute(
            "UPDATE episodes SET summarized = 1 WHERE id = ?",
            (episode_id,),
        )
        self.db_conn.commit()

    def mark_episode_unconsolidated(self, episode_id):
        """Mark an episode for reconsolidation (summarized = 0)."""
        self.db_conn.execute(
            "UPDATE episodes SET summarized = 0 WHERE id = ?",
            (episode_id,),
        )
        self.db_conn.commit()

    def get_episode_count(self):
        row = self.db_conn.execute("SELECT COUNT(*) FROM episodes").fetchone()
        return row[0]

    def get_turn_count(self):
        row = self.db_conn.execute("SELECT COUNT(*) FROM turns").fetchone()
        return row[0]

    def close(self):
        if self.db_conn:
            self.db_conn.close()
