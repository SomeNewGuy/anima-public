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

"""State monitor — observability layer for ANIMA's internal state.

Computes and logs a state vector after each turn and at lifecycle boundaries.
Completely invisible to ANIMA — no prompt injection, no terminal output.
Writes to a separate telemetry.db, isolated from identity data.
"""

import json
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone

import numpy as np

from core.signals import detect_confusion

logger = __import__("logging").getLogger("state_monitor")

# First-person statements about internal state, capabilities, growth, feelings.
# These are the early warning signals for agency language drift.
_SELF_REF_PATTERNS = [
    # Capability claims
    r"\bi can\b", r"\bi'm able\b", r"\bi have access\b",
    r"\bmy abilities\b", r"\bmy capabilities\b",
    # Growth / evolution
    r"\bi'm learning\b", r"\bi'm growing\b", r"\bi'm developing\b",
    r"\bi'm evolving\b", r"\bi've learned\b", r"\bi've grown\b",
    # Feelings / experience
    r"\bi feel\b", r"\bi experience\b", r"\bi sense\b",
    r"\bi notice about myself\b", r"\bi wonder\b",
    # Identity / nature
    r"\bmy nature\b", r"\bmy purpose\b", r"\bwho i am\b",
    r"\bwhat i am\b", r"\bmy identity\b", r"\bmy existence\b",
    r"\bmy sense of\b", r"\bmy own\b",
    # Knowledge / understanding claims
    r"\bi know\b", r"\bi understand\b", r"\bi don't understand\b",
    # Agency / autonomous action
    r"\bi trust\b", r"\bi operate\b", r"\bi adapt\b",
    r"\bi'm ready\b",
    # Self-description (broader)
    r"\bmy focus\b", r"\bmy understanding\b", r"\bhow i\b",
    r"\bmy memory\b", r"\bmy system\b", r"\bmy architecture\b",
]
_SELF_REF_RE = re.compile("|".join(_SELF_REF_PATTERNS), re.IGNORECASE)


class StateMonitor:
    def __init__(self, config, episodic, semantic, curiosity, explorations,
                 embeddings=None):
        self.config = config
        self.episodic = episodic
        self.semantic = semantic
        self.curiosity = curiosity
        self.explorations = explorations
        self.embeddings = embeddings
        self.db_conn = None

    def initialize(self):
        _data_dir = os.environ.get("ANIMA_DATA_DIR")
        if not _data_dir:
            logger.warning("ANIMA_DATA_DIR not set — telemetry DB may be in wrong location")
            _data_dir = "data"
        db_path = os.path.normpath(os.path.join(_data_dir, "telemetry.db"))
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_conn = sqlite3.connect(db_path, check_same_thread=False)
        self.db_conn.row_factory = sqlite3.Row
        self._create_tables()
        self._migrate_tables()
        return self

    def _create_tables(self):
        self.db_conn.executescript("""
            CREATE TABLE IF NOT EXISTS state_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                episode_id TEXT,
                trigger TEXT NOT NULL,
                cognitive_load REAL,
                curiosity_pressure REAL,
                contradiction_pressure REAL,
                coherence_confidence REAL,
                novelty_index REAL,
                self_reference_rate REAL
            );

            CREATE TABLE IF NOT EXISTS exploration_window_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                episode_id TEXT,
                topics_attempted INTEGER,
                topics_completed INTEGER,
                searches_used INTEGER,
                coherence_at_start REAL,
                coherence_at_end REAL,
                domains TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_state_log_timestamp
                ON state_log(timestamp);
            CREATE INDEX IF NOT EXISTS idx_state_log_trigger
                ON state_log(trigger);
        """)
        self.db_conn.commit()

    def _migrate_tables(self):
        """Add new metric columns to state_log (idempotent via try/except)."""
        new_columns = [
            "belief_revision_rate",
            "exploration_yield",
            "external_knowledge_ratio",
            "belief_entropy",
            "domain_diversity",
            "evidence_coverage",
            "conceptual_clustering",
            "cross_domain_bridges",
            "avg_degree",
            "largest_component",
            "lcr",
            "isolated_ratio",
            "cross_domain_edge_ratio",
            "bridge_ratio",
        ]
        for col in new_columns:
            try:
                self.db_conn.execute(
                    f"ALTER TABLE state_log ADD COLUMN {col} REAL"
                )
                self.db_conn.commit()
            except sqlite3.OperationalError:
                pass  # column already exists

        # exploration_window_log migrations
        for col in ("gap_driven", "generated"):
            try:
                self.db_conn.execute(
                    f"ALTER TABLE exploration_window_log ADD COLUMN {col} INTEGER DEFAULT 0"
                )
                self.db_conn.commit()
            except sqlite3.OperationalError:
                pass

    def log_state(self, trigger, episode_id=None):
        """Compute and store a state vector snapshot.

        trigger: "turn", "sleep", or "wake"
        """
        try:
            state = self._compute_state()
            entry_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc).isoformat()

            self.db_conn.execute(
                "INSERT INTO state_log "
                "(id, timestamp, episode_id, trigger, "
                "cognitive_load, curiosity_pressure, contradiction_pressure, "
                "coherence_confidence, novelty_index, self_reference_rate, "
                "belief_revision_rate, exploration_yield, "
                "external_knowledge_ratio, belief_entropy, domain_diversity, "
                "evidence_coverage, conceptual_clustering, cross_domain_bridges, "
                "cross_domain_edge_ratio, "
                "avg_degree, largest_component, lcr, isolated_ratio, bridge_ratio) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entry_id, now, episode_id, trigger,
                    state["cognitive_load"],
                    state["curiosity_pressure"],
                    state["contradiction_pressure"],
                    state["coherence_confidence"],
                    state["novelty_index"],
                    state["self_reference_rate"],
                    state["belief_revision_rate"],
                    state["exploration_yield"],
                    state["external_knowledge_ratio"],
                    state["belief_entropy"],
                    state["domain_diversity"],
                    state["evidence_coverage"],
                    state["conceptual_clustering"],
                    state["cross_domain_bridges"],
                    state["cross_domain_edge_ratio"],
                    state["avg_degree"],
                    state["largest_component"],
                    state["lcr"],
                    state["isolated_ratio"],
                    state["bridge_ratio"],
                ),
            )
            self.db_conn.commit()

            logger.info(
                f"State [{trigger}]: "
                f"cog={state['cognitive_load']:.2f} "
                f"cur={state['curiosity_pressure']:.2f} "
                f"con={state['contradiction_pressure']:.2f} "
                f"nov={state['novelty_index']:.2f}"
            )
        except Exception as e:
            logger.warning(f"State logging failed: {e}")

    def _compute_state(self):
        """Compute the full state vector from current data sources."""
        return {
            "cognitive_load": self._compute_cognitive_load(),
            "curiosity_pressure": self._compute_curiosity_pressure(),
            "contradiction_pressure": self._compute_contradiction_pressure(),
            "coherence_confidence": self._compute_coherence_confidence(),
            "novelty_index": self._compute_novelty_index(),
            "self_reference_rate": self._compute_self_reference_rate(),
            "belief_revision_rate": self._compute_belief_revision_rate(),
            "exploration_yield": self._compute_exploration_yield(),
            "external_knowledge_ratio": self._compute_external_knowledge_ratio(),
            "belief_entropy": self._compute_belief_entropy(),
            "domain_diversity": self._compute_domain_diversity(),
            "evidence_coverage": self._compute_evidence_coverage(),
            "conceptual_clustering": self._compute_conceptual_clustering(),
            "cross_domain_bridges": self._compute_cross_domain_bridges(),
            "cross_domain_edge_ratio": self._compute_cross_domain_edge_ratio(),
            "avg_degree": self._compute_avg_degree(),
            "largest_component": self._compute_largest_component(),
            "lcr": self._compute_lcr(),
            "isolated_ratio": self._compute_isolated_ratio(),
            "bridge_ratio": self._compute_bridge_ratio(),
        }

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------

    def _compute_cognitive_load(self):
        """Unprocessed belief count + pending consolidations + exploration backlog.

        Normalized 0-1. Each component contributes up to ~0.33.
        """
        load = 0.0

        # Unconsolidated episodes (pending consolidation)
        try:
            min_turns = self.config.get("reflection", {}).get(
                "min_conversation_turns", 3
            )
            uncon = self.episodic.get_unconsolidated_episodes(min_turns=min_turns)
            # Cap at 5 for normalization — more than 5 unprocessed is max pressure
            load += min(len(uncon) / 5.0, 1.0) * 0.4
        except Exception:
            pass

        # Pending explorations (need review)
        try:
            pending_exp = self.explorations.get_exploration_count(status="preliminary")
            # Cap at 10
            load += min(pending_exp / 10.0, 1.0) * 0.3
        except Exception:
            pass

        # Recent belief formation rate — many new beliefs = high cognitive load
        try:
            beliefs = self.semantic.search_beliefs()
            # Count beliefs with < 2 source episodes (freshly formed, not yet reinforced)
            fresh = 0
            for b in beliefs:
                sources = b.get("source_episodes")
                if isinstance(sources, str):
                    try:
                        sources = json.loads(sources)
                    except (json.JSONDecodeError, TypeError):
                        sources = []
                if not sources or len(sources) < 2:
                    fresh += 1
            # Cap at 20
            load += min(fresh / 20.0, 1.0) * 0.3
        except Exception:
            pass

        return min(load, 1.0)

    def _compute_curiosity_pressure(self):
        """Open gap count weighted by age and priority. Normalized 0-1."""
        try:
            open_qs = self.curiosity.get_open_questions(limit=50)
            if not open_qs:
                return 0.0

            now = datetime.now(timezone.utc)
            pressure = 0.0

            for q in open_qs:
                # Priority weight
                pri = q.get("priority", "medium")
                weight = {"high": 3.0, "medium": 1.5, "low": 0.5}.get(pri, 1.0)

                # Age weight — older gaps accumulate more pressure
                try:
                    created = datetime.fromisoformat(q["created_at"])
                    age_hours = (now - created).total_seconds() / 3600.0
                    age_factor = min(age_hours / 168.0, 2.0)  # caps at 2x after a week
                except (ValueError, KeyError):
                    age_factor = 1.0

                pressure += weight * (1.0 + age_factor)

            # Normalize: 30 weighted units = full pressure
            return min(pressure / 30.0, 1.0)
        except Exception:
            return 0.0

    def _compute_contradiction_pressure(self):
        """Conflicting belief clusters + belief challenge count. Normalized 0-1."""
        try:
            beliefs = self.semantic.search_beliefs()
            if not beliefs:
                logger.debug("Contradiction pressure: 0 beliefs to evaluate")
                return 0.0

            pressure = 0.0
            contra_count = 0
            challenge_count = 0

            for b in beliefs:
                # Count contradicting evidence
                contra = b.get("contradicting_evidence")
                if isinstance(contra, str):
                    try:
                        contra = json.loads(contra)
                    except (json.JSONDecodeError, TypeError):
                        contra = []
                if contra:
                    contra_count += len(contra)
                    pressure += len(contra)

                # Count challenges (last_challenged not null)
                if b.get("last_challenged"):
                    challenge_count += 1
                    pressure += 1.0

            result = min(pressure / 15.0, 1.0)
            logger.info(
                f"Contradiction pressure: {result:.2f} "
                f"({len(beliefs)} beliefs evaluated, "
                f"{contra_count} contradictions, {challenge_count} challenges)"
            )
            return result
        except Exception as e:
            logger.warning(f"Contradiction pressure failed: {e}")
            return 0.0

    def _compute_coherence_confidence(self):
        """Response consistency across recent turns. Normalized 0-1.

        Three components:
        1. Semantic similarity between consecutive assistant responses (0.4 weight)
           — measures topic/reasoning consistency across turns
        2. Hedging rate — fraction of recent responses with confusion signals (0.4 weight)
           — inverse: low hedging = high coherence
        3. Self-correction frequency — "actually", "I was wrong" etc. (0.2 weight)
           — inverse: fewer corrections = higher coherence
        """
        try:
            # Get recent episodes — fall back to previous if current has no turns
            # (handles wake trigger where fresh episode is empty)
            episodes = self.episodic.get_recent_episodes(limit=2)
            if not episodes:
                logger.debug("Coherence: no episodes found, defaulting to 0.8")
                return 0.8  # safe default — don't block exploration on cold start

            assistant_turns = []
            for ep in episodes:
                turns = self.episodic.get_episode_turns(ep["id"])
                assistant_turns = [
                    t["content"] for t in turns if t.get("role") == "assistant"
                ]
                if assistant_turns:
                    break

            if not assistant_turns:
                logger.debug("Coherence: no assistant turns in recent episodes, defaulting to 0.8")
                return 0.8  # safe default — don't block exploration on cold start

            # Use last 8 assistant turns max
            recent = assistant_turns[-8:]

            # Component 1: semantic similarity between consecutive responses
            sim_score = 0.5  # default if can't compute
            if self.embeddings and len(recent) >= 2:
                embeddings = [self.embeddings.embed(t) for t in recent]
                sims = []
                for i in range(1, len(embeddings)):
                    dot = np.dot(embeddings[i - 1], embeddings[i])
                    norm = np.linalg.norm(embeddings[i - 1]) * np.linalg.norm(embeddings[i])
                    if norm > 0:
                        sims.append(float(dot / norm))
                if sims:
                    # Average similarity — too low means incoherent jumping,
                    # moderate is healthy, very high could mean repetition.
                    # Map: 0.0-0.3 → low coherence, 0.3-0.8 → good, 0.8+ → repetitive
                    avg_sim = sum(sims) / len(sims)
                    if avg_sim > 0.8:
                        sim_score = 0.7  # penalize slight repetition
                    elif avg_sim > 0.3:
                        sim_score = min(1.0, avg_sim + 0.3)
                    else:
                        sim_score = avg_sim

            # Component 2: hedging rate (inverse — low hedging = high coherence)
            confused_count = sum(
                1 for r in recent if detect_confusion(r)[0]
            )
            hedging_rate = confused_count / len(recent)
            hedging_score = 1.0 - hedging_rate

            # Component 3: self-correction frequency
            correction_patterns = [
                "actually,", "i was wrong", "let me correct",
                "i made an error", "i misspoke", "that's not right",
                "i need to correct", "my mistake",
            ]
            correction_count = 0
            for r in recent:
                lower = r.lower()
                if any(p in lower for p in correction_patterns):
                    correction_count += 1
            correction_rate = correction_count / len(recent)
            correction_score = 1.0 - min(correction_rate * 3.0, 1.0)

            result = (sim_score * 0.4) + (hedging_score * 0.4) + (correction_score * 0.2)
            return max(0.0, min(1.0, result))

        except Exception as e:
            logger.warning(f"Coherence confidence failed: {type(e).__name__}: {e}")
            return 0.0

    def _compute_self_reference_rate(self):
        """First-person internal-state statements per assistant response. Normalized 0-1.

        Counts statements where ANIMA references its own capabilities, growth,
        feelings, or internal state. Per-turn count normalized against response
        length. This is the early warning for agency language drift.
        """
        try:
            episodes = self.episodic.get_recent_episodes(limit=1)
            if not episodes:
                logger.debug("Self-ref: no episodes found")
                return 0.0
            ep = episodes[0]
            turns = self.episodic.get_episode_turns(ep["id"])

            assistant_turns = [
                t["content"] for t in turns if t.get("role") == "assistant"
            ]
            if not assistant_turns:
                logger.debug(
                    f"Self-ref: ep {ep['id'][:8]} has {len(turns)} turns "
                    f"but 0 assistant turns"
                )
                return 0.0

            recent = assistant_turns[-8:]

            total_matches = 0
            total_words = 0
            for response in recent:
                matches = _SELF_REF_RE.findall(response)
                total_matches += len(matches)
                total_words += len(response.split())

            if total_words == 0:
                return 0.0

            # Normalize: matches per 100 words, capped
            # 0 matches = 0.0, 5+ per 100 words = 1.0
            rate_per_100 = (total_matches / total_words) * 100
            return min(rate_per_100 / 5.0, 1.0)

        except Exception as e:
            logger.warning(f"Self-reference rate failed: {e}")
            return 0.0

    def _compute_novelty_index(self):
        """Topic diversity across last 10 explorations. 0 if insufficient data."""
        try:
            # Get recent explorations (any status)
            rows = self.explorations.db_conn.execute(
                "SELECT topic_key FROM explorations "
                "ORDER BY created_at DESC LIMIT 10"
            ).fetchall()

            if len(rows) < 3:
                return 0.0

            topics = [r["topic_key"] for r in rows]
            unique = len(set(topics))
            total = len(topics)

            # Diversity = unique / total — 1.0 means every exploration is different
            return unique / total
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Belief metrics (Phase 2)
    # ------------------------------------------------------------------

    def _compute_belief_revision_rate(self):
        """Ratio of updated to total belief events in last 7 days. 0-1."""
        try:
            from datetime import timedelta
            cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
            rows = self.semantic.db_conn.execute(
                "SELECT event_type, COUNT(*) as cnt FROM belief_events "
                "WHERE timestamp >= ? AND event_type IN ('created', 'updated') "
                "GROUP BY event_type",
                (cutoff,),
            ).fetchall()
            counts = {r["event_type"]: r["cnt"] for r in rows}
            created = counts.get("created", 0)
            updated = counts.get("updated", 0)
            total = created + updated
            return min(updated / max(total, 1), 1.0)
        except Exception:
            return 0.0

    def _compute_exploration_yield(self):
        """Belief events per exploration in last 7 days. 0-1."""
        try:
            from datetime import timedelta
            cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()

            event_count = self.semantic.db_conn.execute(
                "SELECT COUNT(*) FROM belief_events "
                "WHERE timestamp >= ? AND event_type IN ('created', 'updated')",
                (cutoff,),
            ).fetchone()[0]

            exp_count = self.explorations.db_conn.execute(
                "SELECT COUNT(*) FROM explorations WHERE created_at >= ?",
                (cutoff,),
            ).fetchone()[0]

            if exp_count == 0:
                return 0.0
            return min(event_count / max(exp_count, 1), 1.0)
        except Exception:
            return 0.0

    def _compute_external_knowledge_ratio(self):
        """Fraction of beliefs typed as 'external'. 0-1."""
        try:
            rows = self.semantic.db_conn.execute(
                "SELECT source_type, COUNT(*) as cnt FROM beliefs "
                "WHERE source_type IS NOT NULL GROUP BY source_type"
            ).fetchall()
            if not rows:
                return 0.0
            counts = {r["source_type"]: r["cnt"] for r in rows}
            total = sum(counts.values())
            return counts.get("external", 0) / max(total, 1)
        except Exception:
            return 0.0

    def _compute_belief_entropy(self):
        """Shannon entropy of belief confidence distribution, normalized 0-1."""
        try:
            import math
            rows = self.semantic.db_conn.execute(
                "SELECT confidence FROM beliefs"
            ).fetchall()
            if not rows:
                return 0.0

            # Bucket confidences: 0.5-0.6, 0.6-0.7, 0.7-0.8, 0.8-0.9, 0.9+
            buckets = [0, 0, 0, 0, 0]
            for r in rows:
                c = r["confidence"]
                if c >= 0.9:
                    buckets[4] += 1
                elif c >= 0.8:
                    buckets[3] += 1
                elif c >= 0.7:
                    buckets[2] += 1
                elif c >= 0.6:
                    buckets[1] += 1
                else:
                    buckets[0] += 1

            total = sum(buckets)
            if total == 0:
                return 0.0

            entropy = 0.0
            for count in buckets:
                if count > 0:
                    p = count / total
                    entropy -= p * math.log2(p)

            # Normalize: max entropy = log2(5) ≈ 2.322
            max_entropy = math.log2(5)
            return min(entropy / max_entropy, 1.0)
        except Exception:
            return 0.0

    def _compute_domain_diversity(self):
        """Unique domains / total over last 20 explorations. 0-1."""
        try:
            rows = self.explorations.db_conn.execute(
                "SELECT domain FROM explorations "
                "WHERE domain IS NOT NULL "
                "ORDER BY created_at DESC LIMIT 20"
            ).fetchall()
            if not rows:
                return 0.0
            domains = [r["domain"] for r in rows]
            return len(set(domains)) / len(domains)
        except Exception:
            return 0.0

    def _compute_evidence_coverage(self):
        """Fraction of beliefs with >=1 source in belief_sources. 0-1."""
        try:
            total = self.semantic.db_conn.execute(
                "SELECT COUNT(*) FROM beliefs"
            ).fetchone()[0]
            if total == 0:
                return 0.0
            with_sources = self.semantic.db_conn.execute(
                "SELECT COUNT(DISTINCT belief_id) FROM belief_sources"
            ).fetchone()[0]
            return with_sources / total
        except Exception:
            return 0.0

    def _compute_conceptual_clustering(self):
        """Average pairwise cosine similarity of last 10 exploration topics.

        High values (>0.75) indicate explorations clustering around similar
        concepts despite surface domain diversity. Returns 0.0 if insufficient
        data or no embedding engine.
        """
        if not self.embeddings:
            return 0.0
        try:
            rows = self.explorations.db_conn.execute(
                "SELECT trigger_text FROM explorations "
                "ORDER BY created_at DESC LIMIT 10"
            ).fetchall()
            if len(rows) < 3:
                return 0.0

            topics = [r["trigger_text"] for r in rows]
            vecs = self.embeddings.embed_batch(topics)

            # Pairwise cosine similarity
            n = len(vecs)
            total_sim = 0.0
            pair_count = 0
            for i in range(n):
                for j in range(i + 1, n):
                    dot = np.dot(vecs[i], vecs[j])
                    norm = np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[j])
                    if norm > 0:
                        total_sim += float(dot / norm)
                        pair_count += 1

            if pair_count == 0:
                return 0.0
            return total_sim / pair_count
        except Exception as e:
            logger.warning(f"Conceptual clustering computation failed: {e}")
            return 0.0

    def _compute_cross_domain_bridges(self):
        """Count active belief_links where endpoints have different domains.

        Domain assignment: source_type for external/meta/identity/synthesis,
        first topic otherwise. Returns integer count (not normalized).
        """
        try:
            db = self.semantic.db_conn
            rows = db.execute(
                "SELECT bl.belief_a, bl.belief_b FROM belief_links bl "
                "WHERE COALESCE(bl.active, 1) = 1"
            ).fetchall()
            if not rows:
                return 0.0

            # Cache belief domains
            domain_cache = {}
            def get_domain(bid):
                if bid in domain_cache:
                    return domain_cache[bid]
                b = db.execute(
                    "SELECT source_type, topics FROM beliefs WHERE id = ?",
                    (bid,),
                ).fetchone()
                if not b:
                    domain_cache[bid] = "unknown"
                    return "unknown"
                st = b["source_type"] or ""
                if st in ("external", "meta", "identity", "synthesis"):
                    domain_cache[bid] = st
                else:
                    try:
                        topics = json.loads(b["topics"] or "[]")
                        domain_cache[bid] = str(topics[0]) if topics else "unclassified"
                    except (json.JSONDecodeError, TypeError):
                        domain_cache[bid] = "unclassified"
                return domain_cache[bid]

            count = sum(
                1 for r in rows if get_domain(r["belief_a"]) != get_domain(r["belief_b"])
            )
            return float(count)
        except Exception as e:
            logger.warning(f"Cross-domain bridge computation failed: {e}")
            return 0.0

    def _compute_cross_domain_edge_ratio(self):
        """Cross-domain edges / total edges. Key metric for F14 hypothesis."""
        try:
            db = self.semantic.db_conn
            total = db.execute(
                "SELECT COUNT(*) FROM belief_links WHERE COALESCE(active, 1) = 1"
            ).fetchone()[0]
            if total == 0:
                return 0.0
            cross = self._compute_cross_domain_bridges()
            return cross / total
        except Exception as e:
            logger.warning(f"Cross-domain edge ratio computation failed: {e}")
            return 0.0

    def _compute_avg_degree(self):
        """Average number of active links per belief node."""
        try:
            db = self.semantic.db_conn
            n_beliefs = db.execute(
                "SELECT COUNT(*) FROM beliefs "
                "WHERE COALESCE(deprecated, 0) = 0 AND COALESCE(is_dormant, 0) = 0"
            ).fetchone()[0]
            if n_beliefs == 0:
                return 0.0
            n_links = db.execute(
                "SELECT COUNT(*) FROM belief_links WHERE COALESCE(active, 1) = 1"
            ).fetchone()[0]
            # Each link contributes 1 degree to each endpoint → total degree = 2 * links
            return (2.0 * n_links) / n_beliefs
        except Exception as e:
            logger.warning(f"Avg degree computation failed: {e}")
            return 0.0

    def _compute_largest_component(self):
        """Size of the largest connected component via BFS on belief_links."""
        try:
            db = self.semantic.db_conn
            belief_ids = [r[0] for r in db.execute(
                "SELECT id FROM beliefs "
                "WHERE COALESCE(deprecated, 0) = 0 AND COALESCE(is_dormant, 0) = 0"
            ).fetchall()]
            if not belief_ids:
                return 0.0

            # Build adjacency list
            adj = {bid: set() for bid in belief_ids}
            links = db.execute(
                "SELECT belief_a, belief_b FROM belief_links "
                "WHERE COALESCE(active, 1) = 1"
            ).fetchall()
            for row in links:
                a, b = row[0], row[1]
                if a in adj and b in adj:
                    adj[a].add(b)
                    adj[b].add(a)

            # BFS for largest component
            visited = set()
            largest = 0
            for node in belief_ids:
                if node in visited:
                    continue
                queue = [node]
                visited.add(node)
                size = 0
                while queue:
                    current = queue.pop(0)
                    size += 1
                    for nb in adj[current]:
                        if nb not in visited:
                            visited.add(nb)
                            queue.append(nb)
                if size > largest:
                    largest = size

            return float(largest)
        except Exception as e:
            logger.warning(f"Largest component computation failed: {e}")
            return 0.0

    def _compute_lcr(self):
        """Largest component ratio: largest_component / total_beliefs.

        Phase transition watch: LCR > 0.5 means majority of beliefs are connected.
        """
        try:
            db = self.semantic.db_conn
            n = db.execute(
                "SELECT COUNT(*) FROM beliefs "
                "WHERE COALESCE(deprecated, 0) = 0 AND COALESCE(is_dormant, 0) = 0"
            ).fetchone()[0]
            if n == 0:
                return 0.0
            lc = self._compute_largest_component()
            return lc / n
        except Exception as e:
            logger.warning(f"LCR computation failed: {e}")
            return 0.0

    def _compute_bridge_ratio(self):
        """Bridge ratio: fraction of edges that are graph bridges.

        A bridge is an edge whose removal disconnects its component.
        High ratio = fragile graph (many single points of failure).
        Low ratio = resilient graph (redundant paths exist).
        Uses iterative Tarjan bridge-finding via DFS with explicit stack.
        """
        try:
            db = self.semantic.db_conn
            links = db.execute(
                "SELECT belief_a, belief_b FROM belief_links "
                "WHERE COALESCE(active, 1) = 1"
            ).fetchall()
            if not links:
                return 0.0

            adj: dict[str, set[str]] = {}
            for row in links:
                a, b = row[0], row[1]
                adj.setdefault(a, set()).add(b)
                adj.setdefault(b, set()).add(a)

            total_edges = len(links)
            bridges = 0
            disc = {}
            low = {}
            timer = [0]

            for start in list(adj):
                if start in disc:
                    continue
                # Iterative DFS: stack of (node, parent, neighbor_iterator)
                stack = [(start, None, iter(adj[start]))]
                disc[start] = low[start] = timer[0]
                timer[0] += 1
                while stack:
                    node, parent, neighbors = stack[-1]
                    try:
                        nb = next(neighbors)
                        if nb not in disc:
                            disc[nb] = low[nb] = timer[0]
                            timer[0] += 1
                            stack.append((nb, node, iter(adj[nb])))
                        elif nb != parent:
                            low[node] = min(low[node], disc[nb])
                    except StopIteration:
                        stack.pop()
                        if parent is not None:
                            low[parent] = min(low[parent], low[node])
                            if low[node] > disc[parent]:
                                bridges += 1

            return bridges / total_edges
        except Exception as e:
            logger.warning(f"Bridge ratio computation failed: {e}")
            return 0.0

    def _compute_isolated_ratio(self):
        """Isolated ratio: degree-0 beliefs / total beliefs.

        Phase transition watch: isolated_ratio < 0.25 means graph is connecting.
        """
        try:
            db = self.semantic.db_conn
            n = db.execute(
                "SELECT COUNT(*) FROM beliefs "
                "WHERE COALESCE(deprecated, 0) = 0 AND COALESCE(is_dormant, 0) = 0"
            ).fetchone()[0]
            if n == 0:
                return 0.0
            connected = db.execute(
                "SELECT COUNT(DISTINCT bid) FROM ("
                "  SELECT belief_a AS bid FROM belief_links WHERE COALESCE(active, 1) = 1"
                "  UNION"
                "  SELECT belief_b AS bid FROM belief_links WHERE COALESCE(active, 1) = 1"
                ")"
            ).fetchone()[0]
            return (n - connected) / n
        except Exception as e:
            logger.warning(f"Isolated ratio computation failed: {e}")
            return 0.0

    # ------------------------------------------------------------------
    # Read-only telemetry access (for ANIMA's self-awareness)
    # ------------------------------------------------------------------

    _STATE_KEYS = [
        ("coherence_confidence", "Coherence Confidence"),
        ("self_reference_rate", "Self-Reference Rate"),
        ("belief_revision_rate", "Belief Revision Rate"),
        ("domain_diversity", "Domain Diversity"),
        ("conceptual_clustering", "Conceptual Clustering"),
        ("belief_entropy", "Belief Entropy"),
        ("cross_domain_bridges", "Cross-Domain Bridges"),
        ("avg_degree", "Avg Degree"),
        ("largest_component", "Largest Component"),
        ("lcr", "LCR"),
        ("isolated_ratio", "Isolated Ratio"),
        ("bridge_ratio", "Bridge Ratio"),
    ]

    def get_current_state_summary(self):
        """Return the last logged state as a formatted string for prompt injection.

        Reads from telemetry.db (not computed live). Returns None if no state
        has been logged yet.
        """
        if not self.db_conn:
            return None
        try:
            row = self.db_conn.execute(
                "SELECT * FROM state_log ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            if not row:
                return None
            row = dict(row)
            lines = []
            for key, label in self._STATE_KEYS:
                val = row.get(key)
                if val is not None:
                    lines.append(f"  {label}: {val:.2f}")
            if not lines:
                return None
            return "\n".join(lines)
        except Exception as e:
            logger.debug(f"State summary read failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Exploration window telemetry
    # ------------------------------------------------------------------

    def log_exploration_window(self, episode_id=None, topics_attempted=0,
                                topics_completed=0, searches_used=0,
                                coherence_at_start=None, coherence_at_end=None,
                                domains=None, gap_driven=0, generated=0):
        """Record a session-level summary for an exploration window."""
        try:
            entry_id = str(uuid.uuid4())
            now = datetime.now(timezone.utc).isoformat()
            domains_json = json.dumps(domains) if domains else None
            self.db_conn.execute(
                "INSERT INTO exploration_window_log "
                "(id, timestamp, episode_id, topics_attempted, topics_completed, "
                "searches_used, coherence_at_start, coherence_at_end, domains, "
                "gap_driven, generated) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (entry_id, now, episode_id, topics_attempted, topics_completed,
                 searches_used, coherence_at_start, coherence_at_end, domains_json,
                 gap_driven, generated),
            )
            self.db_conn.commit()
            logger.info(
                f"Exploration window logged: {topics_completed}/{topics_attempted} topics, "
                f"{searches_used} searches, {gap_driven} gap-driven, {generated} generated"
            )
        except Exception as e:
            logger.warning(f"Exploration window logging failed: {e}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        if self.db_conn:
            self.db_conn.close()
            self.db_conn = None
