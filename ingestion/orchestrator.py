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

"""DMS Lifecycle Orchestrator — continuous knowledge metabolism loop.

Sequences: SCAN → INGEST → SLEEP(N) → CURIOSITY_MATCH → RE-READ → SLEEP(N)

This is an outer loop that calls core systems through their existing public
interfaces. It does NOT modify core engine behavior. If it crashes, ANIMA's
normal cycle is completely unaffected.

The orchestrator is the piece that turns indexed documents into a living
knowledge system — not a one-shot pipeline but a continuous cycle of
ingestion, dreaming, questioning, and re-reading.
"""

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

logger = logging.getLogger("ingestion.orchestrator")


class DMSOrchestrator:
    """Orchestrates the full DMS knowledge metabolism loop."""

    def __init__(self, evolution_engine, embedding_engine, curiosity_engine, config):
        """Initialize with references to existing engines.

        All engines are used read-only through their public APIs.
        The orchestrator never modifies engine internals.
        """
        self.engine = evolution_engine
        self.embeddings = embedding_engine
        self.curiosity = curiosity_engine
        self.config = config
        self.db = evolution_engine.semantic.db_conn

        # Lazy imports — only created when needed
        self._pipeline = None
        self._matcher = None
        self._extractor = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            from ingestion.pipeline import DocumentPipeline
            self._pipeline = DocumentPipeline(self.engine)
        return self._pipeline

    @property
    def matcher(self):
        if self._matcher is None:
            from ingestion.corpus_matcher import CorpusMatcher
            self._matcher = CorpusMatcher(
                self.db, self.embeddings, self.config
            )
        return self._matcher

    @property
    def extractor(self):
        if self._extractor is None:
            from ingestion.extractor import DocumentExtractor
            self._extractor = DocumentExtractor(
                self.engine.inference, self.config
            )
        return self._extractor

    def run_cycle(self, scan=True, ingest=True, dream_cycles=None,
                  curiosity_match=True, reread=True, dream_cycles_post=None):
        """Run one full DMS lifecycle cycle.

        Args:
            scan: Whether to scan for new/changed files.
            ingest: Whether to ingest indexed documents.
            dream_cycles: Number of dream passes after ingestion (None = from config).
            curiosity_match: Whether to match gaps to corpus.
            reread: Whether to do targeted re-reads from matches.
            dream_cycles_post: Dream passes after re-reads (None = from config).

        Returns dict with stats from each phase.
        """
        extraction_cfg = self.config.get("extraction", {})
        if dream_cycles is None:
            dream_cycles = extraction_cfg.get("post_ingest_dream_cycles", 5)
        if dream_cycles_post is None:
            dream_cycles_post = extraction_cfg.get("post_reread_dream_cycles", 5)

        stats = {"phases": {}}
        logger.info("=== DMS Lifecycle Cycle Starting ===")

        # Phase 1: Scan
        if scan:
            logger.info("Phase 1: Scanning for new documents...")
            scan_stats = self.pipeline.run_scan()
            stats["phases"]["scan"] = scan_stats
            logger.info(f"Scan: {scan_stats}")

        # Phase 2: Ingest
        if ingest:
            logger.info("Phase 2: Ingesting indexed documents...")
            ingest_stats = self.pipeline.run_ingest()
            stats["phases"]["ingest"] = ingest_stats
            logger.info(f"Ingest: {ingest_stats}")

        # Phase 3: Dream (post-ingest)
        if dream_cycles > 0:
            logger.info(f"Phase 3: Running {dream_cycles} dream passes...")
            completed = self.pipeline.run_post_ingest_dreams(dream_cycles)
            stats["phases"]["dreams_post_ingest"] = completed

        # Phase 4: Curiosity-to-corpus matching
        if curiosity_match and self.curiosity:
            logger.info("Phase 4: Matching curiosity gaps to corpus...")
            match_stats = self._run_curiosity_match()
            stats["phases"]["curiosity_match"] = match_stats

            # Phase 5: Targeted re-reads
            if reread and match_stats.get("matches"):
                logger.info("Phase 5: Targeted re-reads from matches...")
                reread_stats = self._run_targeted_rereads(match_stats["matches"])
                stats["phases"]["reread"] = reread_stats

                # Phase 6: Dream (post-reread)
                if dream_cycles_post > 0 and reread_stats.get("beliefs_total", 0) > 0:
                    logger.info(
                        f"Phase 6: Running {dream_cycles_post} dream passes (post-reread)..."
                    )
                    completed = self.pipeline.run_post_ingest_dreams(dream_cycles_post)
                    stats["phases"]["dreams_post_reread"] = completed

        logger.info(f"=== DMS Lifecycle Cycle Complete === Stats: {stats}")
        return stats

    def _run_curiosity_match(self):
        """Match open curiosity gaps against the document corpus.

        Skips gaps marked as low-yield (negative feedback).
        """
        if not self.curiosity:
            return {"gaps": 0, "matches": []}

        # Get open gaps
        gaps = self.curiosity.get_open_questions(limit=20)
        if not gaps:
            logger.info("No open curiosity gaps to match.")
            return {"gaps": 0, "matches": []}

        # Filter out low-yield gaps (negative feedback)
        gap_dicts = []
        skipped_low_yield = 0
        for g in gaps:
            g_dict = dict(g)
            gap_id = g_dict.get("id", "")
            if self._is_low_yield(gap_id):
                skipped_low_yield += 1
                continue
            gap_dicts.append(g_dict)

        if not gap_dicts:
            logger.info(f"All {skipped_low_yield} gaps marked low-yield, none to match.")
            return {"gaps": 0, "skipped_low_yield": skipped_low_yield, "matches": []}

        matches = self.matcher.match_gaps_to_corpus(gap_dicts[:10])

        logger.info(
            f"Curiosity match: {len(gap_dicts)} active gaps "
            f"({skipped_low_yield} low-yield skipped), "
            f"{len(matches)} with corpus matches"
        )
        return {
            "gaps": len(gap_dicts),
            "skipped_low_yield": skipped_low_yield,
            "matched": len(matches),
            "matches": matches,
        }

    def _run_targeted_rereads(self, matches):
        """Run targeted extraction on corpus-matched document sections.

        Tracks per-gap and per-document effectiveness metrics.
        Applies negative feedback: marks gaps as low-yield after repeated failures.
        """
        auto_accept = self.config.get("extraction", {}).get(
            "auto_accept_extraction", False
        )
        low_yield_threshold = self.config.get("extraction", {}).get(
            "gap_low_yield_threshold", 0.1
        )
        low_yield_min_matches = self.config.get("extraction", {}).get(
            "gap_low_yield_min_matches", 3
        )

        stats = {
            "processed": 0, "beliefs_total": 0, "beliefs_novel": 0,
            "beliefs_duplicate": 0, "gaps_addressed": 0,
        }

        # Flatten all (gap, doc_match) pairs for parallel dispatch.
        # Investigation gaps have a match cap (max 10) to prevent broad
        # gaps from dominating re-reads.
        INVESTIGATION_MATCH_CAP = 10
        work_items = []
        for match_entry in matches:
            gap = match_entry["gap"]
            gap_id = gap.get("id", "")

            # Check if investigation gap and over cap
            if gap_id.startswith("coverage_") or not gap_id:
                pass  # Coverage gaps have no cap
            else:
                try:
                    existing_rereads = self.db.execute(
                        "SELECT COUNT(*) as cnt FROM dms_reread_log WHERE gap_id=?",
                        (gap_id,),
                    ).fetchone()["cnt"]
                    if existing_rereads >= INVESTIGATION_MATCH_CAP:
                        continue  # Skip saturated gaps
                except Exception:
                    pass

            for doc_match in match_entry.get("matches", []):
                work_items.append((gap, doc_match))

        # Determine parallel worker count from available extraction models
        worker_count = self.pipeline._get_extraction_worker_count()
        logger.info(
            f"Re-read dispatch: {len(work_items)} sections across "
            f"{len(matches)} gaps, {worker_count} parallel workers"
        )

        def _extract_one(gap, doc_match):
            """Extract beliefs from one gap×doc section (runs in thread pool)."""
            gap_question = gap.get("question", "")
            existing_beliefs = self._get_related_beliefs(gap_question)
            beliefs = self.extractor.extract_targeted(
                section=doc_match["section"],
                filename=doc_match["filename"],
                sha256=doc_match["document_sha"],
                gap_question=gap_question,
                gap_context=gap.get("context", ""),
                existing_beliefs=existing_beliefs,
            )
            return gap, doc_match, beliefs, existing_beliefs

        # Dispatch — parallel if multiple workers, sequential otherwise
        results = []
        if worker_count > 1 and len(work_items) > 1:
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                futures = {
                    pool.submit(_extract_one, g, dm): (g, dm)
                    for g, dm in work_items
                }
                for future in as_completed(futures):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        logger.error(f"Parallel re-read extraction failed: {e}")
        else:
            for g, dm in work_items:
                try:
                    results.append(_extract_one(g, dm))
                except Exception as e:
                    logger.error(f"Re-read extraction failed: {e}")

        # Process results sequentially (governance + DB writes are not thread-safe)
        gap_accepted = {}  # gap_id -> total accepted
        for gap, doc_match, beliefs, existing_beliefs in results:
            gap_id = gap.get("id", "")
            gap_question = gap.get("question", "")
            existing_set = set(b.lower().strip() for b in existing_beliefs)

            extracted_count = len(beliefs)
            novel_count = 0
            dup_count = 0

            if beliefs:
                for b in beliefs:
                    stmt = b.get("statement", "").lower().strip()
                    if any(self._text_similarity(stmt, e) > 0.85 for e in existing_set):
                        dup_count += 1
                    else:
                        novel_count += 1

                accepted = self.pipeline._route_beliefs(
                    beliefs, doc_match["document_sha"], auto_accept
                )
                gap_accepted[gap_id] = gap_accepted.get(gap_id, 0) + accepted
                stats["beliefs_total"] += accepted
                stats["beliefs_novel"] += novel_count
                stats["beliefs_duplicate"] += dup_count
                stats["processed"] += 1

                if accepted > 0:
                    try:
                        self.db.execute(
                            "UPDATE document_ledger "
                            "SET belief_count = COALESCE(belief_count, 0) + ? "
                            "WHERE sha256 = ?",
                            (accepted, doc_match["document_sha"]),
                        )
                        self.db.commit()
                    except Exception:
                        pass

            self._log_reread(
                gap_id, doc_match["document_sha"],
                extracted_count, accepted if beliefs else 0,
                dup_count, novel_count, existing_beliefs,
            )

        # Update gap effectiveness (aggregate per gap)
        seen_gaps = set()
        for match_entry in matches:
            gap = match_entry["gap"]
            gap_id = gap.get("id", "")
            if gap_id in seen_gaps:
                continue
            seen_gaps.add(gap_id)

            if gap_id:
                self._update_gap_effectiveness(
                    gap_id, len(match_entry.get("matches", [])),
                    gap_accepted.get(gap_id, 0),
                )
                self._check_low_yield(
                    gap_id, low_yield_threshold, low_yield_min_matches
                )

            if gap_accepted.get(gap_id, 0) > 0:
                stats["gaps_addressed"] += 1

        accept_rate = (
            stats["beliefs_total"] / max(stats["processed"], 1)
            if stats["processed"] > 0 else 0
        )
        stats["accept_rate"] = round(accept_rate, 2)

        logger.info(
            f"Targeted re-reads: {stats['processed']} sections, "
            f"{stats['beliefs_total']} accepted ({stats['beliefs_novel']} novel, "
            f"{stats['beliefs_duplicate']} duplicate), "
            f"{stats['gaps_addressed']} gaps addressed, "
            f"accept_rate={stats['accept_rate']}"
        )
        return stats

    def _log_reread(self, gap_id, document_sha, extracted, accepted,
                    duplicates, novel, existing_beliefs):
        """Record re-read effectiveness to dms_reread_log."""
        try:
            # Compute avg similarity to existing beliefs
            avg_sim = 0.0
            if existing_beliefs and extracted > 0:
                # Rough proxy — actual similarity computed during novelty check
                avg_sim = duplicates / max(extracted, 1)

            now = datetime.now(timezone.utc).isoformat()
            self.db.execute(
                """INSERT INTO dms_reread_log
                   (gap_id, document_sha, beliefs_extracted, beliefs_accepted,
                    beliefs_duplicate, beliefs_novel, avg_similarity_to_existing,
                    created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (gap_id, document_sha, extracted, accepted,
                 duplicates, novel, round(avg_sim, 3), now),
            )
            self.db.commit()
        except Exception as e:
            logger.warning(f"Failed to log re-read: {e}")

    def _update_gap_effectiveness(self, gap_id, docs_matched, beliefs_accepted):
        """Update cumulative gap effectiveness counters."""
        try:
            now = datetime.now(timezone.utc).isoformat()
            existing = self.db.execute(
                "SELECT * FROM dms_gap_effectiveness WHERE gap_id = ?",
                (gap_id,),
            ).fetchone()

            if existing:
                self.db.execute(
                    """UPDATE dms_gap_effectiveness SET
                       times_matched = times_matched + 1,
                       total_beliefs_generated = total_beliefs_generated + ?,
                       total_accepted = total_accepted + ?,
                       last_matched_at = ?
                       WHERE gap_id = ?""",
                    (docs_matched, beliefs_accepted, now, gap_id),
                )
            else:
                self.db.execute(
                    """INSERT INTO dms_gap_effectiveness
                       (gap_id, times_matched, total_beliefs_generated,
                        total_accepted, last_matched_at, marked_low_yield)
                       VALUES (?, 1, ?, ?, ?, 0)""",
                    (gap_id, docs_matched, beliefs_accepted, now),
                )
            self.db.commit()
        except Exception as e:
            logger.warning(f"Failed to update gap effectiveness: {e}")

    def _check_low_yield(self, gap_id, threshold, min_matches):
        """Mark gap as low-yield if it's been matched enough times with poor results.

        Rule: if times_matched >= min_matches AND accept_rate < threshold → low-yield.
        """
        try:
            row = self.db.execute(
                "SELECT times_matched, total_accepted, total_beliefs_generated "
                "FROM dms_gap_effectiveness WHERE gap_id = ?",
                (gap_id,),
            ).fetchone()

            if not row or row["times_matched"] < min_matches:
                return

            accept_rate = row["total_accepted"] / max(row["total_beliefs_generated"], 1)
            if accept_rate < threshold:
                self.db.execute(
                    "UPDATE dms_gap_effectiveness SET marked_low_yield = 1 WHERE gap_id = ?",
                    (gap_id,),
                )
                self.db.commit()
                logger.info(
                    f"Gap {gap_id[:12]} marked low-yield: "
                    f"{row['times_matched']} matches, accept_rate={accept_rate:.2f}"
                )
        except Exception as e:
            logger.warning(f"Failed to check low-yield: {e}")

    def _is_low_yield(self, gap_id):
        """Check if a gap has been marked as low-yield."""
        if not gap_id:
            return False
        try:
            row = self.db.execute(
                "SELECT marked_low_yield FROM dms_gap_effectiveness WHERE gap_id = ?",
                (gap_id,),
            ).fetchone()
            return row and row["marked_low_yield"] == 1
        except Exception:
            return False

    @staticmethod
    def _text_similarity(a, b):
        """Quick word-overlap similarity for novelty check. Not embedding-based."""
        words_a = set(a.split())
        words_b = set(b.split())
        if not words_a or not words_b:
            return 0.0
        overlap = len(words_a & words_b)
        return overlap / max(len(words_a), len(words_b))

    def _get_related_beliefs(self, gap_question, top_k=5):
        """Get belief statements related to a gap for dedup context."""
        try:
            results = self.engine.semantic.search_beliefs(
                topics=[gap_question], limit=top_k
            )
            return [r["statement"] for r in results if r.get("statement")]
        except Exception:
            return []

    def resolve_gaps(self):
        """Check if any open gaps have been answered by recent beliefs.

        For each open gap, find beliefs with high similarity to the gap text.
        If a strong match exists, mark the gap as answered.

        Returns dict with resolution stats.
        """
        if not self.curiosity:
            return {"checked": 0, "resolved": 0}

        gaps = self.curiosity.get_open_questions(limit=50)
        if not gaps:
            return {"checked": 0, "resolved": 0}

        stats = {"checked": 0, "resolved": 0, "resolved_gaps": []}
        threshold = self.config.get("extraction", {}).get(
            "gap_resolution_threshold", 0.70
        )

        for gap in gaps:
            gap_dict = dict(gap)
            gap_text = gap_dict.get("question", "")
            gap_id = gap_dict.get("id", "")
            if not gap_text or not gap_id:
                continue

            stats["checked"] += 1

            # Embed gap and find similar beliefs
            try:
                gap_emb = self.embeddings.embed(gap_text)
                results = self.engine.semantic.search_beliefs(
                    topics=[gap_text], limit=5
                )

                for belief in results:
                    belief_emb = self.embeddings.embed(belief.get("statement", ""))
                    import numpy as np
                    sim = float(np.dot(gap_emb, belief_emb) / (
                        np.linalg.norm(gap_emb) * np.linalg.norm(belief_emb) + 1e-10
                    ))

                    if sim >= threshold:
                        # Gap answered — mark resolved
                        self.curiosity.resolve_question(
                            gap_id,
                            answer=f"Addressed by belief: {belief['statement'][:100]}",
                            answered_by="dms_orchestrator",
                        )
                        stats["resolved"] += 1
                        stats["resolved_gaps"].append({
                            "gap_id": gap_id,
                            "gap_text": gap_text[:80],
                            "resolved_by": belief["statement"][:80],
                            "similarity": round(sim, 3),
                        })
                        logger.info(
                            f"Gap resolved: '{gap_text[:60]}' → "
                            f"'{belief['statement'][:60]}' (sim={sim:.3f})"
                        )
                        break

            except Exception as e:
                logger.warning(f"Gap resolution check failed for {gap_id[:12]}: {e}")

        logger.info(
            f"Gap resolution: {stats['checked']} checked, {stats['resolved']} resolved"
        )
        return stats

    def run_autonomous(self, chat_lock, interval_minutes=None, max_cycles=5):
        """Run the DMS lifecycle loop autonomously in a background thread.

        Args:
            chat_lock: The _chat_lock from web_server — acquired during each phase.
            interval_minutes: Minutes between cycles (None = from config).
            max_cycles: Max cycles to run (default 5, None/0 = unlimited — explicit override).
        """
        extraction_cfg = self.config.get("extraction", {})
        if interval_minutes is None:
            interval_minutes = extraction_cfg.get("lifecycle_interval_minutes", 30)
        interval_sec = interval_minutes * 60

        self._autonomous_running = True
        self._autonomous_cycle_count = 0
        logger.info(
            f"DMS autonomous loop starting: interval={interval_minutes}m, "
            f"max_cycles={max_cycles or 'infinite'}"
        )

        while self._autonomous_running:
            if max_cycles and self._autonomous_cycle_count >= max_cycles:
                logger.info(f"Autonomous loop: max cycles ({max_cycles}) reached")
                break

            try:
                self._autonomous_cycle_count += 1
                logger.info(
                    f"=== Autonomous cycle {self._autonomous_cycle_count} starting ==="
                )

                with chat_lock:
                    stats = self.run_cycle()

                # Resolve gaps after each cycle
                with chat_lock:
                    resolve_stats = self.resolve_gaps()
                    stats["phases"]["gap_resolution"] = resolve_stats

                logger.info(
                    f"=== Autonomous cycle {self._autonomous_cycle_count} complete === "
                    f"Stats: {stats}"
                )

            except Exception as e:
                logger.error(f"Autonomous cycle {self._autonomous_cycle_count} failed: {e}")

            # Wait for next cycle
            for _ in range(interval_sec):
                if not self._autonomous_running:
                    break
                time.sleep(1)

        logger.info("DMS autonomous loop stopped")

    def stop_autonomous(self):
        """Signal the autonomous loop to stop."""
        self._autonomous_running = False

    def get_status(self):
        """Get orchestrator status — what's been done, what's pending."""
        ledger = self.pipeline.scanner.get_ledger_summary()
        belief_count = self.db.execute("SELECT COUNT(*) as cnt FROM beliefs").fetchone()["cnt"]
        edge_count = self.db.execute("SELECT COUNT(*) as cnt FROM belief_links").fetchone()["cnt"]

        gap_count = 0
        if self.curiosity:
            gaps = self.curiosity.get_open_questions(limit=100)
            gap_count = len(gaps) if gaps else 0

        entity_count = self.db.execute(
            "SELECT COUNT(*) as cnt FROM document_entities"
        ).fetchone()["cnt"]

        autonomous = {
            "running": getattr(self, "_autonomous_running", False),
            "cycles_completed": getattr(self, "_autonomous_cycle_count", 0),
        }

        return {
            "ledger": ledger,
            "beliefs": belief_count,
            "edges": edge_count,
            "entities": entity_count,
            "open_gaps": gap_count,
            "autonomous": autonomous,
        }
