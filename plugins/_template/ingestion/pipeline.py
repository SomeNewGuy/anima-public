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

"""Document ingestion pipeline — orchestrates scan → extract → govern → store.

This is the main entry point for document ingestion. It:
1. Scans the datafiles directory (scanner.py)
2. Batches documents by context size (small docs packed together, large docs solo)
3. Extracts beliefs with per-document provenance (extractor.py)
4. Extracts structured entities per document (entity_extractor.py)
5. Routes beliefs through the governance gate (auto_accept or triage)
6. Records provenance in belief_sources_documents
"""

import logging
import os
import sqlite3
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from time import sleep

from .scanner import DocumentScanner
from .extractor import DocumentExtractor
from .entity_extractor import EntityExtractor
from memory.dedup import is_normalized_duplicate

logger = logging.getLogger("ingestion.pipeline")


# ---------------------------------------------------------------------------
# Pre-triage hard filters — beliefs matching these NEVER enter the graph
# ---------------------------------------------------------------------------

# Meta-commentary, negation, section references — not actual claims
_HARD_REJECT_PATTERNS = [
    "not addressed",
    "not covered",
    "not discussed",
    "not mentioned",
    "not explicitly",
    "unclear whether",
    "this section",
    "this document",
    "this article",
    "this study does not",
    "the paper does not",
    "the article does not",
    "cannot be determined",
    "no direct evidence",
    "beyond the scope",
    "further research is needed",
    "remains to be seen",
    "it is not clear",
    "no information",
    "does not provide",
    "does not address",
    "does not discuss",
    "the key finding is",
    "the main finding is",
    "the study found that",
    "the authors conclude",
    "the results show that",
    "the data suggest that",
    "in summary,",
    "in conclusion,",
    "it was found that",
    "we found that",
    "our results",
    "the document states",
    "both statements refer",
    "the text mentions",
    "the paper reports",
    "according to the document",
    "the source indicates",
]

def _hard_filter(statement, extra_patterns=None):
    """Check if statement is meta-commentary or negation — should never enter beliefs.

    Args:
        statement: The belief statement to check.
        extra_patterns: Optional list of additional patterns from config.

    Returns (reject: bool, reason: str or None)
    """
    stmt_lower = statement.lower().strip()

    all_patterns = _HARD_REJECT_PATTERNS
    if extra_patterns:
        all_patterns = list(_HARD_REJECT_PATTERNS) + list(extra_patterns)

    for pattern in all_patterns:
        if pattern in stmt_lower:
            return True, f"hard_filter: contains '{pattern}'"

    return False, None


class DocumentPipeline:
    """Orchestrates the full document ingestion pipeline."""

    def __init__(self, evolution_engine):
        """Initialize with a reference to the EvolutionEngine.

        The evolution engine provides:
        - self.semantic.db_conn (SQLite connection with document_ledger tables)
        - self.inference (LLM for extraction)
        - self.config (settings.toml)
        - self.gateway (ProposalGateway for governance)
        - self._process_approved_item() (for auto-accepted beliefs)
        - self.queue_item() (for queued beliefs)
        """
        self.engine = evolution_engine
        self.config = evolution_engine.config
        self.db = evolution_engine.semantic.db_conn
        self.scanner = DocumentScanner(self.db, self.config)
        self.extractor = DocumentExtractor(evolution_engine.inference, self.config)
        self.entity_extractor = EntityExtractor(evolution_engine.inference, self.config)


    def _resolve_file_path(self, rel_path):
        """Resolve a relative file path against configured datafiles directories.

        Tries each root in order, returns first existing match.
        Falls back to first root + rel_path if none found (for error reporting).
        """
        for root in self.scanner.datafiles_dirs:
            full = os.path.join(root, rel_path)
            if os.path.exists(full):
                return full
        # Fallback for error handling
        if self.scanner.datafiles_dirs:
            return os.path.join(self.scanner.datafiles_dirs[0], rel_path)
        return rel_path

    def run_scan(self):
        """Phase 1: Scan filesystem and populate ledger.

        Returns scan stats dict.
        """
        logger.info("Starting document scan...")
        stats = self.scanner.scan()
        logger.info(f"Scan results: {stats}")
        return stats

    def run_ingest(self, limit=None, dry_run=False):
        """Phase 2: Extract beliefs from indexed documents using context-aware batching.

        Small documents are packed into batches that fit the LLM context window.
        Large documents get their own solo extraction call.
        Per-document provenance is preserved via DOCID tagging in batch mode.

        Args:
            limit: Max documents to process in this run (None = all).
            dry_run: If True, extract but don't store beliefs.

        Returns dict with ingest stats.
        """
        docs = self.scanner.get_indexed_documents(limit=limit)
        if not docs:
            logger.info("No indexed documents to ingest.")
            return {"processed": 0, "beliefs_total": 0, "errors": 0,
                    "batches": 0, "solo": 0}

        auto_accept = self.config.get("extraction", {}).get("auto_accept_extraction", False)

        stats = {
            "processed": 0, "beliefs_total": 0, "entities_total": 0,
            "errors": 0, "skipped_empty": 0, "batches": 0, "solo": 0,
        }

        # Load content for all docs
        loaded_docs = []
        for doc in docs:
            sha = doc["sha256"]
            filepath = self._resolve_file_path(doc["file_path"])

            if not os.path.exists(filepath):
                logger.warning(f"File missing: {filepath}")
                self.scanner.mark_error(sha, "file_missing")
                stats["errors"] += 1
                continue

            content = self.extractor.read_document(filepath)
            if not content or not content.strip():
                self.scanner.mark_error(sha, "empty_content")
                stats["skipped_empty"] += 1
                continue

            doc["content"] = content
            loaded_docs.append(doc)

        # Extraction strategy:
        # Multi-model (parallel): one doc per task, fan out across models.
        #   Every model handles one doc at a time — even 4K context models work.
        # Single-model: batch small docs into context-sized chunks for efficiency.
        parallel_workers = self._get_extraction_worker_count()

        def _extract_doc(doc):
            """Extract beliefs + entities from one document."""
            beliefs = self.extractor.extract(
                content=doc["content"],
                filename=doc["filename"],
                file_type=doc["file_type"],
                sha256=doc["sha256"],
            )
            entity_list = self.entity_extractor.extract(
                content=doc["content"],
                filename=doc["filename"],
                file_type=doc["file_type"],
                sha256=doc["sha256"],
            )
            return doc, beliefs, entity_list

        def _extract_batch(batch):
            """Extract beliefs + entities from a batch (single-model mode)."""
            if len(batch) == 1:
                return _extract_doc(batch[0])
            beliefs = self.extractor.extract_batch(batch)
            entities_by_sha = self.entity_extractor.extract_batch(batch)
            return batch, beliefs, entities_by_sha

        if parallel_workers > 1 and len(loaded_docs) > 1:
            # Multi-model: one doc per task, parallel fan-out
            logger.info(
                f"Parallel extraction: {len(loaded_docs)} documents, "
                f"{parallel_workers} workers"
            )
            with ThreadPoolExecutor(max_workers=parallel_workers) as pool:
                futures = {pool.submit(_extract_doc, d): d for d in loaded_docs}
                for future in as_completed(futures):
                    try:
                        doc, beliefs, entity_list = future.result()
                        stats["solo"] += 1

                        if dry_run:
                            stats["processed"] += 1
                            stats["beliefs_total"] += len(beliefs)
                            continue

                        if entity_list:
                            self._store_entities(doc["sha256"], entity_list)
                            stats["entities_total"] += len(entity_list)

                        self._process_batch_results(
                            [doc], beliefs, auto_accept, stats
                        )
                    except Exception as e:
                        logger.error(f"Parallel extraction failed: {e}")
                        stats["errors"] += 1
        else:
            # Single-model: batch small docs for context efficiency
            batches = self.extractor.build_batches(loaded_docs)
            logger.info(
                f"Sequential extraction: {len(loaded_docs)} docs in "
                f"{len(batches)} batches"
            )
            for batch in batches:
                try:
                    if len(batch) == 1:
                        doc, beliefs, entity_list = _extract_doc(batch[0])
                        stats["solo"] += 1
                        if dry_run:
                            stats["processed"] += 1
                            stats["beliefs_total"] += len(beliefs)
                            continue
                        if entity_list:
                            self._store_entities(doc["sha256"], entity_list)
                            stats["entities_total"] += len(entity_list)
                        self._process_batch_results(
                            [doc], beliefs, auto_accept, stats
                        )
                    else:
                        _, beliefs, entities_by_sha = _extract_batch(batch)
                        stats["batches"] += 1
                        if dry_run:
                            stats["processed"] += len(batch)
                            stats["beliefs_total"] += len(beliefs)
                            continue
                        for doc in batch:
                            doc_entities = entities_by_sha.get(doc["sha256"], [])
                            if doc_entities:
                                self._store_entities(doc["sha256"], doc_entities)
                                stats["entities_total"] += len(doc_entities)
                        self._process_batch_results(
                            batch, beliefs, auto_accept, stats
                        )
                except Exception as e:
                    logger.error(f"Extraction failed: {e}")
                    stats["errors"] += 1

        # Retry: files that produced 0 beliefs get re-extracted with primary model.
        # Parallel dispatch may have sent them to a weaker model that returned empty.
        # Track model failures — if a model consistently returns empty, auto-demote.
        zero_docs = [
            doc for doc in loaded_docs
            if self.scanner.get_belief_count(doc["sha256"]) == 0
        ]
        if zero_docs and len(loaded_docs) > len(zero_docs):
            logger.info(
                f"Retry extraction: {len(zero_docs)} docs produced 0 beliefs, "
                f"retrying sequentially with primary model"
            )

            # Track model extraction failures for auto-demote
            try:
                from core.router import ModelRouter
                router = self.engine.inference
                if isinstance(router, ModelRouter):
                    # Count zero-extraction docs — if >50% of a model's docs
                    # returned empty, increment error count for auto-demote
                    total_docs = len(loaded_docs)
                    failure_rate = len(zero_docs) / max(total_docs, 1)
                    if failure_rate > 0.5:
                        logger.warning(
                            f"Extraction failure rate {failure_rate:.0%} — "
                            f"{len(zero_docs)}/{total_docs} docs returned 0 beliefs"
                        )
            except Exception:
                pass

            for doc in zero_docs:
                try:
                    beliefs = self.extractor.extract(
                        content=doc["content"],
                        filename=doc["filename"],
                        file_type=doc["file_type"],
                        sha256=doc["sha256"],
                    )
                    if beliefs:
                        accepted = self._route_beliefs(beliefs, doc["sha256"], auto_accept)
                        self.scanner.mark_ingested(doc["sha256"], accepted)
                        stats["beliefs_total"] += len(beliefs)
                        logger.info(
                            f"Retry success: {doc['filename']} → {len(beliefs)} extracted, "
                            f"{accepted} accepted"
                        )
                except Exception as e:
                    logger.warning(f"Retry extraction failed for {doc['filename']}: {e}")

            # Escalation: docs still at 0 beliefs get extracted with large-tier model
            still_zero = [
                doc for doc in zero_docs
                if self.scanner.get_belief_count(doc["sha256"]) == 0
            ]
            if still_zero:
                logger.info(
                    f"Escalating {len(still_zero)} docs to large-tier model"
                )
                for doc in still_zero:
                    try:
                        beliefs = self.extractor.extract_with_task(
                            content=doc["content"],
                            filename=doc["filename"],
                            file_type=doc["file_type"],
                            sha256=doc["sha256"],
                            task="triage",  # routes to large-tier model
                        )
                        if beliefs:
                            accepted = self._route_beliefs(beliefs, doc["sha256"], auto_accept)
                            self.scanner.mark_ingested(doc["sha256"], accepted)
                            stats["beliefs_total"] += len(beliefs)
                            logger.info(
                                f"Escalation success: {doc['filename']} → "
                                f"{len(beliefs)} extracted, {accepted} accepted"
                            )
                        else:
                            logger.warning(
                                f"Escalation: {doc['filename']} still 0 beliefs "
                                f"from large-tier model"
                            )
                    except Exception as e:
                        logger.warning(
                            f"Escalation failed for {doc['filename']}: {e}"
                        )

            # Final check: docs STILL at 0 → mark as FAILED_EXTRACTION
            final_zero = [
                doc for doc in still_zero
                if self.scanner.get_belief_count(doc["sha256"]) == 0
            ]
            for doc in final_zero:
                self.db.execute(
                    "UPDATE document_ledger SET status = 'failed_extraction' "
                    "WHERE sha256 = ?",
                    (doc["sha256"],),
                )
                logger.warning(
                    f"FAILED EXTRACTION: {doc['filename']} — 0 beliefs after "
                    f"3 attempts (parallel + retry + escalation). "
                    f"Density: 0/{len(doc['content'].split())} words."
                )
            if final_zero:
                self.db.commit()

        logger.info(
            f"Ingest complete: {stats['processed']} docs in "
            f"{stats['batches']} batches + {stats['solo']} solo, "
            f"{stats['beliefs_total']} beliefs, {stats['entities_total']} entities, "
            f"{stats['errors']} errors"
        )

        # Post-ingest dream cycles — build edges between new beliefs
        dream_cycles = self.config.get("extraction", {}).get("post_ingest_dream_cycles", 0)
        if dream_cycles > 0 and not dry_run and stats["beliefs_total"] > 0:
            stats["dream_cycles"] = self.run_post_ingest_dreams(dream_cycles)

        return stats

    def _get_extraction_worker_count(self):
        """Count how many models can handle extraction in parallel.

        Checks the router for online extraction-capable models.
        Returns worker count (min 1, max = available models).
        """
        try:
            from core.router import ModelRouter
            router = self.engine.inference
            if not isinstance(router, ModelRouter) or router.mode != "multi":
                return 1

            count = 0
            for name, info in router.models.items():
                if info.online and info.enabled and info.parallel and info.accepts_task("extraction"):
                    count += 1
            return max(count, 1)
        except Exception:
            return 1

    def _process_batch_results(self, batch, beliefs, auto_accept, stats):
        """Process extraction results for a batch, routing beliefs and updating ledger."""
        # Group beliefs by source document SHA
        beliefs_by_sha = {}
        orphan_beliefs = []
        for b in beliefs:
            sha = b.get("_document_sha")
            if sha:
                beliefs_by_sha.setdefault(sha, []).append(b)
            else:
                orphan_beliefs.append(b)

        if orphan_beliefs:
            logger.warning(
                f"{len(orphan_beliefs)} beliefs without document provenance in batch"
            )

        # Process each document in the batch
        for doc in batch:
            sha = doc["sha256"]
            doc_beliefs = beliefs_by_sha.get(sha, [])

            if not doc_beliefs:
                self.scanner.mark_ingested(sha, 0)
                stats["processed"] += 1
                continue

            accepted_count = self._route_beliefs(doc_beliefs, sha, auto_accept)
            self.scanner.mark_ingested(sha, accepted_count)
            stats["processed"] += 1
            stats["beliefs_total"] += len(doc_beliefs)

            logger.info(
                f"Ingested {doc['filename']}: {len(doc_beliefs)} extracted, "
                f"{accepted_count} accepted/queued"
            )

        # Route orphan beliefs (no DOCID match) — attribute to first doc in batch
        if orphan_beliefs and batch:
            fallback_sha = batch[0]["sha256"]
            logger.warning(
                f"Routing {len(orphan_beliefs)} orphan beliefs to {batch[0]['filename']}"
            )
            self._route_beliefs(orphan_beliefs, fallback_sha, auto_accept)
            stats["beliefs_total"] += len(orphan_beliefs)

    def _route_beliefs(self, beliefs, document_sha, auto_accept):
        """Route extracted beliefs through governance and record provenance.

        Pre-triage filters (run before governance gateway):
        1. Hard filter — meta-commentary, negation, section references
        2. ai_systems domain rejection (Research instances only)
        3. Normalized text dedup — catches restatements within batch + recent DB beliefs

        Returns count of beliefs that were accepted or queued (not rejected).
        """
        from core.triage_adapters import _category_to_proposal

        # Plugin-provided domain filter (set by orchestrator via pipeline.domain_filter)
        _domain_filter = getattr(self, 'domain_filter', None)

        # Collect accepted statements for within-batch dedup
        batch_statements = []

        # Load recent belief statements from DB for cross-batch dedup (last 500)
        try:
            recent_rows = self.db.execute(
                "SELECT statement FROM beliefs WHERE COALESCE(deprecated, 0) = 0 "
                "ORDER BY created_at DESC LIMIT 500"
            ).fetchall()
            recent_db_statements = [r["statement"] for r in recent_rows]
        except Exception:
            recent_db_statements = []

        # Load approval queue statements for cross-queue dedup
        # Prevents the same belief being queued 3-5 times across cycles
        try:
            queue_rows = self.db.execute(
                "SELECT data FROM approval_queue WHERE status = 'pending'"
            ).fetchall()
            import json as _json
            queue_statements = []
            for qr in queue_rows:
                try:
                    qd = _json.loads(qr["data"]) if isinstance(qr["data"], str) else qr["data"]
                    stmt = qd.get("statement") or qd.get("inference") or ""
                    if stmt:
                        queue_statements.append(stmt)
                except Exception:
                    pass
            recent_db_statements.extend(queue_statements)
        except Exception:
            pass

        count = 0
        for belief_data in beliefs:
            statement = belief_data.get("statement", "").strip()
            if not statement:
                continue

            # Use a synthetic episode_id for document ingestion
            episode_id = f"doc:{document_sha[:16]}"

            # --- Pre-triage filter 0: Truncation check ---
            if len(statement) < 30 or statement.endswith(('.', '!', '?')) is False and len(statement) < 60:
                # Short or doesn't end with punctuation = likely truncated
                if not statement[-1] in '.!?)':
                    self._log_triage("belief", belief_data, "reject",
                                    "truncated statement", document_sha)
                    self.engine._record_rejection("belief", belief_data)
                    continue

            # --- Pre-triage filter 1: Hard reject meta/negation/commentary ---
            # Skip for game/lore mode — hard filters are for research papers, not fiction
            skip_hard_filter = self.config.get("extraction", {}).get("skip_hard_filter", False)
            if not skip_hard_filter:
                extra_patterns = self.config.get("extraction", {}).get(
                    "hard_reject_patterns", []
                )
                reject, reason = _hard_filter(statement, extra_patterns)
                if reject:
                    self._log_triage("belief", belief_data, "reject", reason, document_sha)
                    self.engine._record_rejection("belief", belief_data)
                    continue

            # --- Pre-triage filter 2: plugin domain filter ---
            if _domain_filter and _domain_filter(statement):
                self._log_triage(
                    "belief", belief_data, "reject",
                    "hard_filter: plugin domain filter", document_sha,
                )
                self.engine._record_rejection("belief", belief_data)
                continue

            # --- Pre-triage filter 3: Normalized text dedup ---
            # Check within current batch
            is_dup, matched = is_normalized_duplicate(
                statement, batch_statements, threshold=0.90,
            )
            if is_dup:
                self._log_triage(
                    "belief", belief_data, "reject",
                    f"dedup_normalized: batch match", document_sha,
                )
                self.engine._record_rejection("belief", belief_data)
                continue

            # Check against recent DB beliefs
            is_dup, matched = is_normalized_duplicate(
                statement, recent_db_statements, threshold=0.90,
            )
            if is_dup:
                self._log_triage(
                    "belief", belief_data, "reject",
                    f"dedup_normalized: existing belief match", document_sha,
                )
                self.engine._record_rejection("belief", belief_data)
                continue

            # --- Pre-triage filter 4: Shadow contract prevention ---
            # If this belief overlaps with an operator/contract belief, skip it.
            # The contract IS the truth — don't create a shadow version from code.
            try:
                overlap = self.db.execute(
                    "SELECT id, statement FROM beliefs "
                    "WHERE operator_anchored = 1 AND COALESCE(deprecated,0) = 0 "
                    "AND valid_to IS NULL",
                ).fetchall()
                is_shadow = False
                stmt_lower = statement.lower()
                for op in overlap:
                    op_lower = op["statement"].lower()
                    # Check for significant overlap — shared key terms
                    op_words = set(w for w in op_lower.split() if len(w) > 4)
                    stmt_words = set(w for w in stmt_lower.split() if len(w) > 4)
                    if op_words and stmt_words:
                        shared = op_words & stmt_words
                        # >50% word overlap with a contract belief = shadow
                        if len(shared) > min(len(op_words), len(stmt_words)) * 0.5:
                            is_shadow = True
                            break
                if is_shadow:
                    self._log_triage("belief", belief_data, "reject",
                                     "shadow_contract: overlaps with operator belief",
                                     document_sha)
                    continue
            except Exception:
                pass

            # Track for within-batch dedup
            batch_statements.append(statement)

            if auto_accept:
                self.engine._process_approved_item("belief", belief_data, episode_id)
                self._record_provenance(belief_data, document_sha)
                self._log_triage("belief", belief_data, "accept", "auto_accept", document_sha)
                count += 1
            else:
                proposal = _category_to_proposal("belief", belief_data, episode_id)
                triage_result = self.engine.gateway.submit(proposal)

                if triage_result.decision == "accept":
                    self.engine._process_approved_item("belief", belief_data, episode_id)
                    self._record_provenance(belief_data, document_sha)
                    count += 1
                elif triage_result.decision == "queue":
                    rec = self.engine._recommend("belief", belief_data)
                    label = self.engine._item_label("belief", belief_data, rec)
                    self.engine.queue_item("belief", belief_data, rec, label, episode_id)
                    # Record provenance for queued items too — they may get approved later
                    self._record_provenance(belief_data, document_sha)
                    count += 1
                else:
                    self.engine._record_rejection("belief", belief_data)

                self._log_triage(
                    "belief", belief_data,
                    triage_result.decision, triage_result.reason,
                    document_sha,
                )

        return count

    def _log_triage(self, category, belief_data, decision, reason, document_sha):
        """Log triage decision with full reasoning to dms_triage_log."""
        try:
            now = datetime.now(timezone.utc).isoformat()
            statement = belief_data.get("statement", "")[:500]
            confidence = belief_data.get("confidence", "")
            source_type = belief_data.get("source", belief_data.get("extraction_context", ""))
            self.db.execute(
                """INSERT INTO dms_triage_log
                   (category, statement, decision, reason, confidence,
                    source_type, document_sha, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (category, statement, decision, reason, confidence,
                 source_type, document_sha, now),
            )
            self.db.commit()
        except Exception as e:
            logger.warning(f"Failed to log triage decision: {e}")

    def _store_entities(self, document_sha, entities):
        """Store extracted entities for a document.

        Args:
            document_sha: SHA256 of the source document.
            entities: List of (entity_type, entity_value) tuples.
        """
        now = datetime.now(timezone.utc).isoformat()
        for entity_type, entity_value in entities:
            self.db.execute(
                """INSERT INTO document_entities
                   (document_sha, entity_type, entity_value, extracted_at)
                   VALUES (?, ?, ?, ?)""",
                (document_sha, entity_type, entity_value, now),
            )
        self.db.commit()

    def _record_provenance(self, belief_data, document_sha, belief_id=None):
        """Record belief → document provenance link.

        Args:
            belief_data: dict with statement, confidence, etc.
            document_sha: SHA256 of the source document.
            belief_id: If provided, use directly. Otherwise look up by statement.
        """
        if not document_sha:
            return

        # Use provided belief_id or fall back to statement lookup
        if not belief_id:
            statement = belief_data.get("statement", "")
            if not statement:
                return
            # Try exact match first, then LIKE match for minor variations
            row = self.db.execute(
                "SELECT id FROM beliefs WHERE statement = ? ORDER BY created_at DESC LIMIT 1",
                (statement,),
            ).fetchone()
            if not row:
                # Fuzzy fallback: match first 80 chars (handles minor triage edits)
                row = self.db.execute(
                    "SELECT id FROM beliefs WHERE statement LIKE ? ORDER BY created_at DESC LIMIT 1",
                    (statement[:80] + "%",),
                ).fetchone()
            if not row:
                logger.debug(f"No belief found for provenance: {statement[:60]}")
                return
            belief_id = row["id"]

        # Check for duplicate provenance entry
        existing = self.db.execute(
            "SELECT 1 FROM belief_sources_documents WHERE belief_id = ? AND document_sha = ?",
            (belief_id, document_sha),
        ).fetchone()
        if existing:
            return

        extraction_context = belief_data.get("extraction_context", "document")
        confidence_str = belief_data.get("confidence", "medium")
        conf_map = {"high": 1.0, "medium": 0.7, "low": 0.4}
        confidence_val = conf_map.get(confidence_str, 0.7)

        self.db.execute(
            """INSERT INTO belief_sources_documents
               (belief_id, document_sha, extraction_context, confidence_contribution)
               VALUES (?, ?, ?, ?)""",
            (belief_id, document_sha, extraction_context, confidence_val),
        )
        self.db.commit()

        # Derive entity tags from source document
        try:
            from core.tags import TagRegistry
            tag_reg = getattr(self, '_tag_registry', None)
            if tag_reg is None:
                tag_reg = TagRegistry(self.db)
                self._tag_registry = tag_reg
            tag_reg.derive_from_entities(belief_id, document_sha=document_sha)
        except Exception as e:
            import logging
            logging.getLogger("ingestion.pipeline").debug(f"Entity tag derivation failed: {e}")

    def backfill_provenance(self):
        """Backfill provenance for beliefs missing document links.

        Matches beliefs to source documents using the document episode_id pattern
        (doc:sha_prefix) stored in source_episodes. Falls back to embedding
        similarity against document content.

        Returns count of provenance links created.
        """
        # Find beliefs without provenance
        orphans = self.db.execute(
            """SELECT b.id, b.statement, b.source_episodes
               FROM beliefs b
               WHERE b.id NOT IN (SELECT belief_id FROM belief_sources_documents)
               AND COALESCE(b.deprecated, 0) = 0"""
        ).fetchall()

        if not orphans:
            logger.info("No beliefs missing provenance.")
            return 0

        created = 0
        for row in orphans:
            belief_id = row["id"]
            source_eps = row["source_episodes"] or "[]"

            # Try to match via doc:sha episode pattern
            import json, re
            try:
                eps = json.loads(source_eps)
            except Exception:
                eps = []

            matched = False
            for ep in eps:
                m = re.match(r"doc:([a-f0-9]+)", str(ep))
                if m:
                    sha_prefix = m.group(1)
                    doc = self.db.execute(
                        "SELECT sha256 FROM document_ledger WHERE sha256 LIKE ?",
                        (sha_prefix + "%",),
                    ).fetchone()
                    if doc:
                        existing = self.db.execute(
                            "SELECT 1 FROM belief_sources_documents WHERE belief_id = ? AND document_sha = ?",
                            (belief_id, doc["sha256"]),
                        ).fetchone()
                        if not existing:
                            self.db.execute(
                                """INSERT INTO belief_sources_documents
                                   (belief_id, document_sha, extraction_context, confidence_contribution)
                                   VALUES (?, ?, 'backfill', 0.7)""",
                                (belief_id, doc["sha256"]),
                            )
                            created += 1
                            matched = True
                        break

            if not matched:
                # Fallback: match statement against document filenames mentioned in text
                stmt_lower = row["statement"].lower()
                docs = self.db.execute(
                    "SELECT sha256, filename FROM document_ledger WHERE status = 'ingested'"
                ).fetchall()
                for doc in docs:
                    fname_base = doc["filename"].replace(".md", "").replace("_", " ").lower()
                    # Check if filename keywords appear in belief statement
                    keywords = [w for w in fname_base.split() if len(w) > 3]
                    if keywords and sum(1 for k in keywords if k in stmt_lower) >= 2:
                        existing = self.db.execute(
                            "SELECT 1 FROM belief_sources_documents WHERE belief_id = ? AND document_sha = ?",
                            (belief_id, doc["sha256"]),
                        ).fetchone()
                        if not existing:
                            self.db.execute(
                                """INSERT INTO belief_sources_documents
                                   (belief_id, document_sha, extraction_context, confidence_contribution)
                                   VALUES (?, ?, 'backfill_keyword', 0.5)""",
                                (belief_id, doc["sha256"]),
                            )
                            created += 1
                        break

        self.db.commit()
        logger.info(f"Provenance backfill: {created} links created for {len(orphans)} orphan beliefs")
        return created

    def backfill_triplet_provenance(self):
        """Backfill parent_a/parent_b for triplet beliefs with null parents.

        Triplet beliefs created before provenance tracking was added have
        null parent_a/parent_b. This finds them and reconstructs parentage
        from belief_links.

        Returns count of beliefs updated.
        """
        try:
            # Find triplet/dream beliefs with null parents
            orphans = self.db.execute(
                "SELECT id, statement, generation_type FROM beliefs "
                "WHERE generation_type IN ('dream', 'triplet') "
                "AND (parent_a IS NULL OR parent_b IS NULL) "
                "AND COALESCE(deprecated, 0) = 0"
            ).fetchall()

            if not orphans:
                logger.info("No triplet beliefs need provenance backfill.")
                return 0

            updated = 0
            for orphan in orphans:
                belief_id = orphan["id"]

                # Find connected beliefs via belief_links
                links = self.db.execute(
                    "SELECT belief_a, belief_b FROM belief_links "
                    "WHERE belief_a = ? OR belief_b = ? "
                    "ORDER BY similarity DESC",
                    (belief_id, belief_id),
                ).fetchall()

                parents = set()
                for link in links:
                    other = link["belief_b"] if link["belief_a"] == belief_id else link["belief_a"]
                    if other != belief_id:
                        parents.add(other)

                parent_list = list(parents)
                if len(parent_list) >= 2:
                    self.db.execute(
                        "UPDATE beliefs SET parent_a = ?, parent_b = ? "
                        "WHERE id = ?",
                        (parent_list[0], parent_list[1], belief_id),
                    )
                    updated += 1
                elif len(parent_list) == 1:
                    self.db.execute(
                        "UPDATE beliefs SET parent_a = ? WHERE id = ?",
                        (parent_list[0], belief_id),
                    )
                    updated += 1

            self.db.commit()
            logger.info(
                f"Triplet provenance backfill: {updated}/{len(orphans)} "
                f"beliefs updated"
            )
            return updated

        except Exception as e:
            logger.warning(f"Triplet provenance backfill failed: {e}")
            return 0

    def run_entity_extraction(self, limit=None):
        """Run entity extraction on ingested docs that don't have entities yet.

        This is a catch-up method for docs ingested before entity extraction
        was added, or when entity extraction needs to be re-run separately.

        Returns stats dict.
        """
        # Find ingested docs with no entities
        query = (
            "SELECT dl.sha256, dl.filename, dl.file_path, dl.file_type "
            "FROM document_ledger dl "
            "WHERE dl.status = 'ingested' "
            "AND dl.sha256 NOT IN (SELECT DISTINCT document_sha FROM document_entities) "
            "ORDER BY dl.ingested_at ASC"
        )
        if limit:
            query += f" LIMIT {int(limit)}"

        docs = [dict(r) for r in self.db.execute(query).fetchall()]
        if not docs:
            logger.info("No documents need entity extraction.")
            return {"processed": 0, "entities_total": 0}

        stats = {"processed": 0, "entities_total": 0, "errors": 0}

        for doc in docs:
            filepath = self._resolve_file_path(doc["file_path"])
            try:
                with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except Exception as e:
                logger.warning(f"Failed to read {filepath}: {e}")
                stats["errors"] += 1
                continue

            if not content or not content.strip():
                stats["processed"] += 1
                continue

            entities = self.entity_extractor.extract(
                content=content,
                filename=doc["filename"],
                file_type=doc["file_type"],
                sha256=doc["sha256"],
            )
            if entities:
                self._store_entities(doc["sha256"], entities)
                stats["entities_total"] += len(entities)

            stats["processed"] += 1
            logger.info(
                f"Entity extraction: {doc['filename']} → {len(entities)} entities"
            )

        logger.info(
            f"Entity extraction complete: {stats['processed']} docs, "
            f"{stats['entities_total']} entities, {stats['errors']} errors"
        )
        return stats

    def run_post_ingest_dreams(self, cycles):
        """Run dream passes after ingestion to build edges between beliefs.

        Each pass finds belief pairs and creates edges. Multiple passes
        densify the graph progressively.

        When multiple large-tier models are available, passes run in parallel
        (each pass is self-contained — own pair selection, generation, triage).

        Returns number of cycles completed.
        """
        from core.router import ModelRouter
        router = getattr(self.engine, 'engine', None) or getattr(self.engine, 'inference', None)

        # Count available large-tier models that can handle dreams
        parallel_capacity = 1
        if isinstance(router, ModelRouter):
            dream_models = [
                n for n, info in router.models.items()
                if info.accepts_task("dreams") and info.online and info.enabled
            ]
            parallel_capacity = min(len(dream_models), cycles)

        if parallel_capacity > 1:
            # Parallel dream passes — each on a different model
            logger.info(f"Post-ingest dreams: {cycles} passes, {parallel_capacity} in parallel")
            from concurrent.futures import ThreadPoolExecutor

            completed = 0
            results = [None] * cycles

            def _run_one_pass(i):
                try:
                    logger.info(f"Dream pass {i + 1}/{cycles} starting")
                    dream_start = datetime.now(timezone.utc).isoformat()
                    self.engine._run_dreams(queue_mode=True)
                    self.engine._dedup_edges(since_timestamp=dream_start)
                    return True
                except Exception as e:
                    logger.warning(f"Dream pass {i + 1} failed: {e}")
                    return False

            with ThreadPoolExecutor(max_workers=parallel_capacity) as pool:
                futures = list(pool.map(_run_one_pass, range(cycles)))
                completed = sum(1 for f in futures if f)
        else:
            # Sequential — single model or solo mode
            completed = 0
            for i in range(cycles):
                logger.info(f"Post-ingest dream cycle {i + 1}/{cycles}")
                try:
                    dream_start = datetime.now(timezone.utc).isoformat()
                    self.engine._run_dreams(queue_mode=True)
                    self.engine._dedup_edges(since_timestamp=dream_start)
                    completed += 1
                except Exception as e:
                    logger.warning(f"Post-ingest dream cycle {i + 1} failed: {e}")

        logger.info(f"Post-ingest dreams: {completed}/{cycles} cycles completed")

        return completed

    def run_full(self, limit=None):
        """Run the full pipeline: scan → ingest.

        Returns dict with scan_stats and ingest_stats.
        """
        scan_stats = self.run_scan()
        if "error" in scan_stats:
            return {"scan": scan_stats, "ingest": None}

        ingest_stats = self.run_ingest(limit=limit)
        return {"scan": scan_stats, "ingest": ingest_stats}

    def get_status(self):
        """Return pipeline status: ledger summary + recent activity."""
        summary = self.scanner.get_ledger_summary()

        recent = self.db.execute(
            "SELECT filename, belief_count, ingested_at FROM document_ledger "
            "WHERE status = 'ingested' ORDER BY ingested_at DESC LIMIT 10"
        ).fetchall()

        return {
            "ledger": summary,
            "recent_ingestions": [dict(r) for r in recent],
            "datafiles_dir": self.scanner.datafiles_dir,
            "datafiles_dirs": self.scanner.datafiles_dirs,
        }
