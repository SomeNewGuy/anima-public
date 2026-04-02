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

"""Document-aware retrieval for DMS conversation layer.

Provides document context alongside belief retrieval. Three-tier resolution:
1. Beliefs (graph-derived, fast) — with provenance pointers
2. Entity matches (deterministic, compressed)
3. Document excerpts (on-demand, only when coverage is thin)

This module does NOT modify core retrieval. It provides supplementary context
that the retrieval engine can inject into the system prompt.
"""

import logging
import os
import time
from datetime import datetime, timezone

logger = logging.getLogger("ingestion.document_retrieval")


class DocumentRetrieval:
    """Provides document-aware context for conversation queries."""

    def __init__(self, db_conn, config, semantic_memory=None):
        self.db = db_conn
        self.config = config
        self.semantic = semantic_memory
        extraction_cfg = config.get("extraction", {})
        self.datafiles_dir = extraction_cfg.get("datafiles_dir", "")
        self.enabled = bool(self.datafiles_dir)

    def get_document_context(self, query, entities, belief_ids=None,
                             max_entity_results=10, max_doc_excerpts=2):
        """Build document-aware context for a query.

        Args:
            query: User's query text.
            entities: Extracted entity names from the query.
            belief_ids: IDs of beliefs already retrieved for this query.
                Used to compute coverage and find provenance.
            max_entity_results: Max entity match results to include.
            max_doc_excerpts: Max document excerpts for thin coverage.

        Returns dict with:
            - document_refs: provenance pointers for retrieved beliefs
            - entity_matches: documents matching query entities
            - coverage: coverage assessment dict
            - excerpts: document excerpts (only if coverage is thin)
            - context_text: formatted string ready for system prompt injection
        """
        if not self.enabled:
            return {"context_text": "", "coverage": {"sufficient": True}, "diagnostics": None}

        t0 = time.monotonic()
        belief_ids = belief_ids or []

        # Step 1: Get provenance for retrieved beliefs
        doc_refs = self._get_belief_provenance(belief_ids)

        # Step 2: Entity search against document index
        entity_matches = self._search_entities(query, entities, max_entity_results)

        # Step 3: Compute coverage
        coverage = self._compute_coverage(belief_ids, doc_refs, entity_matches)

        # Step 4: Determine tier
        tier_initial = "belief" if belief_ids else "none"
        fallback_triggered = not coverage["sufficient"]
        tier_final = tier_initial

        # Step 5: If coverage is thin, pull document excerpts
        excerpts = []
        if fallback_triggered and self.datafiles_dir:
            excerpts = self._get_document_excerpts(
                query, entity_matches, doc_refs, max_doc_excerpts
            )
            if excerpts:
                tier_final = "document"
            elif entity_matches:
                tier_final = "entity"

        # Step 6: Format context text
        context_text = self._format_context(doc_refs, entity_matches, excerpts)

        latency_ms = int((time.monotonic() - t0) * 1000)

        # Step 7: Record diagnostics
        diagnostics = {
            "tier_initial": tier_initial,
            "fallback_triggered": fallback_triggered,
            "tier_final": tier_final,
            "belief_hits": coverage["belief_count"],
            "provenance_density": coverage["provenance_density"],
            "diversity": coverage["unique_source_docs"],
            "entity_matches": coverage["entity_doc_matches"],
            "excerpts_used": len(excerpts),
            "latency_ms": latency_ms,
        }
        self._log_query(query, diagnostics)

        return {
            "document_refs": doc_refs,
            "entity_matches": entity_matches,
            "coverage": coverage,
            "excerpts": excerpts,
            "context_text": context_text,
            "diagnostics": diagnostics,
        }

    def _log_query(self, query, diagnostics):
        """Record tier selection diagnostics to dms_query_log."""
        try:
            now = datetime.now(timezone.utc).isoformat()
            self.db.execute(
                """INSERT INTO dms_query_log
                   (query_text, tier_initial, fallback_triggered, tier_final,
                    belief_hits, provenance_density, diversity, entity_matches,
                    excerpts_used, latency_ms, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    query[:500], diagnostics["tier_initial"],
                    1 if diagnostics["fallback_triggered"] else 0,
                    diagnostics["tier_final"],
                    diagnostics["belief_hits"],
                    diagnostics["provenance_density"],
                    diagnostics["diversity"],
                    diagnostics["entity_matches"],
                    diagnostics["excerpts_used"],
                    diagnostics["latency_ms"],
                    now,
                ),
            )
            self.db.commit()
        except Exception as e:
            logger.warning(f"Failed to log query diagnostics: {e}")

    def _get_belief_provenance(self, belief_ids):
        """Get document sources for a list of belief IDs."""
        if not belief_ids:
            return []

        placeholders = ",".join("?" * len(belief_ids))
        rows = self.db.execute(
            f"""SELECT bsd.belief_id, bsd.document_sha, dl.filename, dl.file_path
                FROM belief_sources_documents bsd
                JOIN document_ledger dl ON bsd.document_sha = dl.sha256
                WHERE bsd.belief_id IN ({placeholders})""",
            belief_ids,
        ).fetchall()

        return [dict(r) for r in rows]

    def _search_entities(self, query, entities, limit):
        """Search document_entities for query terms and extracted entities."""
        results = []
        seen = set()

        # Search by extracted entity names
        search_terms = list(entities) if entities else []
        # Also split query into words for broader matching
        query_words = [w for w in query.lower().split() if len(w) > 3]
        search_terms.extend(query_words[:5])

        for term in search_terms:
            if len(results) >= limit:
                break
            rows = self.db.execute(
                """SELECT DISTINCT de.document_sha, dl.filename, de.entity_type,
                          de.entity_value, dl.belief_count
                   FROM document_entities de
                   JOIN document_ledger dl ON de.document_sha = dl.sha256
                   WHERE LOWER(de.entity_value) LIKE ?
                   LIMIT ?""",
                (f"%{term.lower()}%", limit),
            ).fetchall()

            for row in rows:
                key = (row["document_sha"], row["entity_type"], row["entity_value"])
                if key not in seen:
                    seen.add(key)
                    results.append(dict(row))

        return results[:limit]

    def _compute_coverage(self, belief_ids, doc_refs, entity_matches):
        """Compute coverage assessment for the query.

        Coverage is sufficient when:
        - belief_hits >= 3
        - AND provenance_density > 0 (at least some beliefs trace to documents)
        - AND diversity > 1 (beliefs from more than one document)
        """
        belief_count = len(belief_ids)
        provenance_count = len(doc_refs)
        unique_docs = len(set(r["document_sha"] for r in doc_refs)) if doc_refs else 0
        entity_doc_count = len(set(r["document_sha"] for r in entity_matches)) if entity_matches else 0

        provenance_density = provenance_count / max(belief_count, 1)

        sufficient = (
            belief_count >= 3
            and provenance_density > 0
            and unique_docs > 1
        )

        return {
            "sufficient": sufficient,
            "belief_count": belief_count,
            "provenance_count": provenance_count,
            "unique_source_docs": unique_docs,
            "provenance_density": round(provenance_density, 2),
            "entity_doc_matches": entity_doc_count,
        }

    def _get_document_excerpts(self, query, entity_matches, doc_refs, max_excerpts):
        """Read relevant sections from top-matching documents.

        Only called when coverage is thin — this is the fallback.
        """
        # Prioritize docs from entity matches, then provenance
        candidate_shas = []
        seen = set()

        for match in entity_matches:
            sha = match["document_sha"]
            if sha not in seen:
                seen.add(sha)
                candidate_shas.append((sha, match.get("file_path", match.get("filename", ""))))

        for ref in doc_refs:
            sha = ref["document_sha"]
            if sha not in seen:
                seen.add(sha)
                candidate_shas.append((sha, ref.get("file_path", ref.get("filename", ""))))

        excerpts = []
        for sha, file_path in candidate_shas[:max_excerpts]:
            # Resolve file path
            if self.datafiles_dir and file_path:
                filepath = os.path.join(self.datafiles_dir, file_path)
            else:
                # Look up path from ledger
                row = self.db.execute(
                    "SELECT file_path FROM document_ledger WHERE sha256 = ?", (sha,)
                ).fetchone()
                if not row:
                    continue
                filepath = os.path.join(self.datafiles_dir, row["file_path"])

            try:
                with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except Exception:
                continue

            # Truncate to reasonable excerpt size
            excerpt = content[:3000] if len(content) > 3000 else content

            # Get filename for display
            row = self.db.execute(
                "SELECT filename FROM document_ledger WHERE sha256 = ?", (sha,)
            ).fetchone()
            filename = row["filename"] if row else file_path

            excerpts.append({
                "document_sha": sha,
                "filename": filename,
                "excerpt": excerpt,
            })

        return excerpts

    def _format_context(self, doc_refs, entity_matches, excerpts):
        """Format document context as a string for system prompt injection."""
        parts = []

        # Document provenance for retrieved beliefs
        if doc_refs:
            unique_docs = {}
            for ref in doc_refs:
                sha = ref["document_sha"]
                if sha not in unique_docs:
                    unique_docs[sha] = ref["filename"]

            lines = ["[DOCUMENT SOURCES for your current beliefs:]"]
            for sha, filename in list(unique_docs.items())[:10]:
                lines.append(f"- {filename}")
            lines.append("When referencing these beliefs, cite the source document.")
            parts.append("\n".join(lines))

        # Entity matches — documents related to the query
        if entity_matches:
            seen_docs = {}
            for match in entity_matches:
                sha = match["document_sha"]
                if sha not in seen_docs:
                    seen_docs[sha] = {
                        "filename": match["filename"],
                        "entities": [],
                    }
                seen_docs[sha]["entities"].append(
                    f"{match['entity_type']}:{match['entity_value']}"
                )

            lines = ["[RELATED DOCUMENTS matching this query:]"]
            for sha, info in list(seen_docs.items())[:10]:
                entity_str = ", ".join(info["entities"][:3])
                lines.append(f"- {info['filename']} ({entity_str})")
            parts.append("\n".join(lines))

        # Document excerpts — fallback when coverage is thin
        if excerpts:
            for exc in excerpts:
                parts.append(
                    f"[DOCUMENT EXCERPT from {exc['filename']}:]\n"
                    f"{exc['excerpt']}\n"
                    f"[END EXCERPT]"
                )

        return "\n\n".join(parts)

    def get_corpus_summary(self):
        """Get lightweight corpus stats for system prompt injection.

        Returns a short string suitable for the system prompt.
        """
        if not self.enabled:
            return ""

        try:
            doc_count = self.db.execute(
                "SELECT COUNT(*) as cnt FROM document_ledger WHERE status = 'ingested'"
            ).fetchone()["cnt"]

            if doc_count == 0:
                return ""

            # Top topics
            topic_rows = self.db.execute(
                """SELECT entity_value, COUNT(*) as cnt
                   FROM document_entities
                   WHERE entity_type = 'topic'
                   GROUP BY LOWER(entity_value)
                   ORDER BY cnt DESC
                   LIMIT 10"""
            ).fetchall()
            topics = [r["entity_value"] for r in topic_rows]

            summary = (
                f"You have access to a corpus of {doc_count} documents"
            )
            if topics:
                summary += f" covering: {', '.join(topics)}"
            summary += (
                ". Your beliefs were extracted from these documents. "
                "When answering, cite source documents. "
                "When your beliefs don't cover something, say so — "
                "the user can search the documents directly."
            )
            return summary

        except Exception as e:
            logger.warning(f"Failed to get corpus summary: {e}")
            return ""
