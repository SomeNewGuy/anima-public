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

"""Curiosity-to-corpus matcher — matches open curiosity gaps to document sections.

When the belief graph has gaps (questions the system can't answer from existing
knowledge), this module finds documents in the corpus that might contain answers.

Documents are read from disk and chunked on-demand — no pre-loading into ChromaDB.
Sections are embedded and compared against gap embeddings to find relevant matches.

This is the bridge between ANIMA's curiosity engine and the document corpus.
"""

import logging
import os
import re

import numpy as np

logger = logging.getLogger("ingestion.corpus_matcher")


def _chunk_document(content, filename, max_chunk_chars=2000):
    """Split a document into sections for targeted matching.

    Strategy:
    - Markdown (.md): split on headings (## or #)
    - Other text: split on double newlines (paragraphs)
    - Each chunk includes its heading/position for context
    """
    if not content or not content.strip():
        return []

    chunks = []
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".md":
        # Split on markdown headings
        sections = re.split(r'\n(?=#{1,3}\s)', content)
        for section in sections:
            section = section.strip()
            if not section:
                continue
            if len(section) > max_chunk_chars:
                # Sub-split large sections on paragraphs
                paragraphs = section.split("\n\n")
                current = ""
                for para in paragraphs:
                    if len(current) + len(para) > max_chunk_chars and current:
                        chunks.append(current.strip())
                        current = para
                    else:
                        current += "\n\n" + para if current else para
                if current.strip():
                    chunks.append(current.strip())
            else:
                chunks.append(section)
    else:
        # Split on double newlines
        paragraphs = content.split("\n\n")
        current = ""
        for para in paragraphs:
            if len(current) + len(para) > max_chunk_chars and current:
                chunks.append(current.strip())
                current = para
            else:
                current += "\n\n" + para if current else para
        if current.strip():
            chunks.append(current.strip())

    # Filter out tiny chunks (< 50 chars)
    return [c for c in chunks if len(c) >= 50]


class CorpusMatcher:
    """Matches curiosity gaps to document sections via embedding similarity."""

    def __init__(self, db_conn, embedding_engine, config):
        self.db = db_conn
        self.embeddings = embedding_engine
        self.config = config
        extraction_cfg = config.get("extraction", {})
        self.datafiles_dir = extraction_cfg.get("datafiles_dir", "")
        self.similarity_threshold = extraction_cfg.get(
            "corpus_match_threshold", 0.45
        )
        self.max_matches_per_gap = extraction_cfg.get(
            "max_corpus_matches_per_gap", 3
        )

    def match_gaps_to_corpus(self, gaps, max_docs=50):
        """Match a list of curiosity gaps against the document corpus.

        Args:
            gaps: List of gap dicts from curiosity engine (must have 'question' key).
            max_docs: Max documents to consider (by recency).

        Returns list of matches:
            [{"gap": gap_dict, "matches": [{"document_sha", "filename", "section", "similarity"}]}]
        """
        if not gaps or not self.datafiles_dir:
            return []

        # Get ingested documents — prioritize under-read files
        # Files with fewer beliefs get higher priority to prevent
        # re-read gravity wells (KRAS at 185 vs MSI at 6)
        docs = self.db.execute(
            """SELECT sha256, filename, file_path, file_type, belief_count
               FROM document_ledger WHERE status = 'ingested'
               ORDER BY COALESCE(belief_count, 0) ASC, ingested_at DESC LIMIT ?""",
            (max_docs,),
        ).fetchall()

        if not docs:
            return []

        # Embed all gaps
        gap_texts = [g["question"] for g in gaps]
        gap_embeddings = self.embeddings.embed_batch(gap_texts)

        # Process each document — chunk, embed sections, match
        results = []
        for gap_idx, gap in enumerate(gaps):
            gap_emb = gap_embeddings[gap_idx]
            gap_matches = []

            for doc in docs:
                filepath = os.path.join(self.datafiles_dir, doc["file_path"])
                try:
                    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                except Exception:
                    continue

                chunks = _chunk_document(content, doc["filename"])
                if not chunks:
                    continue

                # Embed chunks and find best match
                chunk_embeddings = self.embeddings.embed_batch(chunks)
                similarities = np.dot(chunk_embeddings, gap_emb) / (
                    np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(gap_emb)
                    + 1e-10
                )

                best_idx = int(np.argmax(similarities))
                best_sim = float(similarities[best_idx])

                if best_sim >= self.similarity_threshold:
                    gap_matches.append({
                        "document_sha": doc["sha256"],
                        "filename": doc["filename"],
                        "file_path": doc["file_path"],
                        "section": chunks[best_idx],
                        "similarity": round(best_sim, 3),
                    })

            # Sort by similarity, take top N
            gap_matches.sort(key=lambda x: x["similarity"], reverse=True)
            if gap_matches:
                results.append({
                    "gap": dict(gap),
                    "matches": gap_matches[:self.max_matches_per_gap],
                })

        logger.info(
            f"Corpus matching: {len(gaps)} gaps → {len(results)} with matches"
        )
        return results

    def match_single_gap(self, gap_text, max_docs=50):
        """Match a single gap question against the corpus.

        Convenience method for ad-hoc matching.
        Returns list of {document_sha, filename, section, similarity}.
        """
        result = self.match_gaps_to_corpus(
            [{"question": gap_text}], max_docs=max_docs
        )
        if result:
            return result[0]["matches"]
        return []
