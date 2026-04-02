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

"""Semantic and text-based duplicate detection for beliefs.

Two detection modes:
1. Embedding similarity (cosine) — catches semantically similar beliefs
2. Normalized text comparison — catches restatements with different word order,
   synonym use, or phrasing (e.g., "5-fluorouracil" vs "5-FU")
"""

import re

import numpy as np

# ---- Statement normalization ----

# Synonym map: domain terms → canonical short forms.
# Longest-first replacement prevents partial matches.
# Load from plugin config per instance so each domain can provide
# its own synonym maps. Empty by default — dedup uses embedding
# similarity when no synonyms are configured.
_SYNONYM_MAP = {}

_NORM_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "must",
    "of", "in", "to", "for", "with", "on", "at", "from", "by", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "about", "against", "within",
    "that", "which", "who", "whom", "this", "these", "those",
    "and", "but", "or", "nor", "not", "so", "yet",
    "it", "its", "their", "them", "they",
    "also", "both", "each", "such", "other", "than", "more", "most",
}


def normalize_statement(text):
    """Normalize a belief statement for dedup comparison.

    Steps:
    1. Lowercase
    2. Replace synonyms with canonical forms
    3. Strip punctuation (keep hyphens in compound terms)
    4. Remove stopwords
    5. Sort remaining words (order-independent comparison)

    Returns normalized string for comparison.
    """
    if not text:
        return ""

    text = text.lower().strip()

    # Replace synonyms (longest first to avoid partial matches)
    for synonym, canonical in sorted(
        _SYNONYM_MAP.items(), key=lambda x: -len(x[0])
    ):
        text = text.replace(synonym, canonical)

    # Strip punctuation except hyphens in compound terms
    text = re.sub(r'[^\w\s-]', ' ', text)

    # Split, remove stopwords and very short words, sort
    words = text.split()
    words = [w for w in words if w not in _NORM_STOPWORDS and len(w) > 1]
    words.sort()

    return " ".join(words)


def is_duplicate(candidate_emb, corpus_embeddings, threshold=0.85):
    """Check if candidate embedding is a semantic duplicate of any in corpus.

    Args:
        candidate_emb: pre-computed embedding vector for the candidate
        corpus_embeddings: list of (embedding_vector, item) tuples
        threshold: cosine similarity threshold

    Returns (is_dup, matched_item_or_none).
    """
    if not corpus_embeddings or candidate_emb is None:
        return False, None

    cand_norm = np.linalg.norm(candidate_emb)
    if cand_norm == 0:
        return False, None

    for emb, item in corpus_embeddings:
        if emb is None:
            continue
        norm = cand_norm * np.linalg.norm(emb)
        if norm == 0:
            continue
        sim = float(np.dot(candidate_emb, emb) / norm)
        if sim >= threshold:
            return True, item

    return False, None


def is_normalized_duplicate(candidate_stmt, corpus_statements, threshold=0.90):
    """Check if candidate statement is a normalized text duplicate.

    Uses statement normalization (synonym replacement, stopword removal,
    word reordering) to detect near-duplicates that embedding similarity
    might miss (e.g., same claim restated with different word order).

    Args:
        candidate_stmt: raw statement text
        corpus_statements: list of raw statement texts to check against
        threshold: Jaccard word overlap threshold (0.90 = 90% word overlap)

    Returns (is_dup, matched_statement_or_none).
    """
    if not corpus_statements or not candidate_stmt:
        return False, None

    norm_cand = normalize_statement(candidate_stmt)
    cand_words = set(norm_cand.split())

    if not cand_words:
        return False, None

    for existing in corpus_statements:
        norm_existing = normalize_statement(existing)
        existing_words = set(norm_existing.split())

        if not existing_words:
            continue

        # Jaccard similarity: |intersection| / |union|
        intersection = cand_words & existing_words
        union = cand_words | existing_words
        if union:
            similarity = len(intersection) / len(union)
            if similarity >= threshold:
                return True, existing

    return False, None
