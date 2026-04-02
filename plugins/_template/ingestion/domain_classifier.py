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

"""Self-organizing domain classifier for Research product.

Discovers domain structure from the corpus itself — no hardcoded keywords.
Works for any field — domains are discovered from the corpus, not hardcoded.

Architecture (Opus spec):
- Option A: Corpus-derived domains via embedding clusters
- Option B: LLM classification fallback for novel beliefs

On first ingestion, reads the corpus and identifies top N topic clusters.
Those become the domain labels. Every subsequent belief gets classified
against corpus-derived domains via embedding similarity. New domains can
emerge as the corpus grows.

The keyword classifier in semantic.py stays for core ANIMA's known domains.
Research product uses this self-organizing approach instead.
"""

import json
import logging
import numpy as np
from collections import Counter
from datetime import datetime, timezone

logger = logging.getLogger("ingestion.domain_classifier")




# Canonical domain base — self-organizing from corpus.
# Starts empty. Domains are discovered and created from ingested content
# via LLM classification. No hardcoded domain knowledge.
CANONICAL_DOMAINS = {}

# Garbage labels — hard kill on startup
GARBAGE_LABELS = {
    "external", "synthesis", "none_of_the", "unclassified", "other",
    "meta", "operator",
}


class DomainClassifier:
    """Domain classifier with fixed canonical base + self-organizing extension."""

    def __init__(self, embedding_engine, inference_engine=None, db=None, config=None):
        self.embeddings = embedding_engine
        self.inference = inference_engine
        self.db = db
        self.config = config or {}

        # Domain clusters: {domain_name: {centroid: [...], count: N}}
        self._domains = {}
        self._canonical_centroids = {}  # precomputed canonical embeddings
        self._initialized = False
        self._frozen = False

        # Config
        self.max_domains = self.config.get("research", {}).get("max_domains", 20)
        self.merge_threshold = 0.85
        self.assign_threshold = 0.50  # lowered from 0.55 — canonical domains are broader
        self.min_beliefs_per_domain = 3

        # Unmatched tracker for new domain creation
        self._unmatched_buffer = []  # beliefs that didn't match any canonical domain

    def _init_canonical_centroids(self):
        """Precompute embeddings for canonical domain descriptions."""
        if self._canonical_centroids:
            return
        for name, description in CANONICAL_DOMAINS.items():
            self._canonical_centroids[name] = self.embeddings.embed(description)
        logger.info(f"Canonical domain centroids initialized: {len(self._canonical_centroids)}")

    def startup_audit(self):
        """Run on startup: kill garbage labels, reclassify, merge duplicates."""
        if not self.db:
            return

        self._init_canonical_centroids()

        import json as _json

        # Find beliefs with garbage domain labels
        garbage_count = 0
        reclassified = 0
        rows = self.db.execute(
            "SELECT id, statement, topics FROM beliefs WHERE COALESCE(deprecated,0)=0"
        ).fetchall()

        for row in rows:
            try:
                topics = _json.loads(row["topics"] or "[]")
            except Exception:
                topics = []

            if not topics:
                continue

            domain = topics[0].lower().strip()

            # Check for garbage labels or truncated strings
            is_garbage = (
                domain in GARBAGE_LABELS
                or len(domain) > 30  # truncated garbage
                or "_and" in domain  # "cellular_signaling_and"
                or domain.startswith("domain_")
                or domain.startswith("none")
            )

            if is_garbage:
                garbage_count += 1
                # Reclassify against canonical domains
                new_domain = self._classify_canonical(row["statement"])
                if new_domain:
                    self.db.execute(
                        "UPDATE beliefs SET topics = ? WHERE id = ?",
                        (_json.dumps([new_domain]), row["id"]),
                    )
                    reclassified += 1

        # Merge near-duplicate domains
        merged = self._merge_duplicate_domains()

        if garbage_count or merged:
            self.db.commit()
            logger.info(
                f"Domain audit: {garbage_count} garbage labels found, "
                f"{reclassified} reclassified, {merged} domains merged"
            )

    def _classify_canonical(self, statement):
        """Classify a statement against canonical domain centroids only.

        No LLM call — pure embedding similarity. Fast.
        """
        self._init_canonical_centroids()

        try:
            emb = self.embeddings.embed(statement)
            best_domain = None
            best_sim = 0

            for name, centroid in self._canonical_centroids.items():
                sim = float(np.dot(emb, centroid) / (
                    np.linalg.norm(emb) * np.linalg.norm(centroid) + 1e-10
                ))
                if sim > best_sim:
                    best_sim = sim
                    best_domain = name

            if best_sim >= 0.35:  # low threshold — canonical descriptions are broad
                return best_domain
            return "unclassified"
        except Exception:
            return "unclassified"

    def _merge_duplicate_domains(self):
        """Merge near-duplicate domain labels via string similarity."""
        if not self.db:
            return 0

        import json as _json
        from collections import Counter

        # Count domain usage
        domain_counts = Counter()
        rows = self.db.execute(
            "SELECT topics FROM beliefs WHERE COALESCE(deprecated,0)=0"
        ).fetchall()
        for r in rows:
            try:
                topics = _json.loads(r["topics"] or "[]")
                if topics:
                    domain_counts[topics[0].lower().strip()] += 1
            except Exception:
                pass

        # Find merge candidates by string similarity
        domains = list(domain_counts.keys())
        merged = 0
        merge_map = {}  # old → new

        for i, d1 in enumerate(domains):
            for d2 in domains[i+1:]:
                # Check if one is a substring of the other
                if d1 in d2 or d2 in d1:
                    # Keep the shorter (more general) one
                    keep = d1 if len(d1) <= len(d2) else d2
                    drop = d2 if keep == d1 else d1
                    if drop not in merge_map:
                        merge_map[drop] = keep

        # Apply merges
        for old, new in merge_map.items():
            rows = self.db.execute(
                "SELECT id, topics FROM beliefs WHERE topics LIKE ?",
                (f'%"{old}"%',),
            ).fetchall()
            for r in rows:
                try:
                    topics = _json.loads(r["topics"] or "[]")
                    topics = [new if t == old else t for t in topics]
                    self.db.execute(
                        "UPDATE beliefs SET topics = ? WHERE id = ?",
                        (_json.dumps(topics), r["id"]),
                    )
                    merged += 1
                except Exception:
                    pass

        return merged

    def initialize_from_corpus(self, beliefs):
        """Build initial domain structure from existing beliefs.

        Clusters beliefs by embedding similarity, then names each cluster
        using the most representative terms.

        Args:
            beliefs: list of dicts with 'statement' and optionally 'id'
        """
        if not beliefs:
            return

        logger.info(f"Initializing domain classifier from {len(beliefs)} beliefs")

        # Embed all beliefs
        statements = [b.get("statement", "") for b in beliefs]
        embeddings = [self.embeddings.embed(s) for s in statements]

        # Simple agglomerative clustering via greedy assignment
        clusters = []  # list of {centroid, members}

        for i, emb in enumerate(embeddings):
            best_cluster = None
            best_sim = 0

            for j, cluster in enumerate(clusters):
                sim = self._cosine_similarity(emb, cluster["centroid"])
                if sim > best_sim:
                    best_sim = sim
                    best_cluster = j

            if best_cluster is not None and best_sim >= self.assign_threshold:
                # Add to existing cluster
                cluster = clusters[best_cluster]
                cluster["members"].append(i)
                # Update centroid (running average)
                n = len(cluster["members"])
                cluster["centroid"] = [
                    (c * (n - 1) + e) / n
                    for c, e in zip(cluster["centroid"], emb)
                ]
            else:
                # New cluster
                clusters.append({
                    "centroid": list(emb),
                    "members": [i],
                })

        # Merge very similar clusters
        merged = True
        while merged:
            merged = False
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    sim = self._cosine_similarity(
                        clusters[i]["centroid"], clusters[j]["centroid"]
                    )
                    if sim >= self.merge_threshold:
                        # Merge j into i — weighted by member count
                        n_i = len(clusters[i]["members"])
                        n_j = len(clusters[j]["members"])
                        w_i = n_i / (n_i + n_j)
                        w_j = n_j / (n_i + n_j)
                        clusters[i]["members"].extend(clusters[j]["members"])
                        clusters[i]["centroid"] = [
                            a * w_i + b * w_j for a, b in zip(
                                clusters[i]["centroid"], clusters[j]["centroid"]
                            )
                        ]
                        clusters.pop(j)
                        merged = True
                        break
                if merged:
                    break

        # Absorb tiny clusters into nearest large cluster
        large = [c for c in clusters if len(c["members"]) >= self.min_beliefs_per_domain]
        small = [c for c in clusters if len(c["members"]) < self.min_beliefs_per_domain]

        for sc in small:
            best_cluster = None
            best_sim = 0
            for lc in large:
                sim = self._cosine_similarity(sc["centroid"], lc["centroid"])
                if sim > best_sim:
                    best_sim = sim
                    best_cluster = lc
            if best_cluster:
                best_cluster["members"].extend(sc["members"])

        clusters = large if large else clusters[:self.max_domains]

        # Name each cluster
        self._domains = {}
        for idx, cluster in enumerate(clusters[:self.max_domains]):
            # Get representative statements
            member_stmts = [statements[i] for i in cluster["members"][:5]]
            domain_name = self._name_cluster(member_stmts, idx)

            self._domains[domain_name] = {
                "centroid": cluster["centroid"],
                "count": len(cluster["members"]),
            }

        self._initialized = True
        # Don't freeze on init — on a clean DB, clusters haven't formed yet.
        # Freeze happens after first extraction cycle populates the graph.
        if len(beliefs) >= 20:
            self._frozen = True
            logger.info(
                f"Domain classifier initialized and FROZEN ({len(beliefs)} beliefs): "
                f"{len(self._domains)} domains — "
                + ", ".join(f"{d} ({v['count']})" for d, v in self._domains.items())
            )
        else:
            logger.info(
                f"Domain classifier initialized (NOT frozen, {len(beliefs)} beliefs < 20): "
                f"{len(self._domains)} domains — "
                + ", ".join(f"{d} ({v['count']})" for d, v in self._domains.items())
            )

        # Persist to DB
        self._save_domains()

    def classify(self, statement):
        """Classify a belief into a domain.

        Priority: canonical domains first (fixed, stable), then corpus clusters,
        then LLM fallback.
        """
        # Try canonical domains first (fast, stable, no LLM)
        canonical = self._classify_canonical(statement)
        if canonical and canonical != "unclassified":
            return canonical

        if not self._initialized:
            self._load_domains()

        if not self._domains:
            return "unclassified"

        emb = self.embeddings.embed(statement)

        # Find nearest corpus-derived domain centroid
        best_domain = None
        best_sim = 0
        for name, data in self._domains.items():
            # Skip garbage labels in corpus domains
            if name.lower() in GARBAGE_LABELS:
                continue
            sim = self._cosine_similarity(emb, data["centroid"])
            if sim > best_sim:
                best_sim = sim
                best_domain = name

        if best_domain and best_sim >= self.assign_threshold:
            return best_domain

        # Fallback: LLM classification for novel beliefs
        # If frozen, LLM can classify into existing domains but cannot create new ones
        if self.inference:
            result = self._llm_classify(statement)
            if result and result != "unclassified":
                # Sanitize: reject garbage, conjunctions, truncation
                result_lower = result.lower().strip()
                if (result_lower in GARBAGE_LABELS
                        or "_and" in result_lower
                        or len(result_lower) > 25
                        or " and " in result_lower):
                    pass  # skip bad LLM output, fall through
                else:
                    return result

        # Fix #5: No "unclassified" allowed. Fall back to nearest known domain
        # by embedding similarity regardless of threshold.
        if best_domain:
            return best_domain  # Nearest domain even if below threshold

        # Last resort: use canonical classify with lowest threshold
        canonical_fallback = self._classify_canonical(statement)
        if canonical_fallback and canonical_fallback != "unclassified":
            return canonical_fallback

        return "general"  # Fallback when no domain can be determined

    def _name_cluster(self, representative_statements, fallback_idx):
        """Name a cluster from its representative beliefs.

        Uses LLM if available, otherwise extracts common noun phrases.
        """
        if self.inference and representative_statements:
            try:
                samples = "\n".join(f"- {s[:150]}" for s in representative_statements[:5])
                prompt = (
                    "These beliefs belong to the same research domain:\n\n"
                    f"{samples}\n\n"
                    "What research domain do they belong to? "
                    "Respond with exactly ONE or TWO words (e.g., 'immunology', "
                    "'molecular biology', 'drug resistance', 'gut microbiome'). "
                    "Nothing else.\n/no_think"
                )
                import re
                name = self.inference.generate_with_messages(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=20, task="triage",  # Use quality model for naming, not small
                    timeout=30,
                )
                name = re.sub(r"<think>.*?</think>", "", name, flags=re.DOTALL).strip()
                # Clean: lowercase, remove quotes/periods, take first 3 words max
                name = name.lower().strip('."\'').strip()
                # Remove common filler words the LLM adds
                name = re.sub(r'^(the |a |an |research |domain |of )+', '', name)
                words = [w for w in name.split() if len(w) > 1][:3]
                name = "_".join(words)
                # Reject garbage labels
                _bad_labels = {"domain", "none", "none_of", "none_of_the", "other",
                              "unknown", "general", "research", "science", "biology"}
                if (name and len(name) > 2
                        and not any(name.startswith(b) for b in _bad_labels)
                        and name not in _bad_labels):
                    return name
                # LLM returned generic — try harder with more specific prompt
                name2 = self.inference.generate_with_messages(
                    messages=[{"role": "user", "content":
                        f"Classify these research statements into a specific sub-field:\n{samples}\n\n"
                        f"Sub-field name (2 words max, e.g. 'drug resistance', 'immune evasion', 'epigenetics'):\n/no_think"}],
                    max_tokens=10, task="triage", timeout=20,
                )
                name2 = re.sub(r"<think>.*?</think>", "", name2, flags=re.DOTALL).strip()
                name2 = name2.lower().strip('."\'').strip()
                name2 = re.sub(r'^(the |a |an )+', '', name2)
                words2 = [w for w in name2.split() if len(w) > 1][:2]
                name2 = "_".join(words2)
                if name2 and len(name2) > 2:
                    return name2
            except Exception as e:
                logger.warning(f"LLM cluster naming failed: {e}")

        return f"domain_{fallback_idx}"

    def _llm_classify(self, statement):
        """Classify a single belief via LLM call.

        Only used when embedding similarity doesn't match any existing domain.
        """
        try:
            domain_list = ", ".join(self._domains.keys())
            import re
            prompt = (
                f"Classify this research belief into one of these domains: {domain_list}\n"
                f"Or name a NEW domain if none fit (one or two words).\n\n"
                f"Belief: {statement[:200]}\n\n"
                f"Domain (one or two words only):\n/no_think"
            )
            name = self.inference.generate_with_messages(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20, task="triage", timeout=20,
            )
            name = re.sub(r"<think>.*?</think>", "", name, flags=re.DOTALL).strip()
            name = name.lower().strip('."\'').strip()
            name = re.sub(r'^(the |a |an )+', '', name)
            words = [w for w in name.split() if len(w) > 1][:3]
            name = "_".join(words)

            # Reject garbage labels from LLM
            _bad = {"domain", "none", "none_of", "none_of_the", "other",
                   "unknown", "general", "research", "science", "biology"}
            if name in _bad or any(name.startswith(b) for b in _bad):
                return best_domain if best_domain else "unclassified"

            # If it's a new domain, add it — unless frozen
            if name and name not in self._domains:
                if self._frozen:
                    logger.debug(
                        f"Domain freeze: LLM proposed new domain '{name}' "
                        f"but classifier is frozen — returning 'unclassified'"
                    )
                    return "unclassified"
                emb = self.embeddings.embed(statement)
                self._domains[name] = {
                    "centroid": list(emb) if hasattr(emb, 'tolist') else emb,
                    "count": 1,
                }
                self._save_domains()
                logger.info(f"New domain emerged: {name}")

            return name if name else "unclassified"
        except Exception:
            return "unclassified"

    def _save_domains(self):
        """Persist domain centroids to DB."""
        if not self.db:
            return
        try:
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS domain_clusters ("
                "name TEXT PRIMARY KEY, centroid TEXT, belief_count INTEGER, "
                "created_at TEXT)"
            )
            now = datetime.now(timezone.utc).isoformat()
            for name, data in self._domains.items():
                # Store centroid as JSON array (compact)
                centroid_json = json.dumps(
                    [round(float(x), 6) for x in data["centroid"]]  # full dimensions, rounded for space
                )
                self.db.execute(
                    "INSERT OR REPLACE INTO domain_clusters (name, centroid, belief_count, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (name, centroid_json, data["count"], now),
                )
            self.db.commit()
        except Exception as e:
            logger.warning(f"Failed to save domains: {e}")

    def _load_domains(self):
        """Load domain centroids from DB."""
        if not self.db:
            return
        try:
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS domain_clusters ("
                "name TEXT PRIMARY KEY, centroid TEXT, belief_count INTEGER, "
                "created_at TEXT)"
            )
            rows = self.db.execute("SELECT * FROM domain_clusters").fetchall()
            for row in rows:
                centroid = json.loads(row["centroid"] if isinstance(row, dict) else row[1])
                name = row["name"] if isinstance(row, dict) else row[0]
                count = row["belief_count"] if isinstance(row, dict) else row[2]
                self._domains[name] = {"centroid": centroid, "count": count}
            if self._domains:
                self._initialized = True
                logger.info(f"Loaded {len(self._domains)} domains from DB")
        except Exception as e:
            logger.warning(f"Failed to load domains: {e}")

    def get_domains(self):
        """Return current domain structure for dashboard."""
        if not self._initialized:
            self._load_domains()
        return {
            name: {"count": data["count"]}
            for name, data in self._domains.items()
        }

    def _cosine_similarity(self, a, b):
        """Cosine similarity between two vectors."""
        a = np.array(a, dtype=np.float32)
        b = np.array(b, dtype=np.float32)
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(dot / norm) if norm > 0 else 0.0
