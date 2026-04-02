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

"""Universal Knowledge Tree — Core ANIMA infrastructure.

Hierarchical tree organizing ALL knowledge. Ships with every instance.
Every product uses it. Never product-specific.

Layer 0: Root ("Reality / Existence")
Layer 1: 6 fixed roots (Physical, Biological, Cognitive, Social, Formal, Applied)
Layer 2: 24 fixed branches
Layer 3+: Dynamic, LLM-grown under 10 rules + 4 constraints

Beliefs get placed into the tree via embedding-first (fast) or LLM traversal
(for ambiguous cases). Each belief can have multiple paths (multi-parent).
"""

import json
import logging
import threading
import uuid
import numpy as np
from datetime import datetime, timezone

logger = logging.getLogger("core.knowledge_tree")


class _NoLock:
    """Dummy context manager when no lock provided."""
    def __enter__(self): return self
    def __exit__(self, *a): pass

# ---------------------------------------------------------------------------
# Fixed tree structure — Layers 0-2. Never modified by the system.
# ---------------------------------------------------------------------------

TREE_ROOT = {
    "id": "root",
    "name": "Reality",
    "layer": 0,
    "children": ["physical", "biological", "cognitive", "social", "formal", "applied"],
}

LAYER_1 = {
    "physical": {"name": "Physical", "description": "Matter, energy, space, time, physical laws"},
    "biological": {"name": "Biological", "description": "Life, cells, genetics, organisms, ecology"},
    "cognitive": {"name": "Cognitive", "description": "Mind, neural systems, perception, intelligence, behavior"},
    "social": {"name": "Social", "description": "Culture, economics, governance, institutions, history"},
    "formal": {"name": "Formal", "description": "Mathematics, logic, computation, statistics"},
    "applied": {"name": "Applied", "description": "Medicine, engineering, technology, agriculture"},
}

LAYER_2 = {
    # Physical
    "matter_materials": {"name": "Matter & Materials", "parent": "physical", "description": "Chemistry, materials science, states of matter"},
    "energy_forces": {"name": "Energy & Forces", "parent": "physical", "description": "Physics, thermodynamics, electromagnetic forces"},
    "space_time": {"name": "Space & Time", "parent": "physical", "description": "Cosmology, relativity, spatial dynamics"},
    "earth_environment": {"name": "Earth & Environment", "parent": "physical", "description": "Geology, climate, environmental science"},
    # Biological
    "molecular_cellular": {"name": "Molecular & Cellular", "parent": "biological", "description": "Cell biology, molecular mechanisms, signaling pathways, organelles"},
    "genetics_evolution": {"name": "Genetics & Evolution", "parent": "biological", "description": "DNA, mutations, epigenetics, evolutionary biology, heredity"},
    "organisms_physiology": {"name": "Organisms & Physiology", "parent": "biological", "description": "Organ systems, physiology, disease, immune system"},
    "ecology_systems": {"name": "Ecology & Systems", "parent": "biological", "description": "Ecosystems, populations, microbiome, symbiosis"},
    # Cognitive
    "neural_systems": {"name": "Neural Systems", "parent": "cognitive", "description": "Neuroscience, brain structure, neural networks"},
    "perception_consciousness": {"name": "Perception & Consciousness", "parent": "cognitive", "description": "Sensory processing, awareness, qualia"},
    "learning_intelligence": {"name": "Learning & Intelligence", "parent": "cognitive", "description": "Machine learning, AI, cognitive development"},
    "behavior": {"name": "Behavior", "parent": "cognitive", "description": "Psychology, decision-making, social behavior"},
    # Social
    "culture_history": {"name": "Culture & History", "parent": "social", "description": "Anthropology, history, cultural evolution"},
    "economics": {"name": "Economics", "parent": "social", "description": "Markets, trade, resource allocation"},
    "governance_power": {"name": "Governance & Power", "parent": "social", "description": "Politics, authority, institutions"},
    "law_institutions": {"name": "Law & Institutions", "parent": "social", "description": "Legal systems, regulation, organizational structures"},
    # Formal
    "mathematics": {"name": "Mathematics", "parent": "formal", "description": "Pure mathematics, number theory, algebra, geometry"},
    "logic": {"name": "Logic", "parent": "formal", "description": "Formal logic, reasoning systems, proofs"},
    "computation": {"name": "Computation", "parent": "formal", "description": "Computer science, algorithms, complexity"},
    "statistics_probability": {"name": "Statistics & Probability", "parent": "formal", "description": "Statistical methods, probability theory, data analysis"},
    # Applied
    "medicine_health": {"name": "Medicine & Health", "parent": "applied", "description": "Medical science, clinical practice, public health, pharmacology"},
    "engineering_technology": {"name": "Engineering & Technology", "parent": "applied", "description": "Engineering disciplines, technology development"},
    "agriculture_food": {"name": "Agriculture & Food", "parent": "applied", "description": "Farming, food science, nutrition"},
    "infrastructure_energy": {"name": "Infrastructure & Energy", "parent": "applied", "description": "Power systems, transportation, civil infrastructure"},
}

# Blacklisted node names — can never be tree nodes
NODE_BLACKLIST = {
    "synthesis", "external", "corpus", "web", "operator",
    "meta", "unknown", "other", "none", "unclassified",
    "domain_0", "none_of_the",
}


class KnowledgeTree:
    """Universal knowledge tree — persistent, hierarchical, multi-parent."""

    def __init__(self, db, embedding_engine, inference_engine=None, config=None):
        self.db = db
        self.embeddings = embedding_engine
        self.inference = inference_engine
        self.config = config or {}
        self._node_cache = {}  # id → {name, embedding, parent_ids, layer}
        self._core_embedding = None
        self._core_backfill_done = False
        self.tag_registry = None  # set externally after TagRegistry init
        self._tree_lock = threading.Lock()  # protects _node_cache + DB writes

    # ------------------------------------------------------------------
    # Schema + initialization
    # ------------------------------------------------------------------

    def ensure_schema(self):
        """Create knowledge_tree table if not exists. Add beliefs columns."""
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_tree (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                embedding BLOB,
                parent_ids TEXT DEFAULT '[]',
                layer INTEGER NOT NULL,
                belief_count INTEGER DEFAULT 0,
                created_at TEXT,
                last_used_at TEXT
            )
        """)

        # Add columns to beliefs if missing
        try:
            self.db.execute("ALTER TABLE beliefs ADD COLUMN tree_paths TEXT DEFAULT '[]'")
        except Exception:
            pass  # column exists
        try:
            self.db.execute("ALTER TABLE beliefs ADD COLUMN core_similarity REAL DEFAULT NULL")
        except Exception:
            pass

        self.db.commit()

    def initialize(self):
        """Pre-populate Layer 0-2 nodes. Runs once on first startup."""
        self.ensure_schema()

        existing = self.db.execute("SELECT COUNT(*) FROM knowledge_tree").fetchone()[0]
        if existing >= 31:
            self._load_cache()
            return  # already populated

        logger.info("Initializing knowledge tree with 31 fixed nodes")
        now = datetime.now(timezone.utc).isoformat()

        # Layer 0 — Root
        root_emb = self.embeddings.embed("All knowledge, reality, existence")
        self._insert_node("root", "Reality", "All knowledge", root_emb, [], 0, now)

        # Layer 1
        for node_id, data in LAYER_1.items():
            emb = self.embeddings.embed(data["description"])
            self._insert_node(node_id, data["name"], data["description"], emb, ["root"], 1, now)

        # Layer 2
        for node_id, data in LAYER_2.items():
            emb = self.embeddings.embed(data["description"])
            self._insert_node(node_id, data["name"], data["description"], emb, [data["parent"]], 2, now)

        self.db.commit()
        self._load_cache()
        logger.info(f"Knowledge tree initialized: {len(self._node_cache)} nodes")

    def _insert_node(self, node_id, name, description, embedding, parent_ids, layer, now):
        """Insert a tree node."""
        emb_bytes = embedding.tobytes() if hasattr(embedding, 'tobytes') else bytes(embedding)
        self.db.execute(
            "INSERT OR IGNORE INTO knowledge_tree "
            "(id, name, description, embedding, parent_ids, layer, belief_count, created_at, last_used_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)",
            (node_id, name, description, emb_bytes, json.dumps(parent_ids), layer, now, now),
        )

    def _load_cache(self):
        """Load all nodes into memory cache."""
        rows = self.db.execute("SELECT * FROM knowledge_tree").fetchall()
        self._node_cache = {}
        for r in rows:
            emb_bytes = r["embedding"]
            emb = np.frombuffer(emb_bytes, dtype=np.float32) if emb_bytes else None
            self._node_cache[r["id"]] = {
                "name": r["name"],
                "description": r["description"],
                "embedding": emb,
                "parent_ids": json.loads(r["parent_ids"] or "[]"),
                "layer": r["layer"],
                "belief_count": r["belief_count"],
            }

    # ------------------------------------------------------------------
    # Core similarity
    # ------------------------------------------------------------------

    def compute_core_embedding(self):
        """Compute and cache the research core question embedding."""
        core_q = self.config.get("research", {}).get("core_question", "")
        if core_q and self.embeddings:
            self._core_embedding = self.embeddings.embed(core_q)
            logger.info(f"Core question embedding computed: '{core_q[:60]}...'")

    def get_core_similarity(self, statement):
        """Compute cosine similarity between a statement and the core question."""
        if self._core_embedding is None:
            return None
        try:
            emb = self.embeddings.embed(statement)
            norm = np.linalg.norm(self._core_embedding) * np.linalg.norm(emb)
            if norm > 0:
                return float(np.dot(self._core_embedding, emb) / norm)
        except Exception:
            pass
        return None

    def backfill_core_similarity(self):
        """One-time backfill of core_similarity for existing beliefs."""
        if self._core_embedding is None:
            return

        # Check if already done
        needs_backfill = self.db.execute(
            "SELECT COUNT(*) FROM beliefs WHERE core_similarity IS NULL AND COALESCE(deprecated,0)=0"
        ).fetchone()[0]

        if needs_backfill == 0:
            return

        logger.info(f"Backfilling core_similarity for {needs_backfill} beliefs")
        rows = self.db.execute(
            "SELECT id, statement FROM beliefs WHERE core_similarity IS NULL AND COALESCE(deprecated,0)=0"
        ).fetchall()

        # Parallel — pure embedding math, no dependencies
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _compute(r):
            sim = self.get_core_similarity(r["statement"])
            return r["id"], sim

        count = 0
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_compute, r): r for r in rows}
            for f in as_completed(futures):
                try:
                    bid, sim = f.result()
                    if sim is not None:
                        self.db.execute(
                            "UPDATE beliefs SET core_similarity = ? WHERE id = ?",
                            (round(sim, 4), bid),
                        )
                        count += 1
                except Exception:
                    pass

        self.db.commit()
        logger.info(f"Core similarity backfill complete: {count} beliefs updated")

    # ------------------------------------------------------------------
    # Belief placement — embedding-first, LLM for ambiguous
    # ------------------------------------------------------------------

    def place_belief(self, statement, belief_id=None, _lock=None):
        """Place a belief into the tree. Returns list of paths.

        L1/L2: LLM classification (embeddings don't work at this level).
        L3+: Iterative descent — embedding match first, LLM for new nodes.
        Every belief gets placed. No "unplaced" state.

        _lock: optional threading.Lock for parallel placement.
        """
        if not self._node_cache:
            self._load_cache()

        import re

        # Step 1: LLM classifies L1 (pick up to 2 branches)
        l1_ids = self._classify_l1(statement)
        if not l1_ids:
            l1_ids = ["applied"]  # safe fallback

        paths = []

        for l1_id in l1_ids[:2]:  # max 2 paths
            # Step 2: LLM classifies L2 under selected L1
            l2_id = self._classify_l2(statement, l1_id)
            if not l2_id:
                paths.append([l1_id])
                continue

            path = [l1_id, l2_id]

            # Step 3: Iterative descent from L2 down
            # Embedding-first at each level (cheap). LLM only for new nodes.
            chain = self._iterative_descent(statement, l2_id)
            for node_id in chain:
                path.append(node_id)

            paths.append(path)

        # Update belief — lock for thread safety in parallel placement
        if belief_id:
            _ctx = _lock if _lock else _NoLock()
            with _ctx:
                self.db.execute(
                    "UPDATE beliefs SET tree_paths = ? WHERE id = ?",
                    (json.dumps(paths), belief_id),
                )
                # Increment belief_count on leaf nodes
                for path in paths:
                    if path:
                        leaf = path[-1]
                        self.db.execute(
                            "UPDATE knowledge_tree SET belief_count = belief_count + 1, "
                            "last_used_at = ? WHERE id = ?",
                            (datetime.now(timezone.utc).isoformat(), leaf),
                        )
                self.db.commit()

        # Derive entity tags (structural tags removed — tree_paths IS the structure)
        if belief_id and self.tag_registry:
            try:
                _ctx2 = _lock if _lock else _NoLock()
                with _ctx2:
                    self.tag_registry.derive_from_entities(belief_id)
            except Exception as e:
                logger.debug(f"Entity tag derivation failed for {belief_id}: {e}")

        return paths

    # ------------------------------------------------------------------
    # L1/L2 LLM classification
    # ------------------------------------------------------------------

    def _classify_l1(self, statement):
        """LLM classifies which Layer 1 branches a belief belongs in.

        Returns list of L1 node IDs (1-2 items).
        """
        if not self.inference:
            return ["applied"]

        import re
        l1_list = ", ".join(f"{k} ({v['name']}: {v['description']})" for k, v in LAYER_1.items())

        try:
            prompt = (
                f"Classify this statement into 1 or 2 of these top-level knowledge domains.\n\n"
                f"physical — The non-living universe: matter, energy, forces, particles, waves, fields, thermodynamics, chemistry of non-living systems, materials science, geology, astronomy, cosmology, fluid dynamics, optics, acoustics. Includes: physics, inorganic chemistry, earth science, atmospheric science. Does NOT include: biochemistry (-> biological), medical devices (-> applied), computational physics (-> formal if theoretical, physical if experimental).\n\n"
                f"biological — Living systems at all scales: molecular biology, biochemistry, cell biology, genetics, genomics, microbiology, immunology, physiology, anatomy, ecology, evolution, disease mechanisms, pathology, pharmacology of biological systems, toxicology. ALL cell signaling, ALL metabolic pathways, ALL disease biology belongs here. Does NOT include: clinical treatment decisions (-> applied), brain/mind/consciousness (-> cognitive), population health statistics (-> social or applied).\n\n"
                f"cognitive — Mind, brain, and intelligence: neuroscience, psychology, psychiatry, consciousness, perception, emotion, memory, learning theory, linguistics, artificial intelligence, cognitive science, behavioral science. Includes: brain imaging, neural networks (biological), computational models of cognition. Does NOT include: molecular cell signaling (-> biological), social behavior of groups (-> social).\n\n"
                f"social — Human collective systems: history, anthropology, sociology, economics, political science, law, ethics, philosophy, religion, cultural studies, demographics, public policy, international relations, media studies. Includes: organizational behavior, social psychology, public health policy. Does NOT include: individual psychology (-> cognitive), economic modeling (-> formal if pure math).\n\n"
                f"formal — Abstract structures independent of physical reality: pure mathematics, mathematical logic, set theory, number theory, algebra, topology, formal language theory, computation theory, information theory, category theory, statistics as pure methodology, game theory. Includes: proofs, theorems, algorithms as abstract objects. Does NOT include: applied statistics on real data (-> whichever domain the data belongs to), computational biology (-> biological), chemical formulas (-> physical or biological).\n\n"
                f"applied — Using knowledge from other domains to act on the world: medicine, clinical practice, surgery, engineering, technology development, agriculture, food science, architecture, urban planning, manufacturing, drug development, clinical trials, therapeutic strategies, diagnostics, public health interventions, environmental engineering. Includes: anything where the primary purpose is intervention, treatment, building, or solving a practical problem.\n\n"
                f"Statement: \"{statement[:300]}\"\n\n"
                f"Return ONLY the domain IDs separated by comma. Example: biological, applied\n/no_think"
            )
            result = self.inference.generate_with_messages(
                [{"role": "user", "content": prompt}],
                max_tokens=30, timeout=60, task="triage",
            )
            result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
            result = result.lower().strip().strip(".")

            ids = [x.strip() for x in result.split(",")]
            valid = [x for x in ids if x in LAYER_1]
            if valid:
                return valid[:2]
        except Exception as e:
            logger.debug(f"L1 classification failed: {e}")

        return ["applied"]

    def _classify_l2(self, statement, l1_id):
        """LLM classifies which Layer 2 node a belief belongs in under a given L1.

        Returns L2 node ID or None.
        """
        if not self.inference:
            return None

        import re
        # Get L2 children of this L1
        l2_children = {k: v for k, v in LAYER_2.items() if v["parent"] == l1_id}
        if not l2_children:
            return None

        l2_list = ", ".join(f"{k} ({v['name']}: {v['description']})" for k, v in l2_children.items())

        try:
            prompt = (
                f"Which subcategory best fits this statement?\n\n"
                f"Categories: {l2_list}\n\n"
                f"Statement: \"{statement[:300]}\"\n\n"
                f"Return ONLY the category ID. Example: medicine_health\n/no_think"
            )
            result = self.inference.generate_with_messages(
                [{"role": "user", "content": prompt}],
                max_tokens=20, timeout=60, task="triage",
            )
            result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
            result = result.lower().strip().strip(".")

            if result in l2_children:
                return result
            # Fuzzy match — LLM might return display name
            for k, v in l2_children.items():
                if k in result or v["name"].lower() in result:
                    return k
        except Exception as e:
            logger.debug(f"L2 classification failed: {e}")

        return None

    # ------------------------------------------------------------------
    # Branch creation (L3+ dynamic growth)
    # ------------------------------------------------------------------

    # Known compound/drug names — too specific for L3/L4 tree nodes.
    # Populated from plugin config. Empty by default.
    _DRUG_NAMES = frozenset()

    def _iterative_descent(self, statement, start_id):
        """Descend one level at a time from start_id. Returns list of node IDs.

        Two-stage decision at each level:
        Stage 1: REUSE — embedding match (>0.60) or LLM forced numbered selection
        Stage 2: CREATE — only on explicit NONE, separate LLM call, must justify

        Reuse is the default. Creation is a last resort.
        """
        import re

        path = []
        current = start_id
        ancestors = {start_id}
        new_count = 0
        emb = self.embeddings.embed(statement)

        for depth in range(20):
            children = self._get_children(current)

            if not children:
                # No children at all — must create first node here
                new_id = self._create_first_child(statement, current, emb)
                if new_id:
                    path.append(new_id)
                    self._selection_stats["create_first"] += 1
                break

            # --- Stage 1: REUSE (embedding then LLM numbered selection) ---

            # 1a. Embedding pre-filter — fast, catches clear matches
            scored = []
            for cid, cnode in children.items():
                if cnode["embedding"] is not None:
                    sim = self._cosine(emb, cnode["embedding"])
                    scored.append((cid, cnode, sim))
            scored.sort(key=lambda x: -x[2])

            # Strong embedding match — descend without LLM
            if scored and scored[0][2] > 0.65:
                best_id = scored[0][0]
                if best_id in ancestors:
                    break
                path.append(best_id)
                ancestors.add(best_id)
                current = best_id
                self._selection_stats["reuse_embedding"] += 1
                continue

            # 1b. LLM forced numbered selection from existing children
            if not self.inference:
                break

            selected = self._stage1_reuse(statement, current, children)

            if selected and selected != "NONE" and selected in children:
                if selected in ancestors:
                    break
                path.append(selected)
                ancestors.add(selected)
                current = selected
                self._selection_stats["reuse_llm"] += 1
                continue

            # Also check if LLM returned something that embedding-matches an existing child
            if selected and selected != "NONE":
                sel_emb = self.embeddings.embed(selected.replace("_", " "))
                for cid, cnode in children.items():
                    if cnode["embedding"] is not None:
                        sim = self._cosine(sel_emb, cnode["embedding"])
                        if sim > 0.60:
                            if cid in ancestors:
                                break
                            path.append(cid)
                            ancestors.add(cid)
                            current = cid
                            self._selection_stats["reuse_llm"] += 1
                            selected = None  # mark as handled
                            break
                if selected is None:
                    continue

            # --- Stage 2: CREATE (only on explicit NONE) ---
            new_count += 1
            if new_count > 3:
                break

            new_id = self._stage2_create(statement, current, children, emb)
            if new_id:
                path.append(new_id)
                ancestors.add(new_id)
                current = new_id
                self._selection_stats["create"] += 1
            else:
                break

        return path

    # Selection tracking — reuse vs create per cycle
    _selection_stats = {"reuse_embedding": 0, "reuse_llm": 0, "create": 0, "create_first": 0}

    @classmethod
    def get_selection_stats(cls):
        return dict(cls._selection_stats)

    @classmethod
    def reset_selection_stats(cls):
        cls._selection_stats = {"reuse_embedding": 0, "reuse_llm": 0, "create": 0, "create_first": 0}

    def _stage1_reuse(self, statement, parent_id, children):
        """Stage 1: Force the LLM to pick from existing children.

        Top 10 candidates by embedding similarity. Explicit NONE as option 0.
        Returns child ID if matched, 'NONE' if nothing fits, or None on error.
        """
        import re
        parent_name = self._node_cache.get(parent_id, {}).get("name", parent_id)

        # Rank children by embedding similarity to belief, take top 10
        emb = self.embeddings.embed(statement)
        scored_children = []
        for cid, cnode in children.items():
            if cnode["embedding"] is not None:
                sim = self._cosine(emb, cnode["embedding"])
                scored_children.append((cid, cnode, sim))
        scored_children.sort(key=lambda x: -x[2])
        top_children = scored_children[:10]

        # Build numbered list with belief counts and descriptions
        child_list = ["0. NONE - no suitable category exists"]
        child_lookup = {"0": "NONE"}
        for i, (cid, cnode, sim) in enumerate(top_children, 1):
            count = cnode.get("belief_count", 0)
            desc = cnode.get("description", "")[:60] if cnode.get("description") else ""
            name = cnode.get("name", cid)
            child_list.append(f"{i}. {name} ({count} beliefs){' - ' + desc if desc else ''}")
            child_lookup[str(i)] = cid
            child_lookup[cid] = cid
            child_lookup[name.lower()] = cid

        prompt = (
            f"You MUST pick one of these existing categories unless NONE of them "
            f"relate to this belief at all.\n\n"
            f"Categories under '{parent_name}':\n"
            f"{chr(10).join(child_list)}\n\n"
            f"Belief: \"{statement[:300]}\"\n\n"
            f"Prefer existing categories even if imperfect. Creating unnecessary "
            f"new categories degrades the system.\n\n"
            f"Pick the NUMBER of the best match. Write ONLY the number.\n"
            f"If nothing fits, write 0."
        )

        try:
            result = self.inference.generate_with_messages(
                [{"role": "user", "content": prompt}],
                max_tokens=50, timeout=60, task="triage",
            )
            result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
            result = re.sub(r"</?answer>", "", result).strip()
            pick = result.split("\n")[0].strip().strip("`.\"'(),")

            # Check for NONE
            if "none" in pick.lower():
                return "NONE"

            # Try number lookup
            num = re.search(r'\d+', pick)
            if num and num.group() in child_lookup:
                return child_lookup[num.group()]

            # Try name lookup
            pick_lower = re.sub(r'[^a-z0-9_]', '_', pick.lower()).strip('_')
            if pick_lower in child_lookup:
                return child_lookup[pick_lower]

            # Try partial match
            for key, cid in child_lookup.items():
                if pick_lower in key or key in pick_lower:
                    return cid

            return "NONE"

        except Exception as e:
            logger.debug(f"Stage 1 reuse failed at {parent_id}: {e}")
            return None

    def _stage2_create(self, statement, parent_id, children, emb):
        """Stage 2: Create a new node. Only called when Stage 1 returned NONE.

        Must be broader than belief, narrower than parent. Checked against
        existing siblings for duplicates.
        """
        import re

        # Check canonical identity against existing siblings
        for sid, snode in children.items():
            if snode["embedding"] is not None:
                sib_sim = self._cosine(emb, snode["embedding"])
                if sib_sim > 0.60:
                    # Close enough — use it (Stage 1 missed it)
                    return sid

        if not self.inference:
            return None

        parent_name = self._node_cache.get(parent_id, {}).get("name", parent_id)
        sibling_names = [c.get("name", cid) for cid, c in children.items()]
        sib_str = ", ".join(sibling_names[:15])

        prompt = (
            f"None of the existing categories under '{parent_name}' fit this belief.\n"
            f"Existing siblings: {sib_str}\n\n"
            f"Belief: \"{statement[:250]}\"\n\n"
            f"Propose a NEW general category (2-3 words, a scientific field or area).\n"
            f"- Must be broader than the belief but narrower than '{parent_name}'\n"
            f"- Must NOT duplicate any existing sibling\n"
            f"- Do NOT use specific drug, compound, or gene names\n\n"
            f"Return ONLY the category name in snake_case. Nothing else."
        )

        try:
            result = self.inference.generate_with_messages(
                [{"role": "user", "content": prompt}],
                max_tokens=50, timeout=60, task="triage",
            )
            result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
            result = re.sub(r"</?answer>", "", result).strip()
            node_name = result.split("\n")[0].strip().strip("`\"'.,").lower()
            node_id = re.sub(r'[^a-z0-9_]', '_', node_name).strip('_')[:48]
        except Exception as e:
            logger.debug(f"Stage 2 create failed at {parent_id}: {e}")
            return None

        if not node_id or node_id in ("stop", "none", parent_id):
            return None

        # Depth rejection
        words = node_id.split("_")
        if any(w in self._DRUG_NAMES for w in words):
            return None
        if len(words) > 4:
            return None

        # Check embedding against existing siblings one more time
        node_emb = self.embeddings.embed(node_name.replace("_", " "))
        for sid, snode in children.items():
            if snode["embedding"] is not None:
                sim = self._cosine(node_emb, snode["embedding"])
                if sim > 0.60:
                    logger.debug(f"Stage 2 creation averted — matched sibling {sid} (sim={sim:.2f})")
                    return sid

        display_name = node_name.replace("_", " ").title()
        return self._create_node({
            "id": node_id,
            "name": display_name,
            "parent": parent_id,
            "reason": statement[:200],
        })

    def _create_first_child(self, statement, parent_id, emb):
        """Create the first child node under a parent with no children.

        This is the only case where creation happens without Stage 1.
        """
        import re

        if not self.inference:
            return None

        parent_name = self._node_cache.get(parent_id, {}).get("name", parent_id)

        prompt = (
            f"What broad category under '{parent_name}' does this belief belong to?\n\n"
            f"Belief: \"{statement[:250]}\"\n\n"
            f"Propose a general category (2-3 words, a scientific field or area).\n"
            f"Must be broader than the belief but narrower than '{parent_name}'.\n\n"
            f"Return ONLY the category name in snake_case. Nothing else."
        )

        try:
            result = self.inference.generate_with_messages(
                [{"role": "user", "content": prompt}],
                max_tokens=50, timeout=60, task="triage",
            )
            result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
            result = re.sub(r"</?answer>", "", result).strip()
            node_name = result.split("\n")[0].strip().strip("`\"'.,").lower()
            node_id = re.sub(r'[^a-z0-9_]', '_', node_name).strip('_')[:48]
        except Exception as e:
            logger.debug(f"First child creation failed at {parent_id}: {e}")
            return None

        if not node_id or node_id in ("stop", "none", parent_id):
            return None

        words = node_id.split("_")
        if any(w in self._DRUG_NAMES for w in words):
            return None

        display_name = node_name.replace("_", " ").title()
        return self._create_node({
            "id": node_id,
            "name": display_name,
            "parent": parent_id,
            "reason": statement[:200],
        })

    def _get_path_to_root(self, node_id):
        """Get display names from root to this node."""
        path = []
        current = node_id
        visited = set()
        while current and current not in visited:
            visited.add(current)
            node = self._node_cache.get(current)
            if not node:
                break
            path.append(node.get("name", current))
            parents = node.get("parent_ids", [])
            current = parents[0] if parents else None
        path.reverse()
        return path

    def _get_children(self, parent_id):
        """Get all direct children of a node."""
        children = {}
        for nid, node in self._node_cache.items():
            if parent_id in node.get("parent_ids", []):
                children[nid] = node
        return children

    def _cosine(self, a, b):
        """Cosine similarity between two vectors."""
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / norm) if norm > 0 else 0.0

    # ------------------------------------------------------------------
    # Post-cycle cleanup — merge duplicates, prune dead, collapse unused
    # ------------------------------------------------------------------

    def post_cycle_cleanup(self):
        """End-of-cycle tree maintenance.

        1. Merge siblings at >0.85 embedding similarity
        2. Prune L3+ nodes with 1 belief and no reuse after 10 cycles
        3. Collapse branches with no beliefs and no children
        """
        if not self._node_cache:
            self._load_cache()

        stats = {"merged": 0, "pruned": 0, "collapsed": 0}

        # 1. Merge duplicate siblings
        parents_checked = set()
        for nid, node in list(self._node_cache.items()):
            if node["layer"] < 3:
                continue
            for pid in node.get("parent_ids", []):
                if pid in parents_checked:
                    continue
                parents_checked.add(pid)
                siblings = self._get_children(pid)
                sib_list = [(k, v) for k, v in siblings.items()]
                for i, (sid_a, snode_a) in enumerate(sib_list):
                    for sid_b, snode_b in sib_list[i+1:]:
                        if (snode_a["embedding"] is not None and
                                snode_b["embedding"] is not None):
                            sim = self._cosine(snode_a["embedding"], snode_b["embedding"])
                            if sim > 0.85:
                                keep = sid_a if snode_a.get("belief_count", 0) >= snode_b.get("belief_count", 0) else sid_b
                                drop = sid_b if keep == sid_a else sid_a
                                # Reparent children
                                for cid, cnode in self._node_cache.items():
                                    if drop in cnode.get("parent_ids", []):
                                        new_parents = [keep if p == drop else p for p in cnode["parent_ids"]]
                                        self.db.execute(
                                            "UPDATE knowledge_tree SET parent_ids = ? WHERE id = ?",
                                            (json.dumps(new_parents), cid),
                                        )
                                # Transfer belief count
                                self.db.execute(
                                    "UPDATE knowledge_tree SET belief_count = belief_count + ? WHERE id = ?",
                                    (snode_b.get("belief_count", 0) if keep == sid_a else snode_a.get("belief_count", 0), keep),
                                )
                                self.db.execute("DELETE FROM knowledge_tree WHERE id = ?", (drop,))
                                stats["merged"] += 1
                                logger.info(f"Cleanup merged: {drop} → {keep} (sim={sim:.2f})")

        # 2. Prune dead nodes: L3+ with 0 beliefs, no children, existed 10+ cycles
        for nid, node in list(self._node_cache.items()):
            if node["layer"] < 3:
                continue
            if node.get("belief_count", 0) > 0:
                continue
            children = self._get_children(nid)
            if children:
                continue
            # Check age — only prune if old enough
            created = node.get("created_at", "")
            if created:
                try:
                    age = (datetime.now(timezone.utc) - datetime.fromisoformat(created)).days
                    if age < 1:  # don't prune nodes created this session
                        continue
                except Exception:
                    pass
            self.db.execute("DELETE FROM knowledge_tree WHERE id = ?", (nid,))
            stats["pruned"] += 1

        # 3. Collapse empty branch points (no beliefs, no grandchildren)
        for nid, node in list(self._node_cache.items()):
            if node["layer"] < 3:
                continue
            if node.get("belief_count", 0) > 0:
                continue
            children = self._get_children(nid)
            if not children:
                continue
            # Check if ALL children are also empty with no sub-children
            all_empty = all(
                c.get("belief_count", 0) == 0 and not self._get_children(cid)
                for cid, c in children.items()
            )
            if all_empty:
                for cid in children:
                    self.db.execute("DELETE FROM knowledge_tree WHERE id = ?", (cid,))
                self.db.execute("DELETE FROM knowledge_tree WHERE id = ?", (nid,))
                stats["collapsed"] += 1

        self.db.commit()
        self._load_cache()

        if any(v > 0 for v in stats.values()):
            logger.info(f"Tree cleanup: {stats}")
        return stats

    # ------------------------------------------------------------------
    # Node creation with constraints
    # ------------------------------------------------------------------

    def _create_node(self, node_spec):
        """Create a new Layer 3+ node with all constraints enforced."""
        node_id = node_spec.get("id", "").lower().replace(" ", "_")
        name = node_spec.get("name", "")
        parent_id = node_spec.get("parent", "")
        reason = node_spec.get("reason", "")

        # Rule 3: No conjunctions
        if any(conj in name.lower() for conj in [" and ", " with ", " & ", "/"]):
            logger.info(f"Node creation rejected (conjunction): {name}")
            return None

        # Rule 8 + 10: Blacklist check
        if node_id in NODE_BLACKLIST or name.lower() in NODE_BLACKLIST:
            logger.info(f"Node creation rejected (blacklisted): {name}")
            return None

        # Constraint 2: Depth stopping — reason must be substantive
        if not reason or len(reason) < 10:
            logger.info(f"Node creation rejected (no reason): {name}")
            return None

        # Parent must exist
        parent = self._node_cache.get(parent_id)
        if not parent:
            logger.info(f"Node creation rejected (parent not found): {parent_id}")
            return None

        # Compute embedding
        emb = self.embeddings.embed(f"{name}: {reason}")

        # Rule 1: Parent validation — must be similar to parent
        # Relaxed for L2 parents: their generic descriptions produce low similarity
        # with any specific content. L3+ parents have specific embeddings.
        if parent["embedding"] is not None:
            parent_sim = self._cosine(emb, parent["embedding"])
            min_sim = 0.15 if parent["layer"] <= 2 else 0.40
            if parent_sim < min_sim:
                logger.info(f"Node creation rejected (too dissimilar to parent {parent_sim:.2f} < {min_sim}): {name}")
                return None

        # Rule 7: Depth must add information — not just a reword
        if parent["embedding"] is not None:
            if parent_sim > 0.95:
                logger.info(f"Node creation rejected (too similar to parent {parent_sim:.2f}): {name}")
                return None

        # Constraint 1: Canonical identity — check siblings
        siblings = self._get_children(parent_id)
        for sid, snode in siblings.items():
            if snode["embedding"] is not None:
                sib_sim = self._cosine(emb, snode["embedding"])
                if sib_sim > 0.85:
                    logger.info(f"Node creation: merged with existing sibling {sid} (sim={sib_sim:.2f})")
                    return sid  # use existing node

        # Create the node — lock protects DB + cache for thread safety
        with self._tree_lock:
            # Re-check siblings under lock (another thread may have created it)
            siblings = self._get_children(parent_id)
            for sid, snode in siblings.items():
                if snode["embedding"] is not None:
                    sib_sim = self._cosine(emb, snode["embedding"])
                    if sib_sim > 0.85:
                        return sid

            now = datetime.now(timezone.utc).isoformat()
            layer = parent["layer"] + 1
            self._insert_node(node_id, name, reason, emb, [parent_id], layer, now)
            self.db.commit()

            self._node_cache[node_id] = {
                "name": name,
                "description": reason,
                "embedding": emb,
                "parent_ids": [parent_id],
                "layer": layer,
                "belief_count": 0,
            }

        logger.info(f"New tree node: {name} ({node_id}) under {parent_id} at layer {layer}")
        return node_id

    # ------------------------------------------------------------------
    # Startup audit
    # ------------------------------------------------------------------

    def run_audit(self):
        """Startup audit: deduplicate, collapse empty, fix orphans."""
        if not self._node_cache:
            self._load_cache()

        issues = {"duplicates_merged": 0, "empty_collapsed": 0, "orphans_fixed": 0}

        # Check for duplicate nodes at same layer
        by_layer = {}
        for nid, node in self._node_cache.items():
            by_layer.setdefault(node["layer"], []).append((nid, node))

        for layer, nodes in by_layer.items():
            if layer < 3:
                continue  # don't touch fixed layers
            for i, (nid_a, node_a) in enumerate(nodes):
                for nid_b, node_b in nodes[i+1:]:
                    if node_a["embedding"] is not None and node_b["embedding"] is not None:
                        sim = self._cosine(node_a["embedding"], node_b["embedding"])
                        if sim > 0.85:
                            # Merge: keep the one with more beliefs
                            keep = nid_a if node_a["belief_count"] >= node_b["belief_count"] else nid_b
                            drop = nid_b if keep == nid_a else nid_a
                            # Reparent children of dropped node
                            for cid, cnode in self._node_cache.items():
                                if drop in cnode.get("parent_ids", []):
                                    new_parents = [keep if p == drop else p for p in cnode["parent_ids"]]
                                    self.db.execute(
                                        "UPDATE knowledge_tree SET parent_ids = ? WHERE id = ?",
                                        (json.dumps(new_parents), cid),
                                    )
                            self.db.execute("DELETE FROM knowledge_tree WHERE id = ?", (drop,))
                            issues["duplicates_merged"] += 1

        # Rule 9: Collapse empty nodes (belief_count < 2 for 10+ cycles)
        # Simplified: collapse Layer 3+ nodes with 0 beliefs
        for nid, node in list(self._node_cache.items()):
            if node["layer"] >= 3 and node["belief_count"] == 0:
                # Check if it has children — don't collapse if it's a branch point
                children = self._get_children(nid)
                if not children:
                    self.db.execute("DELETE FROM knowledge_tree WHERE id = ?", (nid,))
                    issues["empty_collapsed"] += 1

        self.db.commit()
        self._load_cache()

        if any(v > 0 for v in issues.values()):
            logger.info(f"Tree audit: {issues}")

        return issues

    # ------------------------------------------------------------------
    # Depth/centrality metric
    # ------------------------------------------------------------------

    def compute_depth_scores(self):
        """Compute depth/centrality scores for all active beliefs.

        Returns dict: belief_id → depth_score
        depth_score = f(degree, cross_branch_edges, confidence, recency)
        """
        scores = {}
        try:
            # Get degree per belief
            degree_rows = self.db.execute("""
                SELECT id, COUNT(*) as deg FROM (
                    SELECT belief_a as id FROM belief_links WHERE active=1
                    UNION ALL
                    SELECT belief_b as id FROM belief_links WHERE active=1
                ) GROUP BY id
            """).fetchall()
            degrees = {r["id"]: r["deg"] for r in degree_rows}

            # Get cross-branch edges (beliefs in different Layer 1 roots)
            # Simplified: count edges where the two beliefs have different tree_paths[0][0]
            cross_branch = {}
            edge_rows = self.db.execute("""
                SELECT bl.belief_a, bl.belief_b, ba.tree_paths as tp_a, bb.tree_paths as tp_b
                FROM belief_links bl
                JOIN beliefs ba ON bl.belief_a = ba.id
                JOIN beliefs bb ON bl.belief_b = bb.id
                WHERE bl.active = 1
            """).fetchall()
            for er in edge_rows:
                try:
                    pa = json.loads(er["tp_a"] or "[]")
                    pb = json.loads(er["tp_b"] or "[]")
                    root_a = pa[0][0] if pa and pa[0] else None
                    root_b = pb[0][0] if pb and pb[0] else None
                    if root_a and root_b and root_a != root_b:
                        cross_branch[er["belief_a"]] = cross_branch.get(er["belief_a"], 0) + 1
                        cross_branch[er["belief_b"]] = cross_branch.get(er["belief_b"], 0) + 1
                except Exception:
                    pass

            # Get confidence
            belief_rows = self.db.execute(
                "SELECT id, confidence FROM beliefs WHERE COALESCE(deprecated,0)=0"
            ).fetchall()

            for br in belief_rows:
                bid = br["id"]
                deg = degrees.get(bid, 0)
                cross = cross_branch.get(bid, 0)
                conf = br["confidence"] or 0.5

                # depth_score: degree contributes most, cross-branch edges are high value
                score = (deg * 1.0) + (cross * 2.0) + (conf * 0.5)
                scores[bid] = round(score, 2)

        except Exception as e:
            logger.warning(f"Depth score computation failed: {e}")

        return scores

    # ------------------------------------------------------------------
    # Meta-domain support (computed from tree structure)
    # ------------------------------------------------------------------

    def get_meta_domain(self, tree_paths):
        """Get the Layer 1 root(s) for a belief's tree paths.

        This IS the meta-domain — the broadest category. Computed from
        the tree structure, not a config mapping.
        """
        roots = set()
        try:
            paths = json.loads(tree_paths) if isinstance(tree_paths, str) else (tree_paths or [])
            for path in paths:
                if path and len(path) >= 1:
                    roots.add(path[0])
        except Exception:
            pass
        return list(roots)
