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

"""Web interface for ANIMA — FastAPI backend serving chat + dashboard.

Architecture:
- Single PersistenceCLI instance created in lifespan handler
- Chat serialized via threading.Lock + single-thread ThreadPoolExecutor
  (GPU can only do one inference at a time)
- Dashboard endpoints use separate read-only SQLite connections
"""

import json
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import logging

# Configure root logger so all ANIMA loggers (evolution, web_server, etc.)
# output to stderr (captured by nohup → log file).
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("web_server")

# Resolve paths relative to project root
_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
_DATA_DIR = os.environ.get(
    "ANIMA_DATA_DIR",
    os.path.join(_PROJECT_ROOT, "data"),
)
_PERSISTENCE_DB = os.path.normpath(
    os.path.join(_DATA_DIR, "sqlite", "persistence.db")
)
_TELEMETRY_DB = os.path.normpath(
    os.path.join(_DATA_DIR, "telemetry.db")
)

# Plugin discovery — runs at import time (cheap: just reads plugin.toml files)
from core.plugin_loader import PluginLoader
_plugin_loader = PluginLoader(os.path.join(_PROJECT_ROOT, "plugins"))
_plugin_loader.discover()

# Shared state — populated during lifespan
_cli = None
_chat_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=4)
_task_queue = None  # lazy init on first /queue/submit


def _ro_connect(db_path):
    """Open a read-only SQLite connection."""
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


_soak_stop = threading.Event()


def _regenerate_belief_graph():
    """Re-run belief graph tool to update the static HTML/JSON files."""
    try:
        import subprocess
        script = os.path.join(_PROJECT_ROOT, "analysis", "tools", "belief_graph.py")
        cmd = ["python3", script]
        # Instance-specific paths for DMS isolation
        if os.environ.get("ANIMA_DATA_DIR"):
            db = os.path.join(_DATA_DIR, "sqlite", "persistence.db")
            out = os.path.join(os.path.dirname(_DATA_DIR), "analysis", "output")
            cmd.extend(["--db", db, "--output-dir", out])
        result = subprocess.run(
            cmd,
            cwd=_PROJECT_ROOT,
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            logger.info("Belief graph regenerated")
        else:
            logger.warning(f"Belief graph regeneration failed: {result.stderr[:200]}")
    except Exception as e:
        logger.warning(f"Belief graph regeneration error: {e}")


def _soak_test_loop():
    """Autonomous soak test loop — runs exploration + fatigue on a timer.

    Protections:
    - Reads config each cycle (flip enabled=false to stop early)
    - Iteration timeout: kills exploration if it takes > cycle_timeout_minutes
    - Inference health check: skips cycle if server is down
    - Circuit breaker: stops after N consecutive inference failures
    - Cycle counter cap: hard limit on total cycles per soak session
    """
    import time
    import requests as _requests
    import toml as _toml

    def _reload_config():
        p = os.environ.get(
            "ANIMA_CONFIG",
            os.path.normpath(os.path.join(_PROJECT_ROOT, "config", "settings.toml")),
        )
        return _toml.load(p)

    cfg = _reload_config().get("soak_test", {})
    interval = cfg.get("interval_minutes", 10) * 60
    max_seconds = cfg.get("max_hours", 6) * 3600
    max_cycles = cfg.get("max_cycles", 50)
    cycle_timeout = cfg.get("cycle_timeout_minutes", 15) * 60
    health_failures_max = cfg.get("health_check_failures_max", 3)
    start_time = time.monotonic()

    # Config verification — log all key parameters from the LOADED config
    # to catch code-on-disk vs code-in-process mismatches
    full_cfg = _reload_config()
    ref_cfg = full_cfg.get("reflection", {})
    sched_mode = _cli.scheduler.mode if _cli.scheduler else "none"
    sched_interval = _cli.scheduler.interval if _cli.scheduler else 0
    logger.info(
        f"Soak test started: interval={interval}s, max={max_seconds}s, "
        f"max_cycles={max_cycles}, cycle_timeout={cycle_timeout}s, "
        f"consolidation={sched_mode}/interval={sched_interval}"
    )
    logger.info(
        f"Soak config verification: max_hours={cfg.get('max_hours', 6)}, "
        f"reflection_interval={ref_cfg.get('reflection_interval', 5)}, "
        f"dream_sim_min={ref_cfg.get('dream_similarity_min', 0.4)}, "
        f"dream_sim_max={ref_cfg.get('dream_similarity_max', 0.75)}, "
        f"consolidation={sched_mode}/interval={sched_interval}"
    )

    cycle = 0
    consecutive_failures = 0

    while not _soak_stop.is_set():
        _soak_stop.wait(timeout=interval)
        if _soak_stop.is_set():
            break

        # Re-read config to check if still enabled
        try:
            cfg = _reload_config().get("soak_test", {})
            if not cfg.get("enabled", False):
                logger.info("Soak test: disabled in config, stopping")
                break
        except Exception:
            pass

        elapsed = time.monotonic() - start_time
        if elapsed >= max_seconds:
            logger.info(f"Soak test: max duration reached ({max_seconds}s), stopping")
            break

        if cycle >= max_cycles:
            logger.info(f"Soak test: max cycles reached ({max_cycles}), stopping")
            break

        if not _cli or _cli._sleeping:
            logger.info("Soak test: ANIMA sleeping, skipping cycle")
            continue

        # Inference health check — skip cycle if server unreachable
        try:
            server_url = _cli.config.get("hardware", {}).get(
                "inference_server", "http://127.0.0.1:8080"
            )
            r = _requests.get(f"{server_url}/health", timeout=5)
            if r.status_code != 200:
                raise ConnectionError(f"status {r.status_code}")
            consecutive_failures = 0
        except Exception as e:
            consecutive_failures += 1
            logger.warning(
                f"Soak test: inference health check failed ({consecutive_failures}/"
                f"{health_failures_max}): {e}"
            )
            if consecutive_failures >= health_failures_max:
                logger.error(
                    f"Soak test: {health_failures_max} consecutive inference failures, "
                    f"stopping soak (circuit breaker)"
                )
                break
            continue

        cycle += 1
        logger.info(f"Soak test cycle {cycle}/{max_cycles} — triggering exploration window")

        # Run exploration with timeout
        cycle_result = [None]  # mutable container for thread result
        cycle_error = [None]

        def _run_cycle():
            try:
                with _chat_lock:
                    _cli.exploration_engine._soak_mode = True
                    try:
                        _cli._run_exploration_window()
                    finally:
                        _cli.exploration_engine._soak_mode = False
                    _cli._check_fatigue()
                cycle_result[0] = True
            except Exception as e:
                cycle_error[0] = e

        cycle_thread = threading.Thread(target=_run_cycle, daemon=True)
        cycle_thread.start()
        cycle_thread.join(timeout=cycle_timeout)

        if cycle_thread.is_alive():
            logger.error(
                f"Soak test cycle {cycle}: TIMEOUT after {cycle_timeout}s — "
                f"exploration hung, moving to next cycle"
            )
            # Thread is stuck but daemon=True, so it won't block shutdown.
            # The _chat_lock will be held until the stuck thread finishes,
            # so future cycles will block on lock acquisition. We use the
            # stop event to signal the exploration engine to abort.
            if hasattr(_cli, 'exploration_engine') and _cli.exploration_engine:
                _cli.exploration_engine.request_stop()
            # Wait a bit for the stuck thread to notice and release the lock
            cycle_thread.join(timeout=30)
            if cycle_thread.is_alive():
                logger.error(
                    f"Soak test cycle {cycle}: stuck thread did not release after "
                    f"abort signal — stopping soak to prevent lock contention"
                )
                break
        elif cycle_error[0]:
            logger.error(f"Soak test cycle {cycle} failed: {cycle_error[0]}", exc_info=False)

        # Consolidation runs even if the cycle errored — don't skip scheduled consolidations.
        # Scheduler decides whether this cycle triggers consolidation.
        if not cycle_thread.is_alive() and _cli.scheduler and _cli.scheduler.should_consolidate(cycle):
            # Health-check inference server first — if previous cycle timed out,
            # the server may still be processing. Wait for it to be ready.
            server_ready = False
            for _attempt in range(3):
                try:
                    r = _requests.get(f"{server_url}/health", timeout=10)
                    if r.status_code == 200:
                        server_ready = True
                        break
                except Exception:
                    pass
                logger.info(f"Soak test: inference server not ready, waiting 30s before consolidation (attempt {_attempt + 1}/3)")
                time.sleep(30)

            if not server_ready:
                logger.error(f"Soak test cycle {cycle}: skipping consolidation — inference server unhealthy after 3 attempts")
            else:
                logger.info(
                    f"Soak test cycle {cycle}: scheduled consolidation "
                    f"(scheduler mode={_cli.scheduler.mode})"
                )
                try:
                    with _chat_lock:
                        _cli._sleep_cycle(extraction_context="exploration")
                    _regenerate_belief_graph()
                except Exception as e:
                    logger.error(f"Soak test forced consolidation failed: {e}")

    logger.info(f"Soak test ended after {cycle} cycles")


_blade_runner = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cli, _blade_runner

    _load_encrypted_keys()

    # ── Blade Architecture: one core, all plugins loaded ──
    from core.blade_runner import BladeRunner

    _blade_runner = BladeRunner()
    _blade_runner.initialize_shared()

    # Determine which plugins to load
    only_plugin = os.environ.get("ANIMA_ONLY_PLUGIN")  # optional: run just one
    if only_plugin:
        _blade_runner.load_plugins(only=[only_plugin])
    else:
        _blade_runner.load_plugins()

    # Set _cli to the active plugin instance for backward compat
    # (existing endpoints reference _cli.semantic, _cli.evolution, etc.)
    active_mode = _blade_runner.core_config.get("product", {}).get("mode")
    if active_mode and active_mode in _blade_runner.plugins:
        _cli = _blade_runner.plugins[active_mode]
    elif _blade_runner.plugins:
        _cli = next(iter(_blade_runner.plugins.values()))
    else:
        # Fallback: create a minimal PersistenceCLI for backward compat
        from interface.cli import PersistenceCLI
        _cli = PersistenceCLI()
        _cli._web_mode = True
        _cli.wake()

    # Make _cli.inference and _cli.config accessible for endpoints
    if hasattr(_cli, 'router') and _cli.router:
        _cli.inference = _cli.router

    # CRITICAL: wire each plugin's endpoints to its OWN plugin instance
    # Without this, all plugins share _cli and read from the wrong database
    for plugin_name, plugin_instance in _blade_runner.plugins.items():
        try:
            # Find the plugin's endpoints module and set _get_engine to return
            # THIS plugin's instance, not the global _cli
            ep_module = None
            try:
                ep_module = __import__(f"plugins.{plugin_name}.endpoints", fromlist=["_get_engine"])
            except ImportError:
                pass

            if ep_module and hasattr(ep_module, "_get_engine"):
                ep_module._get_engine = lambda pi=plugin_instance: pi
                logger.info(f"Wired endpoints for '{plugin_name}' to its own instance")

            if ep_module and hasattr(ep_module, "_plugin_instance"):
                ep_module._plugin_instance = plugin_instance.plugin
        except Exception as e:
            logger.warning(f"Failed to wire endpoints for {plugin_name}: {e}")

    logger.info(f"ANIMA started — {len(_blade_runner.plugins)} plugins loaded: "
                f"{list(_blade_runner.plugins.keys())}")

    yield

    if _blade_runner:
        _blade_runner.shutdown()
    _executor.shutdown(wait=False)
    logger.info("ANIMA shut down")


app = FastAPI(title="ANIMA", lifespan=lifespan)

# CORS — allow dashboard (port 8800) to fetch from DMS instances
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register plugin endpoints (routes reference _cli global, populated in lifespan)
_plugin_loader.register_all_endpoints(app, lambda: _cli)


@app.get("/plugins")
async def list_plugins():
    """List all discovered plugins and their status."""
    active_mode = None
    if _cli:
        active_mode = _cli.config.get("product", {}).get("mode")
    plugins = []
    for p in _plugin_loader.all():
        port = p.manifest.get("plugin", {}).get("port", "")
        plugins.append({
            "name": p.name,
            "version": p.version,
            "mode": p.mode,
            "port": port,
            "plugin_dir": p.plugin_dir,
            "description": p.manifest.get("plugin", {}).get("description", ""),
            "active": active_mode == p.mode if active_mode else False,
        })
    # Enrich with blade runner status
    if _blade_runner:
        for p in plugins:
            blade = _blade_runner.plugins.get(p["name"])
            if blade:
                p["active"] = True
                p["beliefs"] = blade.status.get("beliefs", 0)
                p["edges"] = blade.status.get("edges", 0)
                p["running"] = blade.status.get("running", False)
    return {"plugins": plugins, "active_mode": active_mode}


# ── Models ──────────────────────────────────────────────────────────

@app.get("/models")
async def list_models():
    """List all configured models with status."""
    router = _blade_runner.router if _blade_runner else None
    if not router:
        if _cli and hasattr(_cli, "inference"):
            router = _cli.inference
        else:
            return {"models": []}
    if not hasattr(router, "models"):
        return {"models": []}
    models = []
    for key, info in router.models.items():
        in_flight = 0
        if hasattr(router, "_in_flight"):
            in_flight = router._in_flight.get(key, 0)
        models.append({
            "key": key,
            "name": getattr(info, "name", key),
            "endpoint": getattr(info, "endpoint", ""),
            "tier": getattr(info, "tier", "unknown"),
            "backend": getattr(info, "backend", "unknown"),
            "online": getattr(info, "online", False),
            "enabled": getattr(info, "enabled", True),
            "context_window": getattr(info, "context_window", 0),
            "tokens_per_second": getattr(info, "tokens_per_second", 0),
            "in_flight": in_flight,
            "profile": info.get_profile() if hasattr(info, 'get_profile') else {},
            "claude_model": getattr(info, "claude_model", None),
        })
    return {"models": models}


# ── Blade Runner Control ─────────────────────────────────────────

@app.get("/blades/status")
async def blade_status():
    """Get blade runner status — all plugins, cycling state."""
    if not _blade_runner:
        return {"error": "Blade runner not initialized"}
    return _blade_runner.status

    # Blade cycling endpoints removed. Plugins control themselves.
    # ANIMA provides routing and memory infrastructure only.

@app.post("/blades/{plugin_name}/activate")
async def blade_activate(plugin_name: str):
    """Switch the active plugin — existing endpoints will use this plugin's data."""
    global _cli
    if not _blade_runner:
        return JSONResponse({"error": "Not initialized"}, status_code=503)
    plugin = _blade_runner.plugins.get(plugin_name)
    if not plugin:
        return JSONResponse({"error": f"Plugin '{plugin_name}' not found"}, status_code=404)
    _cli = plugin
    if hasattr(_cli, 'router') and _cli.router:
        _cli.inference = _cli.router
    return {"ok": True, "active": plugin_name}

@app.get("/blades/active")
async def blade_active():
    """Get the currently active plugin name."""
    if _cli and hasattr(_cli, 'name'):
        return {"active": _cli.name}
    return {"active": None}


class ChatRequest(BaseModel):
    message: str


# ── Read-only query (no storage, no side effects) ──────────────────

def _build_citations(context, db):
    """Build citation list from retrieved beliefs — deterministic, server-side.

    Looks up which beliefs were used in the retrieval context, traces each
    back to its source document via belief_sources_documents → document_ledger.
    Also checks discovery_ledger for web-sourced beliefs.

    Returns list of dicts: [{filename, source_type, belief_count}, ...]
    """
    sources = {}  # filename → {source_type, belief_count}

    # Get belief IDs from semantic context parts
    belief_ids = []
    for part in context.get("context_parts", []):
        if part.get("type") == "semantic":
            bid = part.get("metadata", {}).get("id")
            if bid:
                belief_ids.append(bid)

    if not belief_ids:
        return []

    try:
        placeholders = ",".join("?" for _ in belief_ids)

        # Corpus documents
        rows = db.execute(
            f"""SELECT DISTINCT dl.filename, COALESCE(bsd.source_type, 'corpus') as src_type,
                       COUNT(DISTINCT bsd.belief_id) as belief_count
                FROM belief_sources_documents bsd
                JOIN document_ledger dl ON bsd.document_sha = dl.sha256
                WHERE bsd.belief_id IN ({placeholders})
                GROUP BY dl.filename, src_type
                ORDER BY belief_count DESC""",
            belief_ids,
        ).fetchall()

        for r in rows:
            fname = r["filename"]
            if fname not in sources:
                sources[fname] = {
                    "filename": fname,
                    "source_type": r["src_type"],
                    "belief_count": r["belief_count"],
                }
            else:
                sources[fname]["belief_count"] += r["belief_count"]

        # Web discoveries
        try:
            disc_rows = db.execute(
                f"""SELECT DISTINCT disc.title, disc.url, disc.source_quality,
                           COUNT(DISTINCT bsd.belief_id) as belief_count
                    FROM belief_sources_documents bsd
                    JOIN discovery_ledger disc ON bsd.document_sha = disc.id
                    WHERE bsd.belief_id IN ({placeholders})
                    GROUP BY disc.id
                    ORDER BY belief_count DESC""",
                belief_ids,
            ).fetchall()
            for r in disc_rows:
                key = r["url"] or r["title"]
                if key not in sources:
                    sources[key] = {
                        "filename": r["title"] or r["url"],
                        "url": r["url"],
                        "source_type": "web_discovery",
                        "source_quality": r["source_quality"],
                        "belief_count": r["belief_count"],
                    }
        except Exception:
            pass  # discovery_ledger may not have entries yet

    except Exception:
        pass

    return list(sources.values())


def _is_meta_query(query):
    """Detect meta-questions about the graph's own knowledge state."""
    q = query.lower()
    _META_PATTERNS = [
        "weakly supported", "most likely wrong", "lack evidence",
        "should be rejected", "missing from", "underexplored",
        "what fails", "what's weak", "what is weak", "where is the graph",
        "which conclusions", "what parts", "where does the system",
        "what hypotheses should", "likely ineffective",
        "contradictions in", "competing mechanisms",
        "what important", "what knowledge is required",
    ]
    return any(p in q for p in _META_PATTERNS)


def _is_strategic_query(query):
    """Detect open-ended strategic questions that need multi-path output."""
    q = query.lower()
    _STRATEGIC_PATTERNS = [
        "best way", "best approach", "best strategy",
        "how would you", "how should", "how could",
        "what would you recommend", "what strategy",
        "what intervention", "what combination",
        "design a", "propose a", "suggest a",
        "what non-obvious", "what novel",
        "way forward", "most promising",
        "optimal", "most effective",
    ]
    return any(p in q for p in _STRATEGIC_PATTERNS)


def _build_research_context(query, cli):
    """Build three-section knowledge context for research queries.

    POSITIVE: confirmed beliefs + hypotheses relevant to query
    NEGATIVE: rejected hypotheses with reasons, constraints, failed investigations
    UNCERTAIN: parked hypotheses, open investigation gaps

    For meta-queries about the graph itself, force-inject top items
    regardless of similarity score + include summary statistics.

    Returns formatted context string, or empty string if no research data.
    """
    try:
        db = cli.evolution.semantic.db_conn
        embeddings = cli.embeddings
        if not embeddings:
            return ""

        is_meta = _is_meta_query(query)

        import numpy as np

        query_emb = embeddings.embed(query)

        # --- Tree-scoped retrieval ---
        # Map query to relevant branches, pull beliefs from those first,
        # then tag-boost from other branches. Faster + more focused than flat search.
        _tree_scoped_ids = set()
        _tag_boosted_ids = set()
        try:
            # Find top L2/L3 branches by embedding similarity
            tree_nodes = db.execute(
                "SELECT id, name, layer, embedding FROM knowledge_tree WHERE layer >= 2"
            ).fetchall()
            branch_scores = []
            for node in tree_nodes:
                if node["embedding"]:
                    node_emb = np.frombuffer(node["embedding"], dtype=np.float32)
                    norm = np.linalg.norm(query_emb) * np.linalg.norm(node_emb)
                    sim = float(np.dot(query_emb, node_emb) / norm) if norm > 0 else 0.0
                    branch_scores.append((node["id"], sim))
            branch_scores.sort(key=lambda x: -x[1])
            top_branches = [bid for bid, sim in branch_scores[:5] if sim > 0.15]

            if top_branches:
                # Pull belief IDs from matched branches (via tree_paths)
                for branch_id in top_branches:
                    rows = db.execute(
                        "SELECT id FROM beliefs WHERE tree_paths LIKE ? "
                        "AND COALESCE(deprecated,0)=0 AND belief_status='active'",
                        (f'%"{branch_id}"%',),
                    ).fetchall()
                    for r in rows:
                        _tree_scoped_ids.add(r["id"])

                # Tag boost — find query-relevant tags, pull beliefs from other branches
                try:
                    tag_rows = db.execute("SELECT id, name FROM tag_registry WHERE belief_count > 0").fetchall()
                    query_tags = []
                    for t in tag_rows:
                        t_emb = embeddings.embed(t["name"])
                        norm = np.linalg.norm(query_emb) * np.linalg.norm(t_emb)
                        sim = float(np.dot(query_emb, t_emb) / norm) if norm > 0 else 0.0
                        if sim > 0.45:
                            query_tags.append(t["id"])

                    if query_tags:
                        placeholders = ",".join("?" * len(query_tags))
                        tag_beliefs = db.execute(
                            f"SELECT DISTINCT belief_id FROM belief_tags "
                            f"WHERE tag_id IN ({placeholders})",
                            query_tags,
                        ).fetchall()
                        for r in tag_beliefs:
                            _tag_boosted_ids.add(r["belief_id"])
                except Exception:
                    pass
        except Exception:
            pass  # tree not available — fall back to flat search

        def _rank_by_similarity(statements_with_meta, limit=10, force_top=0):
            """Rank items by cosine similarity to query + tree/tag boost.

            Tree-scoped beliefs get +0.10 boost.
            Tag-matched beliefs get +0.05 boost per matching tag.
            """
            scored = []
            for item in statements_with_meta:
                stmt = item.get("statement", "")
                if not stmt:
                    continue
                try:
                    emb = embeddings.embed(stmt)
                    norm = np.linalg.norm(query_emb) * np.linalg.norm(emb)
                    sim = float(np.dot(query_emb, emb) / norm) if norm > 0 else 0.0
                    # Tree scope boost
                    bid = item.get("_belief_id", "")
                    if bid and bid in _tree_scoped_ids:
                        sim += 0.10
                    if bid and bid in _tag_boosted_ids:
                        sim += 0.05
                    item["_sim"] = sim
                    scored.append(item)
                except Exception:
                    continue
            scored.sort(key=lambda x: x["_sim"], reverse=True)
            if force_top > 0:
                return scored[:max(limit, force_top)]
            # Normal mode: filter by threshold
            return [s for s in scored[:limit] if s["_sim"] >= 0.35]

        # For meta-queries, force-inject top items regardless of similarity
        force_n = 5 if is_meta else 0

        sections = []

        # --- POSITIVE KNOWLEDGE ---
        positive_items = []

        # Confirmed hypotheses
        try:
            confirmed = db.execute(
                "SELECT statement, evidence_confidence FROM hypothesis_queue "
                "WHERE status = 'confirmed' LIMIT 20"
            ).fetchall()
            for r in confirmed:
                positive_items.append({
                    "statement": r["statement"],
                    "label": f"[CONFIRMED hyp, conf={r['evidence_confidence']:.2f}]",
                })
        except Exception:
            pass

        # Evidence-ready hypotheses
        try:
            ev_ready = db.execute(
                "SELECT statement, evidence_confidence FROM hypothesis_queue "
                "WHERE status = 'evidence_ready' LIMIT 20"
            ).fetchall()
            for r in ev_ready:
                positive_items.append({
                    "statement": r["statement"],
                    "label": f"[EVIDENCE-READY hyp, conf={r['evidence_confidence']:.2f}]",
                })
        except Exception:
            pass

        ranked_positive = _rank_by_similarity(positive_items, limit=10, force_top=force_n)
        if ranked_positive:
            lines = ["", "[POSITIVE KNOWLEDGE — confirmed and evidence-ready hypotheses:]"]
            for item in ranked_positive:
                lines.append(f"  {item['label']} {item['statement']}")
            sections.append("\n".join(lines))

        # --- NEGATIVE KNOWLEDGE ---
        negative_items = []

        # Rejected hypotheses — grouped by reason type
        try:
            rejected = db.execute(
                "SELECT statement, user_decision, evidence_confidence FROM hypothesis_queue "
                "WHERE status = 'rejected' AND user_decision IS NOT NULL LIMIT 30"
            ).fetchall()
            for r in rejected:
                reason = r["user_decision"] or "rejected"
                reason_lower = reason.lower()
                # Classify rejection reason
                if "invalid" in reason_lower or "auto-rejected" in reason_lower:
                    group = "INVALID"
                elif "insufficient" in reason_lower or "parked" in reason_lower:
                    group = "INSUFFICIENT EVIDENCE"
                elif "duplicate" in reason_lower or "dedup" in reason_lower:
                    group = "DUPLICATE"
                elif "vague" in reason_lower:
                    group = "TOO VAGUE"
                else:
                    group = "REJECTED"
                negative_items.append({
                    "statement": r["statement"],
                    "label": f"[{group}: {reason}]",
                    "_group": group,
                })
        except Exception:
            pass

        # Constraints (anti-beliefs from revocations)
        try:
            constraints = db.execute(
                "SELECT reason FROM hypothesis_constraints LIMIT 15"
            ).fetchall()
            for r in constraints:
                negative_items.append({
                    "statement": r["reason"],
                    "label": "[REVOKED — explicitly invalidated:]",
                    "_group": "REVOKED",
                })
        except Exception:
            pass

        ranked_negative = _rank_by_similarity(negative_items, limit=10, force_top=force_n)
        if ranked_negative:
            lines = ["", "[NEGATIVE KNOWLEDGE — grouped by rejection reason. "
                     "'INVALID' = proven wrong. 'INSUFFICIENT EVIDENCE' = not yet proven. "
                     "'REVOKED' = explicitly invalidated by operator:]"]
            # Group the output
            groups = {}
            for item in ranked_negative:
                g = item.get("_group", "REJECTED")
                groups.setdefault(g, []).append(item)
            for g in ["INVALID", "REVOKED", "INSUFFICIENT EVIDENCE", "TOO VAGUE", "DUPLICATE", "REJECTED"]:
                if g in groups:
                    lines.append(f"  ── {g} ──")
                    for item in groups[g]:
                        lines.append(f"    {item['statement']}")
            sections.append("\n".join(lines))

        # --- UNCERTAIN ---
        uncertain_items = []

        # Parked hypotheses
        try:
            parked = db.execute(
                "SELECT statement, evidence_confidence, user_decision FROM hypothesis_queue "
                "WHERE status = 'parked' LIMIT 30"
            ).fetchall()
            for r in parked:
                conf = r["evidence_confidence"] or 0.5
                reason = r["user_decision"] or ""
                uncertain_items.append({
                    "statement": r["statement"],
                    "label": f"[PARKED conf={conf:.2f}] {reason}",
                })
        except Exception:
            pass

        # Low-priority hypotheses
        try:
            low_pri = db.execute(
                "SELECT statement, evidence_confidence FROM hypothesis_queue "
                "WHERE status = 'low_priority' LIMIT 15"
            ).fetchall()
            for r in low_pri:
                uncertain_items.append({
                    "statement": r["statement"],
                    "label": f"[LOW-PRIORITY conf={r['evidence_confidence']:.2f}]",
                })
        except Exception:
            pass

        # Open investigation gaps
        try:
            curiosity = getattr(cli, "curiosity", None)
            if curiosity:
                gaps = curiosity.get_open_questions(limit=20)
                for g in gaps:
                    gd = dict(g)
                    uncertain_items.append({
                        "statement": gd.get("question", ""),
                        "label": f"[OPEN GAP, priority={gd.get('priority','?')}]",
                    })
        except Exception:
            pass

        ranked_uncertain = _rank_by_similarity(uncertain_items, limit=10, force_top=force_n)
        if ranked_uncertain:
            lines = ["", "[UNCERTAIN — parked hypotheses and open questions. "
                     "Evidence is incomplete, not conclusive:]"]
            for item in ranked_uncertain:
                lines.append(f"  {item['label']} {item['statement']}")
            sections.append("\n".join(lines))

        # For meta-queries, prepend summary statistics
        if is_meta and sections:
            try:
                stats_lines = ["\n[SYSTEM KNOWLEDGE SUMMARY:]"]
                for status in ["confirmed", "evidence_ready", "parked", "rejected", "low_priority"]:
                    cnt = db.execute(
                        "SELECT COUNT(*) FROM hypothesis_queue WHERE status = ?",
                        (status,),
                    ).fetchone()[0]
                    stats_lines.append(f"  {status}: {cnt} hypotheses")
                constraint_cnt = 0
                try:
                    constraint_cnt = db.execute(
                        "SELECT COUNT(*) FROM hypothesis_constraints"
                    ).fetchone()[0]
                except Exception:
                    pass
                stats_lines.append(f"  constraints (anti-beliefs): {constraint_cnt}")
                gap_cnt = 0
                try:
                    curiosity = getattr(cli, "curiosity", None)
                    if curiosity:
                        gap_cnt = len(curiosity.get_open_questions(limit=200))
                except Exception:
                    pass
                stats_lines.append(f"  open investigation gaps: {gap_cnt}")
                sections.insert(0, "\n".join(stats_lines))
            except Exception:
                pass

        return "\n".join(sections) if sections else ""

    except Exception:
        return ""


def _classify_query_type(query):
    """Classify query as recall, synthesis, or chained. Keyword heuristic, no LLM call.

    Chained: query references specific documents/dates AND asks for synthesis.
    Example: "What themes appear across Day 15-17 docs?" — needs doc lookup then synthesis.
    """
    import re as _re_cls
    q = query.lower().strip()

    synthesis_signals = [
        "relate" in q, "connect" in q, "in common" in q,
        "how does" in q, "what would" in q, "what could" in q,
        q.startswith("why "), "compare" in q,
        "relationship between" in q, "what connects" in q,
        "themes" in q, "summarize" in q, "evolution of" in q,
        "across" in q,
    ]
    recall_signals = [
        q.startswith("how many"), q.startswith("what is "),
        q.startswith("what was"), q.startswith("when did"),
        q.startswith("when was"), q.startswith("who "),
        q.startswith("which "), q.startswith("list "),
        "what is the" in q, "what does" in q,
    ]
    reference_signals = [
        bool(_re_cls.search(r'day\s*\d+', q)),
        bool(_re_cls.search(r'soak\s*\d+', q)),
        bool(_re_cls.search(r'document[s]?\s', q)),
        bool(_re_cls.search(r'\d+\s*(?:through|to|-)\s*\d+', q)),
        "doc " in q, "file " in q, ".md" in q,
    ]

    has_synthesis = any(synthesis_signals)
    has_reference = any(reference_signals)

    if has_synthesis and has_reference:
        return "chained"
    if has_synthesis:
        return "synthesis"
    if any(recall_signals):
        return "recall"
    if has_reference:
        return "recall"
    return "synthesis"


def _get_entity_grounding(query, entities):
    """Get entity search results as grounding context for queries.

    Includes fuzzy matching: filename search, date normalization,
    "Day N" → "day-N" alias, number range expansion.
    """
    import re as _re_ent
    if not _cli:
        return ""
    db = _cli.evolution.semantic.db_conn

    # Search terms: NER entities + significant words from query
    search_terms = list(entities) if entities else []
    search_terms.extend(
        w for w in query.lower().split()
        if len(w) > 3 and w not in ("what", "when", "where", "which", "does", "have", "that", "this", "from", "with", "been", "were", "they", "their", "about", "between", "through", "across")
    )

    # Alias expansion: "Day 15" → "day-15", "soak 28" → "soak 28"
    day_matches = _re_ent.findall(r'day\s*(\d+)', query, _re_ent.IGNORECASE)
    for d in day_matches:
        search_terms.append(f"day-{d}")
        search_terms.append(f"2026-03-{int(d)+2:02d}")  # approximate date mapping

    # Range expansion: "15 through 17" or "15-17" → individual values
    range_matches = _re_ent.findall(r'(\d+)\s*(?:through|to|-)\s*(\d+)', query, _re_ent.IGNORECASE)
    for start, end in range_matches:
        for n in range(int(start), int(end) + 1):
            search_terms.append(f"day-{n}")
            search_terms.append(str(n))

    if not search_terms:
        return ""

    seen_docs = {}

    # Entity value search
    for term in search_terms[:12]:
        try:
            rows = db.execute(
                """SELECT DISTINCT de.document_sha, dl.filename, de.entity_type,
                          de.entity_value, dl.belief_count
                   FROM document_entities de
                   JOIN document_ledger dl ON de.document_sha = dl.sha256
                   WHERE LOWER(de.entity_value) LIKE ?
                   LIMIT 5""",
                (f"%{term.lower()}%",),
            ).fetchall()
            for row in rows:
                sha = row["document_sha"]
                if sha not in seen_docs:
                    seen_docs[sha] = {
                        "filename": row["filename"],
                        "belief_count": row["belief_count"] or 0,
                        "entities": [],
                    }
                seen_docs[sha]["entities"].append(
                    f"{row['entity_type']}:{row['entity_value']}"
                )
        except Exception:
            continue

    # Filename search — catches cases where entity values don't match but filenames do
    for term in search_terms[:8]:
        try:
            rows = db.execute(
                """SELECT sha256, filename, belief_count
                   FROM document_ledger
                   WHERE LOWER(filename) LIKE ? AND status = 'ingested'
                   LIMIT 5""",
                (f"%{term.lower()}%",),
            ).fetchall()
            for row in rows:
                sha = row["sha256"]
                if sha not in seen_docs:
                    seen_docs[sha] = {
                        "filename": row["filename"],
                        "belief_count": row["belief_count"] or 0,
                        "entities": ["filename_match"],
                    }
        except Exception:
            continue

    if not seen_docs:
        return ""

    lines = []
    for sha, info in list(seen_docs.items())[:10]:
        ents = ", ".join(info["entities"][:4])
        lines.append(f"- {info['filename']} ({ents}) [{info['belief_count']} beliefs]")

    return "\n".join(lines)


def _get_document_beliefs(query, entities):
    """Resolve documents from query references, return their beliefs as context.

    For chained queries: finds docs by entity/filename match, then pulls
    all beliefs sourced from those documents. Returns formatted context.
    """
    import re as _re_db
    if not _cli:
        return "", []
    db = _cli.evolution.semantic.db_conn

    # Find matching document SHAs via entity grounding logic
    search_terms = list(entities) if entities else []
    day_matches = _re_db.findall(r'day\s*(\d+)', query, _re_db.IGNORECASE)
    for d in day_matches:
        search_terms.append(f"day-{d}")
    range_matches = _re_db.findall(r'(\d+)\s*(?:through|to|-)\s*(\d+)', query, _re_db.IGNORECASE)
    for start, end in range_matches:
        for n in range(int(start), int(end) + 1):
            search_terms.append(f"day-{n}")
    search_terms.extend(
        w for w in query.lower().split()
        if len(w) > 3 and w not in ("what", "when", "where", "which", "does", "have", "that", "this", "from", "with", "been", "were", "they", "their", "about", "between", "through", "across", "themes", "appear", "documents", "show")
    )

    matched_shas = set()
    for term in search_terms[:12]:
        try:
            for row in db.execute(
                "SELECT sha256 FROM document_ledger WHERE LOWER(filename) LIKE ? AND status='ingested' LIMIT 5",
                (f"%{term.lower()}%",),
            ).fetchall():
                matched_shas.add(row["sha256"])
            for row in db.execute(
                """SELECT DISTINCT document_sha FROM document_entities
                   WHERE LOWER(entity_value) LIKE ? LIMIT 5""",
                (f"%{term.lower()}%",),
            ).fetchall():
                matched_shas.add(row["document_sha"])
        except Exception:
            continue

    if not matched_shas:
        return "", []

    # Pull beliefs from matched documents
    placeholders = ",".join("?" * len(matched_shas))
    sha_list = list(matched_shas)
    belief_rows = db.execute(
        f"""SELECT DISTINCT b.statement, b.confidence, dl.filename
            FROM belief_sources_documents bsd
            JOIN beliefs b ON bsd.belief_id = b.id
            JOIN document_ledger dl ON bsd.document_sha = dl.sha256
            WHERE bsd.document_sha IN ({placeholders})
            AND COALESCE(b.deprecated, 0) = 0
            ORDER BY b.confidence DESC LIMIT 30""",
        sha_list,
    ).fetchall()

    # Also get document names for context
    doc_rows = db.execute(
        f"SELECT filename, belief_count FROM document_ledger WHERE sha256 IN ({placeholders})",
        sha_list,
    ).fetchall()

    lines = ["[DOCUMENTS MATCHING YOUR QUERY:]"]
    for doc in doc_rows:
        lines.append(f"- {doc['filename']} ({doc['belief_count'] or 0} beliefs)")

    if belief_rows:
        lines.append("")
        lines.append("[BELIEFS FROM THESE DOCUMENTS:]")
        for b in belief_rows:
            lines.append(f"- {b['statement']} (confidence: {b['confidence']:.1f}, source: {b['filename']})")

    # Get entities from matched docs for additional context
    entity_rows = db.execute(
        f"""SELECT entity_type, entity_value FROM document_entities
            WHERE document_sha IN ({placeholders})
            AND entity_type = 'topic'
            GROUP BY LOWER(entity_value) ORDER BY COUNT(*) DESC LIMIT 10""",
        sha_list,
    ).fetchall()
    if entity_rows:
        topics = [r["entity_value"] for r in entity_rows]
        lines.append(f"\n[KEY TOPICS across these documents: {', '.join(topics)}]")

    return "\n".join(lines), [b["statement"] for b in belief_rows]


@app.post("/query")
async def query(req: ChatRequest):
    """Query ANIMA's knowledge without storing anything.

    Read-only: no episodes, no turns, no consolidation triggers.
    Uses the same retrieval context and LLM as /chat but leaves
    the graph completely untouched.
    """
    if not _cli:
        return JSONResponse({"error": "ANIMA not initialized"}, status_code=400)

    import asyncio, time as _time
    loop = asyncio.get_event_loop()

    def _do_query():
        acquired = _chat_lock.acquire(timeout=5)
        if not acquired:
            return None
        try:
            import re as _re
            t0 = _time.monotonic()

            query_type = _classify_query_type(req.message)
            entities = _cli.ner.extract_names(req.message)

            # Both paths get belief graph context
            context = _cli.retrieval.reconstruct(
                req.message, current_episode_id=None
            )
            system_context = context["system_context"]

            # Strip tool use instructions — /query is direct answer, no tools
            system_context = _re.sub(
                r'TOOLS:.*?(?=\n\[|\nEXAMPLES|\nWRONG|\Z)',
                'You are answering a direct question. Do not use tools. Answer from your beliefs and knowledge only.\n'
                'The correct answer is the one that most accurately reflects the available evidence, '
                'including identifying lack of evidence, contradictions, and uncertainty. '
                'Clear negative or inconclusive findings are preferred over speculative or optimistic conclusions. '
                'When answering, explicitly distinguish between: supported by evidence, '
                'inconclusive or limited evidence, and not supported by available evidence. '
                'If multiple plausible interpretations exist, present them rather than collapsing to a single conclusion.\n',
                system_context,
                flags=_re.DOTALL,
            )

            # Entity grounding — both paths get this
            entity_context = _get_entity_grounding(req.message, entities)

            # Three-section knowledge context (positive/negative/uncertain)
            research_context = _build_research_context(req.message, _cli)
            if research_context:
                system_context += research_context

            # Strategic query detection — force multi-path structured output
            if _is_strategic_query(req.message):
                system_context += (
                    "\n\n[STRATEGIC QUERY — MULTI-PATH OUTPUT REQUIRED]\n"
                    "This is an open-ended strategic question. You MUST structure your response as:\n\n"
                    "STRONGLY SUPPORTED:\n"
                    "Approaches with the most graph evidence. Cite specific beliefs.\n\n"
                    "EMERGING / MODERATELY SUPPORTED:\n"
                    "Alternative paths with partial evidence. Note evidence gaps.\n\n"
                    "SPECULATIVE:\n"
                    "Paths with weak or synthesis-only support. Flag as speculative.\n\n"
                    "WHAT COULD INVALIDATE THIS:\n"
                    "Contradicting evidence, known failures, gaps that could collapse the above.\n"
                    "Use NEGATIVE KNOWLEDGE and UNCERTAIN sections above.\n\n"
                    "DO NOT present a single 'best' answer. Present competing paths with "
                    "different evidence profiles. The researcher needs to see alternatives "
                    "and uncertainty, not a confident recommendation.\n"
                )

            if query_type == "chained":
                # Chained: resolve documents first, then synthesize from their beliefs
                doc_context, doc_beliefs = _get_document_beliefs(req.message, entities)
                if doc_context:
                    system_context += (
                        "\n\n[QUERY TYPE: DOCUMENT-TARGETED SYNTHESIS — "
                        "the user is asking about specific documents. "
                        "Answer using the beliefs and content from the matched documents below. "
                        "Synthesize themes, patterns, and connections across them. "
                        "Cite source documents.]\n\n" + doc_context
                    )
                if entity_context:
                    system_context += "\n\n[ADDITIONAL REFERENCES:]\n" + entity_context

            elif query_type == "recall":
                # Recall: entity context is primary, beliefs are supplementary
                system_context += (
                    "\n\n[QUERY TYPE: FACTUAL RECALL — answer this specific question. "
                    "Use document references and entity data below as your primary source. "
                    "Be specific. If you find the answer, cite the source document.]\n"
                )
                if entity_context:
                    system_context += "\n" + entity_context
            else:
                # Synthesis: beliefs are primary, entity context is supplementary grounding
                if entity_context:
                    system_context += (
                        "\n\n[REFERENCE FACTS from document index — "
                        "use as grounding if relevant:]\n" + entity_context
                    )

            messages = [{"role": "system", "content": system_context}]
            messages.append({"role": "user", "content": req.message + " /no_think"})
            response = _cli.inference.generate_with_messages(messages, max_tokens=1024, task="chat")
            response_text = _re.sub(r'<think>.*?</think>', '', response.strip(), flags=_re.DOTALL)
            response_text = response_text.replace('/no_think', '').strip()
            latency_ms = int((_time.monotonic() - t0) * 1000)

            # Server-side citation injection — deterministic, not LLM-dependent.
            # Match retrieved beliefs back to source documents via provenance.
            sources = _build_citations(context, _cli.evolution.semantic.db_conn)

            return {
                "response": response_text,
                "latency_ms": latency_ms,
                "query_type": query_type,
                "sources": sources,
                "retrieval_metadata": context.get("retrieval_metadata", {}),
            }
        finally:
            _chat_lock.release()

    result = await loop.run_in_executor(_executor, _do_query)
    if result is None:
        return JSONResponse(
            {"error": "System busy — lifecycle cycle in progress, try again shortly"},
            status_code=503,
        )
    return JSONResponse(result)


# ── Dispatch — raw model access through the router ────────────────

class DispatchRequest(BaseModel):
    prompt: str
    system_prompt: str = ""
    task: str = "extraction"
    max_tokens: int = 4096
    temperature: float = 0.1
    include_context: bool = False
    context_query: str = ""


@app.post("/dispatch")
async def dispatch(req: DispatchRequest):
    """Dispatch a prompt to the router with optional graph context.

    The router picks the model based on task type (defined in plugin config).
    If include_context=true, relevant beliefs from the knowledge graph are
    injected into the system prompt before dispatch.

    Body: {
        "prompt": "Write the storage module...",
        "system_prompt": "You are an engineer...",
        "task": "code_generation",
        "max_tokens": 4096,
        "temperature": 0.1,
        "include_context": true,                     (optional, default false)
        "context_query": "storage layer requirements" (optional, uses prompt if empty)
    }
    """
    import asyncio
    if not _cli:
        return JSONResponse({"error": "ANIMA not initialized"}, status_code=400)

    loop = asyncio.get_event_loop()

    def _do():
        system_parts = []
        if req.system_prompt:
            system_parts.append(req.system_prompt)

        # Optional: pull relevant beliefs from graph as context
        beliefs_injected = 0
        if req.include_context:
            try:
                context_text = req.context_query or req.prompt
                db = _cli.evolution.semantic.db_conn
                embeddings = _cli.embeddings
                import numpy as np

                query_emb = embeddings.embed(context_text)

                # Get active beliefs, rank by similarity
                rows = db.execute(
                    "SELECT id, statement, confidence, tree_paths FROM beliefs "
                    "WHERE COALESCE(deprecated,0)=0 AND belief_status='active' "
                    "AND valid_to IS NULL"
                ).fetchall()

                scored = []
                for r in rows:
                    try:
                        b_emb = embeddings.embed(r["statement"])
                        norm = np.linalg.norm(query_emb) * np.linalg.norm(b_emb)
                        sim = float(np.dot(query_emb, b_emb) / norm) if norm > 0 else 0.0
                        if sim >= 0.35:
                            scored.append((sim, r))
                    except Exception:
                        continue

                scored.sort(key=lambda x: -x[0])
                top = scored[:15]

                if top:
                    context_lines = ["[RELEVANT PROJECT KNOWLEDGE — use as context:]"]
                    for sim, r in top:
                        conf = r["confidence"] or 0.5
                        context_lines.append(f"  [{conf:.2f}] {r['statement']}")
                    system_parts.append("\n".join(context_lines))
                    beliefs_injected = len(top)
            except Exception:
                pass

        messages = []
        if system_parts:
            messages.append({"role": "system", "content": "\n\n".join(system_parts)})
        messages.append({"role": "user", "content": req.prompt})

        import time as _t
        start = _t.monotonic()
        response = _cli.inference.generate_with_messages(
            messages, max_tokens=req.max_tokens,
            temperature=req.temperature, task=req.task,
        )
        elapsed_ms = int((_t.monotonic() - start) * 1000)

        # Clean artifacts
        import re as _re
        response = _re.sub(r'<think>.*?</think>', '', response.strip(), flags=_re.DOTALL)
        response = _re.sub(r'</?answer>', '', response).strip()
        response = response.replace('/no_think', '').strip()

        return {
            "response": response,
            "task": req.task,
            "latency_ms": elapsed_ms,
            "beliefs_injected": beliefs_injected,
        }

    try:
        result = await loop.run_in_executor(_executor, _do)
        return JSONResponse(result)
    except Exception as e:
        logger.error(f"Dispatch failed: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Priority Queue — async task submission + parallel dispatch ──────

class QueueSubmitRequest(BaseModel):
    prompt: str
    system_prompt: str = ""
    task: str = "extraction"
    priority: int = 2          # 0=critical, 1=high, 2=normal, 3=bulk
    max_tokens: int = 4096
    temperature: float = 0.1
    include_context: bool = False
    context_query: str = ""
    metadata: dict = {}


def _get_queue():
    """Lazy init the task queue."""
    global _task_queue
    if _task_queue is None:
        from core.queue import TaskQueue
        router = getattr(_cli, "router", None) or getattr(_cli, "inference", None)

        # Context provider — queries beliefs by embedding similarity
        def _context_provider(query_text):
            try:
                import numpy as np
                db = _cli.evolution.semantic.db_conn
                embeddings = _cli.embeddings
                query_emb = embeddings.embed(query_text)
                rows = db.execute(
                    "SELECT statement, confidence FROM beliefs "
                    "WHERE COALESCE(deprecated,0)=0 AND belief_status='active' "
                    "AND valid_to IS NULL"
                ).fetchall()
                scored = []
                for r in rows:
                    try:
                        b_emb = embeddings.embed(r["statement"])
                        norm = np.linalg.norm(query_emb) * np.linalg.norm(b_emb)
                        sim = float(np.dot(query_emb, b_emb) / norm) if norm > 0 else 0.0
                        if sim >= 0.35:
                            scored.append((sim, f"[{r['confidence']:.2f}] {r['statement']}"))
                    except Exception:
                        continue
                scored.sort(key=lambda x: -x[0])
                return [s[1] for s in scored[:15]]
            except Exception:
                return []

        _task_queue = TaskQueue(router, context_provider=_context_provider)
    return _task_queue


@app.post("/queue/submit")
async def queue_submit(req: QueueSubmitRequest):
    """Submit a task to the priority queue. Returns task_id instantly.

    Priority: 0=critical, 1=high, 2=normal, 3=bulk.
    High priority tasks are processed before low priority.
    Workers dispatch across all available models in parallel.
    """
    if not _cli:
        return JSONResponse({"error": "ANIMA not initialized"}, status_code=400)

    q = _get_queue()
    task_id = q.submit(
        prompt=req.prompt,
        system_prompt=req.system_prompt,
        task=req.task,
        priority=req.priority,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        include_context=req.include_context,
        context_query=req.context_query,
        metadata=req.metadata,
    )
    return JSONResponse({"task_id": task_id, "priority": req.priority})


@app.get("/queue/status")
async def queue_status():
    """Queue health: pending tasks, active workers, completion stats."""
    # Prefer scheduler status if available
    if _blade_runner and _blade_runner.scheduler:
        return JSONResponse(_blade_runner.scheduler.get_status())
    if not _cli:
        return JSONResponse({"error": "ANIMA not initialized"}, status_code=400)
    q = _get_queue()
    return JSONResponse(q.get_status())


@app.get("/queue/{task_id}")
async def queue_result(task_id: str):
    """Poll for task result. Returns status + response when complete."""
    if not _cli:
        return JSONResponse({"error": "ANIMA not initialized"}, status_code=400)

    q = _get_queue()
    result = q.get_result(task_id)
    return JSONResponse(result)


@app.post("/queue/cancel/{task_id}")
async def queue_cancel(task_id: str):
    """Cancel a pending task. Cannot cancel running tasks."""
    if not _cli:
        return JSONResponse({"error": "ANIMA not initialized"}, status_code=400)

    q = _get_queue()
    cancelled = q.cancel(task_id)
    return JSONResponse({"cancelled": cancelled, "task_id": task_id})


# ── Chat ────────────────────────────────────────────────────────────

@app.post("/chat")
async def chat(req: ChatRequest):
    """Send a message and get ANIMA's response. Serialized via lock."""
    if not _cli or _cli._sleeping:
        return JSONResponse(
            {"error": "ANIMA is sleeping"}, status_code=400
        )

    import asyncio
    loop = asyncio.get_event_loop()

    def _do_chat():
        with _chat_lock:
            result = _cli._process_turn(req.message, capture_output=True)
        return result

    result = await loop.run_in_executor(_executor, _do_chat)

    if isinstance(result, dict):
        return JSONResponse({
            "response": result["response"],
            "turn_count": _cli.turn_count,
            "notifications": result.get("notifications", []),
        })
    # Fallback — _process_turn returned bool (shouldn't happen with capture_output)
    return JSONResponse({
        "response": "(no response)",
        "turn_count": _cli.turn_count,
        "notifications": [],
    })


# ── Sleep ──────────────────────────────────────────────────────────

@app.post("/sleep")
async def sleep():
    """Put ANIMA to sleep — full consolidation, dreams, belief processing."""
    import asyncio
    loop = asyncio.get_event_loop()

    if not _cli or _cli._sleeping:
        return JSONResponse({"status": "already_sleeping"})

    def _do_sleep():
        with _chat_lock:
            _cli.web_sleep()

    await loop.run_in_executor(_executor, _do_sleep)
    return JSONResponse({"status": "sleeping"})


@app.get("/status")
async def status():
    """Check if ANIMA is awake or sleeping."""
    if not _cli:
        return JSONResponse({"awake": False})
    return JSONResponse({"awake": not _cli._sleeping})


# ── Free Time ─────────────────────────────────────────────────────

@app.post("/freetime")
async def freetime():
    """Trigger an exploration window session."""
    import asyncio
    loop = asyncio.get_event_loop()

    if not _cli or _cli._sleeping:
        return JSONResponse(
            {"error": "ANIMA is sleeping"}, status_code=400
        )

    if not _cli.exploration_engine:
        return JSONResponse(
            {"error": "Exploration engine not available"}, status_code=400
        )

    def _do_freetime():
        with _chat_lock:
            results = _cli.exploration_engine.explore_session(
                episode_id=_cli.current_episode_id
            )
            if not results:
                return {"response": None, "count": 0}

            # Build the system message (same as CLI report)
            parts = [
                f"[SYSTEM: You just completed an exploration window. You explored "
                f"{len(results)} topic(s). Below are your findings classified by "
                f"evidence strength. Report what you found and what remains "
                f"uncertain. Do NOT end with a question.]\n"
            ]
            for i, result in enumerate(results, 1):
                search_label = (
                    "web search" if result["search_used"]
                    else "your own reasoning"
                )
                priority = result.get("priority_ai", "normal")
                priority_tag = (
                    f" [{priority.upper()}]" if priority != "normal" else ""
                )
                parts.append(
                    f"--- Exploration {i} (via {search_label}){priority_tag} ---"
                )
                parts.append(f"Topic: {result['trigger_text']}")
                if result.get("confirmed"):
                    parts.append(f"Confirmed: {result['confirmed']}")
                if result.get("inferred"):
                    parts.append(f"Inferred: {result['inferred']}")
                if result.get("uncertain"):
                    parts.append(f"Uncertain: {result['uncertain']}")
                if result.get("open_questions"):
                    parts.append(
                        f"Open questions: "
                        f"{'; '.join(result['open_questions'][:3])}"
                    )
                parts.append("")
            turn_result = _cli._process_turn(
                "\n".join(parts), capture_output=True
            )
            return {
                "response": (
                    turn_result["response"]
                    if isinstance(turn_result, dict)
                    else None
                ),
                "count": len(results),
                "notifications": (
                    turn_result.get("notifications", [])
                    if isinstance(turn_result, dict)
                    else []
                ),
            }

    result = await loop.run_in_executor(_executor, _do_freetime)

    if result["response"] is None and result["count"] == 0:
        return JSONResponse({
            "response": None,
            "count": 0,
            "message": "Nothing to explore — no open gaps or topics.",
        })

    return JSONResponse({
        "response": result["response"],
        "turn_count": _cli.turn_count,
        "count": result["count"],
        "notifications": result.get("notifications", []),
    })


# ── History ─────────────────────────────────────────────────────────

@app.get("/history")
async def history():
    """Get current session conversation history."""
    turns = [
        {"role": role, "content": content}
        for role, content in (_cli.conversation_history if _cli else [])
    ]
    return JSONResponse({
        "turns": turns,
        "turn_count": _cli.turn_count if _cli else 0,
    })


# ── Dashboard endpoints (read-only DB connections) ──────────────────

def _query_state():
    """Latest state vector from telemetry.db."""
    try:
        conn = _ro_connect(_TELEMETRY_DB)
        row = conn.execute(
            "SELECT * FROM state_log ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        conn.close()
        return dict(row) if row else {}
    except Exception:
        return {}


def _query_beliefs():
    """Belief summary + recent beliefs."""
    try:
        conn = _ro_connect(_PERSISTENCE_DB)
        total = conn.execute("SELECT COUNT(*) FROM beliefs").fetchone()[0]
        avg_row = conn.execute("SELECT AVG(confidence) FROM beliefs").fetchone()
        avg_conf = round(avg_row[0], 3) if avg_row[0] is not None else 0.0

        dist = {}
        for label, lo, hi in [
            ("low", 0.0, 0.4), ("medium", 0.4, 0.7), ("high", 0.7, 1.01)
        ]:
            count = conn.execute(
                "SELECT COUNT(*) FROM beliefs WHERE confidence >= ? AND confidence < ?",
                (lo, hi),
            ).fetchone()[0]
            dist[label] = count

        recent = conn.execute(
            "SELECT id, statement, confidence, created_at "
            "FROM beliefs ORDER BY created_at DESC LIMIT 10"
        ).fetchall()
        conn.close()

        return {
            "total": total,
            "avg_confidence": avg_conf,
            "confidence_distribution": dist,
            "recent": [dict(r) for r in recent],
        }
    except Exception:
        return {"total": 0, "avg_confidence": 0.0, "confidence_distribution": {}, "recent": []}


def _query_explorations():
    """Exploration summary."""
    try:
        conn = _ro_connect(_PERSISTENCE_DB)
        total = conn.execute("SELECT COUNT(*) FROM explorations").fetchone()[0]
        pending = conn.execute(
            "SELECT COUNT(*) FROM explorations WHERE status='preliminary'"
        ).fetchone()[0]
        accepted = conn.execute(
            "SELECT COUNT(*) FROM explorations WHERE status='accepted'"
        ).fetchone()[0]
        rejected = conn.execute(
            "SELECT COUNT(*) FROM explorations WHERE status='rejected'"
        ).fetchone()[0]

        recent = conn.execute(
            "SELECT id, trigger_text, status, search_used, internal_confidence, created_at "
            "FROM explorations ORDER BY created_at DESC LIMIT 10"
        ).fetchall()
        conn.close()

        recent_list = []
        for r in recent:
            d = dict(r)
            d["search_used"] = bool(d.get("search_used", 0))
            recent_list.append(d)

        return {
            "total": total, "pending": pending,
            "accepted": accepted, "rejected": rejected,
            "recent": recent_list,
        }
    except Exception:
        return {"total": 0, "pending": 0, "accepted": 0, "rejected": 0, "recent": []}


def _query_gaps():
    """Knowledge gaps summary."""
    try:
        conn = _ro_connect(_PERSISTENCE_DB)
        open_count = conn.execute(
            "SELECT COUNT(*) FROM questions WHERE status='open'"
        ).fetchone()[0]
        total = conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
        resolved = conn.execute(
            "SELECT COUNT(*) FROM questions WHERE status IN ('answered','dissolved')"
        ).fetchone()[0]

        top_open = conn.execute(
            "SELECT id, question, priority, created_at FROM questions "
            "WHERE status='open' "
            "ORDER BY CASE priority "
            "WHEN 'high' THEN 0 WHEN 'medium' THEN 1 WHEN 'low' THEN 2 END, "
            "created_at ASC LIMIT 5"
        ).fetchall()
        conn.close()

        return {
            "open": open_count, "total": total, "resolved": resolved,
            "top_open": [dict(r) for r in top_open],
        }
    except Exception:
        return {"open": 0, "total": 0, "resolved": 0, "top_open": []}


def _query_sleep_history():
    """Recent consolidated episodes."""
    try:
        conn = _ro_connect(_PERSISTENCE_DB)
        rows = conn.execute(
            "SELECT id, timestamp, summary, turn_count, topics, key_insights "
            "FROM episodes WHERE summarized=1 "
            "ORDER BY timestamp DESC LIMIT 10"
        ).fetchall()
        conn.close()

        cycles = []
        for r in rows:
            d = dict(r)
            for field in ("topics", "key_insights"):
                val = d.get(field)
                if val:
                    try:
                        d[field] = json.loads(val)
                    except (json.JSONDecodeError, TypeError):
                        d[field] = []
                else:
                    d[field] = []
            cycles.append(d)
        return {"cycles": cycles}
    except Exception:
        return {"cycles": []}


def _query_contradictions():
    """Contradiction resolution data for dashboard."""
    try:
        if not _cli or not _cli.evolution:
            return {"resolved": [], "pending": [], "resolved_count": 0, "pending_count": 0}
        resolved = _cli.evolution.get_contradiction_resolutions(limit=20)
        pending = _cli.evolution.get_pending_contradictions()
        return {
            "resolved": resolved,
            "pending": pending,
            "resolved_count": len(resolved),
            "pending_count": len(pending),
        }
    except Exception:
        return {"resolved": [], "pending": [], "resolved_count": 0, "pending_count": 0}


def _query_approvals():
    """Pending approval items."""
    try:
        if not _cli or not _cli.evolution:
            return {"items": [], "count": 0}
        items = _cli.evolution.get_pending_approvals()
        return {"items": items, "count": len(items)}
    except Exception:
        return {"items": [], "count": 0}


def _query_approval_accuracy():
    """Category confidence scoring from approval_accuracy in telemetry.db."""
    import math
    try:
        conn = _ro_connect(_TELEMETRY_DB)

        # All-time per-category stats
        rows = conn.execute(
            "SELECT category, COUNT(*) as total, "
            "SUM(agreed) as agreed_count "
            "FROM approval_accuracy GROUP BY category"
        ).fetchall()

        # Volume in last 100 decisions (for suppression warning)
        recent_rows = conn.execute(
            "SELECT category, COUNT(*) as cnt FROM ("
            "  SELECT category FROM approval_accuracy "
            "  ORDER BY timestamp DESC LIMIT 100"
            ") GROUP BY category"
        ).fetchall()
        conn.close()

        recent_volume = {r["category"]: r["cnt"] for r in recent_rows}

        # Total decisions across all categories (for checking if we have 100+)
        total_decisions = sum(r["total"] for r in rows)

        categories = {}
        suppressed = []
        for r in rows:
            total = r["total"]
            agreed = r["agreed_count"] or 0
            rate = round(agreed / total, 3) if total > 0 else 0.0

            # Weighted accuracy: agreement_rate * (1 / log(total))
            # Use log base 2, floor total at 2 to avoid log(1)=0
            log_total = math.log2(max(total, 2))
            weighted_rate = round(rate * (1.0 / log_total), 3)

            # Use weighted accuracy for auto-widen threshold
            categories[r["category"]] = {
                "total": total,
                "agreed": agreed,
                "overridden": total - agreed,
                "agreement_rate": rate,
                "weighted_accuracy": weighted_rate,
                "auto_widen_candidate": weighted_rate >= 0.85 and total >= 20,
            }

            # Suppression check: category below 5 in last 100 decisions
            vol = recent_volume.get(r["category"], 0)
            if total_decisions >= 100 and vol < 5:
                suppressed.append(r["category"])

        return {"categories": categories, "suppressed": suppressed}
    except Exception:
        return {"categories": {}, "suppressed": []}


@app.get("/dashboard")
async def dashboard():
    """Batched dashboard data — all panels in one response."""
    return JSONResponse({
        "state": _query_state(),
        "beliefs": _query_beliefs(),
        "explorations": _query_explorations(),
        "gaps": _query_gaps(),
        "sleep_history": _query_sleep_history(),
        "approvals": _query_approvals(),
        "approval_accuracy": _query_approval_accuracy(),
        "contradictions": _query_contradictions(),
    })


@app.get("/state")
async def state():
    """Latest state vector from telemetry.db."""
    return JSONResponse(_query_state())


@app.get("/beliefs")
async def beliefs():
    """Belief summary + recent beliefs."""
    return JSONResponse(_query_beliefs())


@app.get("/explorations")
async def explorations():
    """Exploration summary."""
    return JSONResponse(_query_explorations())


@app.get("/gaps")
async def gaps():
    """Knowledge gaps / questions summary."""
    return JSONResponse(_query_gaps())


@app.get("/sleep-history")
async def sleep_history():
    """Recent consolidated episodes (sleep cycles)."""
    return JSONResponse(_query_sleep_history())


@app.get("/cycle-stats")
async def cycle_stats():
    """Episode statistics — same data as sleep-history, aliased for dashboard."""
    return JSONResponse(_query_sleep_history())


# ── Approval queue ──────────────────────────────────────────────────

@app.get("/approval-queue")
async def approval_queue():
    """Get pending approval items (beliefs, dreams, reflections)."""
    return JSONResponse(_query_approvals())


class ApprovalAction(BaseModel):
    approved: bool


@app.post("/approval-queue/accept-recommended")
async def accept_recommended():
    """Accept all recommended items, reject the rest."""
    import asyncio
    loop = asyncio.get_event_loop()

    def _do_batch():
        acquired = _chat_lock.acquire(timeout=5)
        if not acquired:
            # If cycle is running, do it without lock — approval queue is safe
            return _cli.evolution.resolve_recommended()
        try:
            return _cli.evolution.resolve_recommended()
        finally:
            _chat_lock.release()

    accepted, rejected = await loop.run_in_executor(_executor, _do_batch)
    return JSONResponse({"accepted": accepted, "rejected": rejected})


@app.post("/approval-queue/{item_id}")
async def resolve_approval(item_id: str, action: ApprovalAction):
    """Approve or reject a single queued item."""
    import asyncio
    loop = asyncio.get_event_loop()

    def _do_resolve():
        with _chat_lock:
            return _cli.evolution.resolve_approval(item_id, action.approved)

    result = await loop.run_in_executor(_executor, _do_resolve)
    if result:
        status = "approved" if action.approved else "rejected"
        return JSONResponse({"status": status, "id": item_id})
    return JSONResponse({"error": "Item not found or already resolved"}, status_code=404)


@app.post("/approval-queue/{item_id}/retriage")
async def retriage_approval(item_id: str):
    """Put a rejected approval queue item back to pending for re-evaluation."""
    if not _cli:
        return JSONResponse({"error": "ANIMA not initialized"}, status_code=400)

    try:
        db = _cli.evolution._get_queue_db()
        row = db.execute(
            "SELECT * FROM approval_queue WHERE id = ? AND status = 'rejected'",
            (item_id,),
        ).fetchone()
        if not row:
            return JSONResponse(
                {"error": "Item not found or not rejected"}, status_code=404
            )

        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        db.execute(
            "UPDATE approval_queue SET status = 'pending', resolved_at = NULL WHERE id = ?",
            (item_id,),
        )
        db.commit()

        data = json.loads(row["data"]) if isinstance(row["data"], str) else row["data"]
        # Re-run recommendation with current graph
        new_rec = _cli.evolution._recommend(row["category"], data)
        db.execute(
            "UPDATE approval_queue SET recommended = ? WHERE id = ?",
            (1 if new_rec else 0, item_id),
        )
        db.commit()

        stmt = data.get("statement") or data.get("inference") or "?"
        return JSONResponse({
            "status": "retriaged",
            "id": item_id,
            "new_recommendation": new_rec,
            "statement": stmt,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/approval-queue/rejected")
async def approval_queue_rejected():
    """Get rejected approval queue items for overturn review."""
    if not _cli:
        return JSONResponse({"items": [], "count": 0})

    try:
        db = _cli.evolution._get_queue_db()
        rows = db.execute(
            "SELECT * FROM approval_queue WHERE status = 'rejected' "
            "ORDER BY resolved_at DESC LIMIT 50"
        ).fetchall()
        items = []
        for r in rows:
            d = dict(r)
            d["data"] = json.loads(d["data"]) if isinstance(d["data"], str) else d["data"]
            d["recommended"] = bool(d["recommended"])
            items.append(d)
        return JSONResponse({"items": items, "count": len(items)})
    except Exception:
        return JSONResponse({"items": [], "count": 0})


# ── Dormant reactivation reviews ─────────────────────────────────────

@app.get("/dormant-reviews")
async def dormant_reviews():
    """Get pending dormant reactivation reviews."""
    try:
        if not _cli or not _cli.semantic:
            return JSONResponse({"items": [], "count": 0})
        items = _cli.semantic.get_pending_reactivations()
        return JSONResponse({"items": items, "count": len(items)})
    except Exception:
        return JSONResponse({"items": [], "count": 0})


class DormantReviewAction(BaseModel):
    approved: bool


@app.post("/dormant-reviews/{review_id}")
async def resolve_dormant_review(review_id: str, action: DormantReviewAction):
    """Approve or dismiss a dormant reactivation review."""
    import asyncio
    loop = asyncio.get_event_loop()

    def _do_resolve():
        with _chat_lock:
            return _cli.semantic.resolve_dormant_review(review_id, action.approved)

    result = await loop.run_in_executor(_executor, _do_resolve)
    if result:
        status = "approved" if action.approved else "dismissed"
        return JSONResponse({"status": status, "id": review_id})
    return JSONResponse({"error": "Review not found or already resolved"}, status_code=404)


# ── Contradiction management ─────────────────────────────────────────

@app.post("/contradictions/{contradiction_id}/restore")
async def restore_contradiction(contradiction_id: str):
    """Restore a deprecated belief from contradiction auto-resolution."""
    import asyncio
    loop = asyncio.get_event_loop()

    def _do_restore():
        with _chat_lock:
            return _cli.evolution.restore_from_contradiction(contradiction_id)

    result = await loop.run_in_executor(_executor, _do_restore)
    if result:
        return JSONResponse({"status": "restored", "id": contradiction_id})
    return JSONResponse({"error": "Contradiction not found"}, status_code=404)


# ── Document Pipeline ──────────────────────────────────────────────

@app.post("/documents/scan")
async def documents_scan():
    """Scan datafiles directory and populate document ledger."""
    import asyncio
    loop = asyncio.get_event_loop()

    if not _cli:
        return JSONResponse({"error": "ANIMA not initialized"}, status_code=400)

    def _do_scan():
        with _chat_lock:
            from ingestion.pipeline import DocumentPipeline
            pipeline = DocumentPipeline(_cli.evolution)
            return pipeline.run_scan()

    stats = await loop.run_in_executor(_executor, _do_scan)
    return JSONResponse({"status": "scan_complete", "stats": stats})


@app.post("/documents/ingest")
async def documents_ingest(limit: int = None):
    """Extract beliefs from indexed documents through governance gate."""
    import asyncio
    loop = asyncio.get_event_loop()

    if not _cli:
        return JSONResponse({"error": "ANIMA not initialized"}, status_code=400)

    def _do_ingest():
        with _chat_lock:
            from ingestion.pipeline import DocumentPipeline
            pipeline = DocumentPipeline(_cli.evolution)
            return pipeline.run_ingest(limit=limit)

    stats = await loop.run_in_executor(_executor, _do_ingest)
    return JSONResponse({"status": "ingest_complete", "stats": stats})


@app.post("/documents/run")
async def documents_run(limit: int = None):
    """Run full pipeline: scan → ingest."""
    import asyncio
    loop = asyncio.get_event_loop()

    if not _cli:
        return JSONResponse({"error": "ANIMA not initialized"}, status_code=400)

    def _do_run():
        with _chat_lock:
            from ingestion.pipeline import DocumentPipeline
            pipeline = DocumentPipeline(_cli.evolution)
            return pipeline.run_full(limit=limit)

    result = await loop.run_in_executor(_executor, _do_run)
    return JSONResponse({"status": "pipeline_complete", "result": result})


@app.post("/documents/cycle")
async def documents_cycle():
    """Run one full DMS lifecycle cycle: scan → ingest → dream → curiosity match → re-read → dream."""
    import asyncio
    loop = asyncio.get_event_loop()

    if not _cli:
        return JSONResponse({"error": "ANIMA not initialized"}, status_code=400)

    def _do_cycle():
        with _chat_lock:
            orch = _get_orchestrator()
            return orch.run_cycle()

    stats = await loop.run_in_executor(_executor, _do_cycle)
    return JSONResponse({"status": "cycle_complete", "stats": stats})


@app.post("/documents/resolve-gaps")
async def documents_resolve_gaps():
    """Manually trigger gap resolution — check if beliefs answer open gaps."""
    import asyncio
    loop = asyncio.get_event_loop()

    if not _cli:
        return JSONResponse({"error": "ANIMA not initialized"}, status_code=400)

    def _do_resolve():
        with _chat_lock:
            orch = _get_orchestrator()
            return orch.resolve_gaps()

    stats = await loop.run_in_executor(_executor, _do_resolve)
    return JSONResponse({"status": "resolve_complete", "stats": stats})


_autonomous_orchestrator = None
_autonomous_thread = None


@app.post("/documents/autonomous/start")
async def documents_autonomous_start(interval_minutes: int = None):
    """Start the autonomous DMS lifecycle loop in a background thread."""
    global _autonomous_orchestrator, _autonomous_thread
    import threading

    if not _cli:
        return JSONResponse({"error": "ANIMA not initialized"}, status_code=400)

    if _autonomous_thread and _autonomous_thread.is_alive():
        return JSONResponse({"status": "already_running"})

    _autonomous_orchestrator = _get_orchestrator()
    _autonomous_thread = threading.Thread(
        target=_autonomous_orchestrator.run_autonomous,
        args=(_chat_lock,),
        kwargs={"interval_minutes": interval_minutes},
        daemon=True,
    )
    _autonomous_thread.start()
    return JSONResponse({"status": "started", "interval_minutes": interval_minutes})


@app.post("/documents/autonomous/stop")
async def documents_autonomous_stop():
    """Stop the autonomous DMS lifecycle loop."""
    global _autonomous_orchestrator
    if _autonomous_orchestrator:
        _autonomous_orchestrator.stop_autonomous()
        return JSONResponse({"status": "stopping"})
    return JSONResponse({"status": "not_running"})


def _get_orchestrator():
    """Create a DMSOrchestrator instance from the current CLI state."""
    from ingestion.orchestrator import DMSOrchestrator
    return DMSOrchestrator(
        _cli.evolution, _cli.embeddings,
        getattr(_cli, "curiosity", None),
        _cli.config,
    )


@app.get("/documents/orchestrator-status")
async def documents_orchestrator_status():
    """Full DMS dashboard data — graph, entities, gaps, tier usage, re-read effectiveness."""
    if not _cli:
        return JSONResponse({"error": "ANIMA not initialized"}, status_code=400)

    db = _cli.evolution.semantic.db_conn

    # ── Ledger ──
    ledger = {}
    for row in db.execute("SELECT status, COUNT(*) as cnt FROM document_ledger GROUP BY status").fetchall():
        ledger[row["status"]] = row["cnt"]

    # ── Graph ──
    belief_count = db.execute("SELECT COUNT(*) as cnt FROM beliefs WHERE COALESCE(deprecated,0)=0 AND COALESCE(is_dormant,0)=0").fetchone()["cnt"]
    edge_count = db.execute("SELECT COUNT(*) as cnt FROM belief_links").fetchone()["cnt"]
    isolated = db.execute(
        """SELECT COUNT(*) as cnt FROM beliefs
           WHERE COALESCE(deprecated,0)=0 AND COALESCE(is_dormant,0)=0
           AND id NOT IN (SELECT belief_a FROM belief_links UNION SELECT belief_b FROM belief_links)"""
    ).fetchone()["cnt"]
    edge_density = round(edge_count / max(belief_count, 1), 2)

    # ── Entities ──
    entity_count = db.execute("SELECT COUNT(*) as cnt FROM document_entities").fetchone()["cnt"]
    top_topics = [
        {"topic": r["entity_value"], "count": r["cnt"]}
        for r in db.execute(
            """SELECT entity_value, COUNT(*) as cnt FROM document_entities
               WHERE entity_type = 'topic' GROUP BY LOWER(entity_value)
               ORDER BY cnt DESC LIMIT 15"""
        ).fetchall()
    ]
    entity_type_dist = {
        r["entity_type"]: r["cnt"]
        for r in db.execute(
            "SELECT entity_type, COUNT(*) as cnt FROM document_entities GROUP BY entity_type ORDER BY cnt DESC"
        ).fetchall()
    }

    # ── Gaps ──
    open_gaps = db.execute("SELECT COUNT(*) as cnt FROM questions WHERE status='open'").fetchone()["cnt"]
    low_yield_gaps = 0
    try:
        low_yield_gaps = db.execute(
            "SELECT COUNT(*) as cnt FROM dms_gap_effectiveness WHERE marked_low_yield=1"
        ).fetchone()["cnt"]
    except Exception:
        pass

    # ── Tier usage (from dms_query_log) ──
    tier_usage = {}
    try:
        for row in db.execute(
            "SELECT tier_final, COUNT(*) as cnt FROM dms_query_log GROUP BY tier_final"
        ).fetchall():
            tier_usage[row["tier_final"]] = row["cnt"]
    except Exception:
        pass

    fallback_rate = 0.0
    try:
        total_queries = db.execute("SELECT COUNT(*) as cnt FROM dms_query_log").fetchone()["cnt"]
        fallback_queries = db.execute(
            "SELECT COUNT(*) as cnt FROM dms_query_log WHERE fallback_triggered=1"
        ).fetchone()["cnt"]
        if total_queries > 0:
            fallback_rate = round(fallback_queries / total_queries, 2)
    except Exception:
        pass

    avg_latency = 0
    try:
        row = db.execute("SELECT AVG(latency_ms) as avg_ms FROM dms_query_log").fetchone()
        avg_latency = int(row["avg_ms"] or 0)
    except Exception:
        pass

    # ── Re-read effectiveness (from dms_reread_log) ──
    reread_stats = {"total_rereads": 0, "total_accepted": 0, "total_novel": 0,
                    "total_duplicate": 0, "accept_rate": 0.0}
    try:
        row = db.execute(
            """SELECT COUNT(*) as cnt,
                      SUM(beliefs_accepted) as accepted,
                      SUM(beliefs_novel) as novel,
                      SUM(beliefs_duplicate) as dup
               FROM dms_reread_log"""
        ).fetchone()
        if row and row["cnt"]:
            reread_stats["total_rereads"] = row["cnt"]
            reread_stats["total_accepted"] = row["accepted"] or 0
            reread_stats["total_novel"] = row["novel"] or 0
            reread_stats["total_duplicate"] = row["dup"] or 0
            total_extracted = (reread_stats["total_novel"] + reread_stats["total_duplicate"]) or 1
            reread_stats["accept_rate"] = round(
                reread_stats["total_accepted"] / total_extracted, 2
            )
    except Exception:
        pass

    # ── Gap effectiveness (top productive + low-yield) ──
    productive_gaps = []
    try:
        productive_gaps = [
            {"gap_id": r["gap_id"], "times_matched": r["times_matched"],
             "total_accepted": r["total_accepted"]}
            for r in db.execute(
                """SELECT gap_id, times_matched, total_accepted
                   FROM dms_gap_effectiveness WHERE total_accepted > 0
                   ORDER BY total_accepted DESC LIMIT 5"""
            ).fetchall()
        ]
    except Exception:
        pass

    # ── Document coverage (beliefs per doc) ──
    docs_with_beliefs = db.execute(
        "SELECT COUNT(*) as cnt FROM document_ledger WHERE status='ingested' AND belief_count > 0"
    ).fetchone()["cnt"]
    docs_without_beliefs = db.execute(
        "SELECT COUNT(*) as cnt FROM document_ledger WHERE status='ingested' AND belief_count = 0"
    ).fetchone()["cnt"]

    # Beliefs with at least one edge (graph integration)
    integrated_beliefs = 0
    if belief_count > 0:
        integrated_beliefs = db.execute(
            """SELECT COUNT(DISTINCT id) as cnt FROM beliefs
               WHERE COALESCE(deprecated,0)=0 AND COALESCE(is_dormant,0)=0
               AND id IN (SELECT belief_a FROM belief_links UNION SELECT belief_b FROM belief_links)"""
        ).fetchone()["cnt"]

    return JSONResponse({
        "ledger": ledger,
        "graph": {
            "beliefs": belief_count,
            "edges": edge_count,
            "edge_density": edge_density,
            "isolated_beliefs": isolated,
            "integrated_beliefs": integrated_beliefs,
            "integration_pct": round(integrated_beliefs / max(belief_count, 1) * 100, 1),
        },
        "entities": {
            "total": entity_count,
            "by_type": entity_type_dist,
            "top_topics": top_topics,
        },
        "gaps": {
            "open": open_gaps,
            "low_yield": low_yield_gaps,
            "productive": productive_gaps,
        },
        "tier_usage": {
            "distribution": tier_usage,
            "fallback_rate": fallback_rate,
            "avg_latency_ms": avg_latency,
        },
        "reread": reread_stats,
        "document_coverage": {
            "docs_with_beliefs": docs_with_beliefs,
            "docs_without_beliefs": docs_without_beliefs,
            "coverage_pct": round(
                docs_with_beliefs / max(docs_with_beliefs + docs_without_beliefs, 1) * 100, 1
            ),
        },
        "triage": _get_triage_stats(db),
    })


def _get_triage_stats(db):
    """Get triage decision distribution and recent reasoning samples."""
    stats = {"total": 0, "by_decision": {}, "recent_rejections": []}
    try:
        for row in db.execute(
            "SELECT decision, COUNT(*) as cnt FROM dms_triage_log GROUP BY decision"
        ).fetchall():
            stats["by_decision"][row["decision"]] = row["cnt"]
            stats["total"] += row["cnt"]

        # Accept rate
        accepted = stats["by_decision"].get("accept", 0) + stats["by_decision"].get("auto_accept", 0)
        stats["accept_rate"] = round(accepted / max(stats["total"], 1), 2)

        # Recent rejections with reasoning
        stats["recent_rejections"] = [
            {"statement": r["statement"][:100], "reason": r["reason"][:200],
             "confidence": r["confidence"], "created_at": r["created_at"]}
            for r in db.execute(
                """SELECT statement, reason, confidence, created_at
                   FROM dms_triage_log WHERE decision = 'reject'
                   ORDER BY created_at DESC LIMIT 10"""
            ).fetchall()
        ]
    except Exception:
        pass
    return stats


@app.post("/documents/extract-entities")
async def documents_extract_entities(limit: int = None):
    """Run entity extraction on ingested docs missing entities."""
    import asyncio
    loop = asyncio.get_event_loop()

    if not _cli:
        return JSONResponse({"error": "ANIMA not initialized"}, status_code=400)

    def _do_extract():
        with _chat_lock:
            from ingestion.pipeline import DocumentPipeline
            pipeline = DocumentPipeline(_cli.evolution)
            return pipeline.run_entity_extraction(limit=limit)

    stats = await loop.run_in_executor(_executor, _do_extract)
    return JSONResponse({"status": "entity_extraction_complete", "stats": stats})


@app.get("/documents/status")
async def documents_status():
    """Get document ledger summary and recent ingestions."""
    if not _cli:
        return JSONResponse({"error": "ANIMA not initialized"}, status_code=400)

    from ingestion.pipeline import DocumentPipeline
    pipeline = DocumentPipeline(_cli.evolution)
    return JSONResponse(pipeline.get_status())


# ── Document query endpoints ───────────────────────────────────────


@app.post("/documents/search")
async def documents_search(request: Request):
    """Search documents by entity filters and/or semantic query.

    Resolution order:
    1. Entity match — structured, fast, exact
    2. Semantic match — embedding similarity against beliefs, fuzzy
    3. Graph enrichment — dream-linked documents via belief bridges
    """
    if not _cli:
        return JSONResponse({"error": "ANIMA not initialized"}, status_code=400)

    body = await request.json()
    query = body.get("query", "")
    filters = body.get("filters", {})
    semantic = body.get("semantic", True)
    limit = body.get("limit", 20)

    db = _cli.evolution.semantic.db_conn
    results = []
    seen_shas = set()

    # Phase 1: Entity search
    if filters:
        entity_type = filters.get("entity_type")
        entity_value = filters.get("entity_value")
        if entity_type and entity_value:
            rows = db.execute(
                """SELECT DISTINCT de.document_sha, dl.filename, dl.belief_count
                   FROM document_entities de
                   JOIN document_ledger dl ON de.document_sha = dl.sha256
                   WHERE de.entity_type = ? AND LOWER(de.entity_value) LIKE ?
                   LIMIT ?""",
                (entity_type, f"%{entity_value.lower()}%", limit),
            ).fetchall()
            for row in rows:
                sha = row["document_sha"]
                if sha not in seen_shas:
                    seen_shas.add(sha)
                    # Get all entities for this doc
                    entities = _get_doc_entities(db, sha)
                    results.append({
                        "document_sha": sha,
                        "filename": row["filename"],
                        "match_type": "entity",
                        "matched_entities": [
                            e for e in entities
                            if e["type"] == entity_type
                            and entity_value.lower() in e["value"].lower()
                        ],
                        "belief_count": row["belief_count"] or 0,
                    })
    elif query:
        # No structured filter — do text search across all entity values
        rows = db.execute(
            """SELECT DISTINCT de.document_sha, dl.filename, dl.belief_count,
                      de.entity_type, de.entity_value
               FROM document_entities de
               JOIN document_ledger dl ON de.document_sha = dl.sha256
               WHERE LOWER(de.entity_value) LIKE ?
               LIMIT ?""",
            (f"%{query.lower()}%", limit),
        ).fetchall()
        for row in rows:
            sha = row["document_sha"]
            if sha not in seen_shas:
                seen_shas.add(sha)
                results.append({
                    "document_sha": sha,
                    "filename": row["filename"],
                    "match_type": "entity",
                    "matched_entities": [{"type": row["entity_type"], "value": row["entity_value"]}],
                    "belief_count": row["belief_count"] or 0,
                })

    # Phase 2: Semantic search via belief embeddings
    if semantic and query and len(results) < limit:
        try:
            similar = _cli.evolution.semantic.find_similar(
                query, top_k=limit * 2
            )
            for belief_id, score in similar:
                if len(results) >= limit:
                    break
                # Find document(s) that sourced this belief
                doc_rows = db.execute(
                    """SELECT bsd.document_sha, dl.filename, dl.belief_count
                       FROM belief_sources_documents bsd
                       JOIN document_ledger dl ON bsd.document_sha = dl.sha256
                       WHERE bsd.belief_id = ?""",
                    (belief_id,),
                ).fetchall()
                for dr in doc_rows:
                    sha = dr["document_sha"]
                    if sha not in seen_shas:
                        seen_shas.add(sha)
                        results.append({
                            "document_sha": sha,
                            "filename": dr["filename"],
                            "match_type": "semantic",
                            "similarity": round(score, 3),
                            "belief_count": dr["belief_count"] or 0,
                        })
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")

    # Phase 3: Graph enrichment — add dream-linked documents
    for result in results[:]:
        sha = result["document_sha"]
        related = _get_related_documents(db, sha)
        if related:
            result["related_documents"] = related

    return JSONResponse({"results": results[:limit]})


@app.get("/documents/{sha}/entities")
async def document_entities(sha: str):
    """Return all extracted entities for a document."""
    if not _cli:
        return JSONResponse({"error": "ANIMA not initialized"}, status_code=400)

    db = _cli.evolution.semantic.db_conn
    entities = _get_doc_entities(db, sha)
    return JSONResponse({"document_sha": sha, "entities": entities})


@app.get("/documents/{sha}/beliefs")
async def document_beliefs(sha: str):
    """Return all beliefs sourced from a document."""
    if not _cli:
        return JSONResponse({"error": "ANIMA not initialized"}, status_code=400)

    db = _cli.evolution.semantic.db_conn
    rows = db.execute(
        """SELECT b.id, b.statement, b.confidence, b.source_type, b.created_at
           FROM belief_sources_documents bsd
           JOIN beliefs b ON bsd.belief_id = b.id
           WHERE bsd.document_sha = ?
           ORDER BY b.created_at DESC""",
        (sha,),
    ).fetchall()

    return JSONResponse({
        "document_sha": sha,
        "beliefs": [dict(r) for r in rows],
    })


@app.get("/documents/{sha}/related")
async def document_related(sha: str):
    """Find all related documents — entity overlap + belief bridges."""
    if not _cli:
        return JSONResponse({"error": "ANIMA not initialized"}, status_code=400)

    db = _cli.evolution.semantic.db_conn

    # Get doc info
    doc = db.execute(
        "SELECT filename, belief_count FROM document_ledger WHERE sha256 = ?",
        (sha,),
    ).fetchone()
    if not doc:
        return JSONResponse({"error": "Document not found"}, status_code=404)

    related = _get_related_documents(db, sha)
    return JSONResponse({
        "document_sha": sha,
        "filename": doc["filename"],
        "related_documents": related,
    })


def _get_doc_entities(db, sha):
    """Get all entities for a document as a list of dicts."""
    rows = db.execute(
        "SELECT entity_type, entity_value, confidence FROM document_entities WHERE document_sha = ?",
        (sha,),
    ).fetchall()
    return [{"type": r["entity_type"], "value": r["entity_value"], "confidence": r["confidence"]} for r in rows]


def _get_related_documents(db, sha):
    """Find documents related via entity overlap and belief bridges."""
    related = []
    seen = set()

    # Entity overlap — shared entities between documents
    entity_rows = db.execute(
        """SELECT DISTINCT b.document_sha, dl.filename, b.entity_type, b.entity_value
           FROM document_entities a
           JOIN document_entities b ON a.entity_type = b.entity_type
             AND LOWER(a.entity_value) = LOWER(b.entity_value)
           JOIN document_ledger dl ON b.document_sha = dl.sha256
           WHERE a.document_sha = ? AND b.document_sha != a.document_sha""",
        (sha,),
    ).fetchall()

    for row in entity_rows:
        rel_sha = row["document_sha"]
        if rel_sha not in seen:
            seen.add(rel_sha)
            related.append({
                "document_sha": rel_sha,
                "filename": row["filename"],
                "link_type": "entity_overlap",
                "shared": f"{row['entity_type']}:{row['entity_value']}",
            })

    # Belief bridges — documents linked via dream edges
    bridge_rows = db.execute(
        """SELECT DISTINCT bsd2.document_sha, dl.filename
           FROM belief_sources_documents bsd1
           JOIN belief_links bl ON (bsd1.belief_id = bl.belief_a OR bsd1.belief_id = bl.belief_b)
           JOIN belief_sources_documents bsd2
             ON (bsd2.belief_id = bl.belief_a OR bsd2.belief_id = bl.belief_b)
           JOIN document_ledger dl ON bsd2.document_sha = dl.sha256
           WHERE bsd1.document_sha = ? AND bsd2.document_sha != bsd1.document_sha""",
        (sha,),
    ).fetchall()

    for row in bridge_rows:
        rel_sha = row["document_sha"]
        if rel_sha not in seen:
            seen.add(rel_sha)
            # Count bridges
            count = db.execute(
                """SELECT COUNT(*) as cnt
                   FROM belief_sources_documents bsd1
                   JOIN belief_links bl ON (bsd1.belief_id = bl.belief_a OR bsd1.belief_id = bl.belief_b)
                   JOIN belief_sources_documents bsd2
                     ON (bsd2.belief_id = bl.belief_a OR bsd2.belief_id = bl.belief_b)
                   WHERE bsd1.document_sha = ? AND bsd2.document_sha = ?""",
                (sha, rel_sha),
            ).fetchone()["cnt"]
            related.append({
                "document_sha": rel_sha,
                "filename": row["filename"],
                "link_type": "belief_bridge",
                "bridge_count": count,
            })

    return related


# Research endpoints moved to plugins/research/endpoints.py

# ── Model routing management ──────────────────────────────────────────


@app.get("/models/status")
async def models_status():
    """Get status of all configured models and routing info."""
    if not _cli:
        return JSONResponse({"error": "ANIMA not initialized"}, status_code=400)
    router = getattr(_cli, "router", None)
    if not router:
        return JSONResponse({"mode": "solo", "models": {}, "error": "No router"})
    return JSONResponse(router.get_status())


@app.get("/router/capacity")
async def router_capacity(task_class: str = None):
    """Query router capacity for a task type. Shows fairness state."""
    router = _blade_runner.router if _blade_runner else None
    if not router:
        return JSONResponse({"error": "No router"}, status_code=503)
    task_desc = None
    if task_class:
        from core.task_presets import resolve_task
        task_desc = resolve_task(task_class)
    return JSONResponse(router.get_capacity(task_desc))


@app.post("/models/health-check")
async def models_health_check():
    """Run health check on all models."""
    if not _cli:
        return JSONResponse({"error": "ANIMA not initialized"}, status_code=400)
    router = getattr(_cli, "router", None)
    if not router:
        return JSONResponse({"error": "No router"}, status_code=400)
    results = router.health_check()
    return JSONResponse(results)


@app.post("/models/{key}/toggle")
async def models_toggle(key: str):
    """Enable/disable a model without removing config."""
    if not _cli:
        return JSONResponse({"error": "ANIMA not initialized"}, status_code=400)
    router = getattr(_cli, "router", None)
    if not router or key not in router.models:
        return JSONResponse({"error": f"Model '{key}' not found"}, status_code=404)

    current = router.models[key].enabled
    router.set_enabled(key, not current)

    # Persist to config
    config_path = os.environ.get("ANIMA_CONFIG", "config/settings.toml")
    router.persist_to_config(config_path)

    return JSONResponse({
        "model": key,
        "enabled": not current,
        "status": "disabled" if current else "enabled",
    })


@app.get("/models/routing-preference")
async def models_get_routing_preference():
    """Routing preference — all enabled models participate in parallel."""
    return JSONResponse({"preference": "parallel"})


@app.post("/models/persist")
async def models_persist():
    """Persist current model config to settings.toml."""
    router = getattr(_cli, "router", None) or getattr(_cli, "inference", None)
    if not router:
        return JSONResponse({"error": "No router"}, status_code=400)
    from core.router import ModelRouter
    if isinstance(router, ModelRouter):
        config_path = os.environ.get("ANIMA_CONFIG", "config/settings.toml")
        router.persist_to_config(config_path)
        return JSONResponse({"status": "persisted"})
    return JSONResponse({"error": "Not a model router"}, status_code=400)


@app.post("/models/set-api-key")
async def models_set_api_key(request: Request):
    """Set API key — encrypted at rest, auto-loaded on restart."""
    try:
        body = await request.json()
        api_key = body.get("api_key", "").strip()
        env_var = body.get("env_var", "ANTHROPIC_API_KEY")
        if not api_key:
            return JSONResponse({"error": "No key provided"}, status_code=400)

        # Set in current process
        os.environ[env_var] = api_key

        # Encrypt and persist — machine-bound, useless if file is copied
        _save_encrypted_key(env_var, api_key)

        return JSONResponse({"status": "set", "persisted": True})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


def _get_machine_key():
    """Derive encryption key from machine-specific identifiers."""
    import hashlib, uuid, socket
    machine_id = f"{socket.gethostname()}-{uuid.getnode()}"
    return hashlib.sha256(machine_id.encode()).digest()


def _save_encrypted_key(env_var, api_key):
    """Encrypt and save API key to core data dir (stable, not plugin-specific)."""
    from cryptography.fernet import Fernet
    import base64, hashlib
    key = base64.urlsafe_b64encode(_get_machine_key())
    f = Fernet(key)
    encrypted = f.encrypt(api_key.encode()).decode()

    # Always save to core data dir, not plugin-specific
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    keyfile = os.path.join(data_dir, ".keys")
    entries = {}
    if os.path.exists(keyfile):
        try:
            import json
            entries = json.loads(open(keyfile).read())
        except Exception:
            entries = {}
    entries[env_var] = encrypted
    import json
    with open(keyfile, "w") as fh:
        fh.write(json.dumps(entries))


def _load_encrypted_keys():
    """Load and decrypt API keys on startup."""
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    keyfile = os.path.join(data_dir, ".keys")
    if not os.path.exists(keyfile):
        return
    try:
        from cryptography.fernet import Fernet
        import base64, json
        key = base64.urlsafe_b64encode(_get_machine_key())
        f = Fernet(key)
        entries = json.loads(open(keyfile).read())
        for env_var, encrypted in entries.items():
            decrypted = f.decrypt(encrypted.encode()).decode()
            os.environ[env_var] = decrypted
            logger.info(f"Loaded encrypted key for {env_var}")
    except ImportError:
        logger.warning("cryptography package not installed — cannot load encrypted keys. pip install cryptography")
    except Exception as e:
        logger.warning(f"Failed to load encrypted keys: {e}")


@app.post("/models/probe-and-add")
async def models_probe_and_add(request: Request):
    """Probe endpoint, auto-detect everything, add model, calibrate, save. One click."""
    if not _cli:
        return JSONResponse({"error": "ANIMA not initialized"}, status_code=400)

    try:
        body = await request.json()
        host = body.get("host", "localhost")
        port = int(body.get("port", 8080))
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    from core.router import probe_endpoint, ModelRouter
    router = getattr(_cli, "router", None) or getattr(_cli, "inference", None)
    if not isinstance(router, ModelRouter):
        return JSONResponse({"error": "No model router"}, status_code=400)

    # Probe
    probe = probe_endpoint(host, port)
    if not probe or not probe.get("detected"):
        return JSONResponse({"status": "not_detected", "error": f"No backend found at {host}:{port}"})

    # Build config from probe + operator overrides
    key = f"model_{int(__import__('time').time() * 1000)}"
    operator_name = body.get("name")
    operator_tier = body.get("tier")
    model_cfg = {
        "name": operator_name or probe.get("auto_name") or probe.get("model_name") or f"{host}:{port}",
        "endpoint": probe["endpoint"],
        "tier": operator_tier or probe["tier"],
        "context_window": probe["context_window"],
        "thinking_prefix": probe["thinking_prefix"],
        "backend": probe["backend"],
        "enabled": True,
    }

    # Add to router (triggers calibration automatically)
    try:
        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor, lambda: router.add_model(key, model_cfg)
        )

        # Persist to config
        config_path = os.environ.get("ANIMA_CONFIG", "config/settings.toml")
        router.persist_to_config(config_path)

        return JSONResponse({
            "status": "added",
            "key": key,
            "model": result,
            "probe": probe,
        })
    except Exception as e:
        return JSONResponse({
            "status": "detected_only",
            "probe": probe,
            "error": str(e),
        })


@app.post("/models/probe")
async def models_probe(request: Request):
    """Probe an endpoint to auto-detect backend, model, and capabilities.

    Body: {"host": "localhost", "port": 8080}
    Returns detected info: backend, model_name, tier, context_window, roles, etc.
    """
    try:
        body = await request.json()
        host = body.get("host", "localhost")
        port = int(body.get("port", 8080))
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    from core.router import probe_endpoint
    result = probe_endpoint(host, port)
    if result and result.get("detected"):
        return JSONResponse({"status": "detected", "probe": result})
    return JSONResponse({"status": "unreachable", "probe": result})


@app.post("/models/add")
async def models_add(request: Request):
    """Add a new model endpoint. Auto-detects if only endpoint provided.

    Body: {"key": "my-model", "endpoint": "http://...:8080"}
    Or full manual: {"key": "...", "name": "...", "endpoint": "...",
           "context_window": 8192, "thinking_prefix": false}
    Capabilities auto-detected from known model registry.
    """
    if not _cli:
        return JSONResponse({"error": "ANIMA not initialized"}, status_code=400)
    router = getattr(_cli, "router", None)
    if not router:
        return JSONResponse({"error": "No router"}, status_code=400)

    body = await request.json()
    key = body.pop("key", None)
    if not key:
        return JSONResponse({"error": "Missing 'key'"}, status_code=400)
    if not body.get("endpoint"):
        return JSONResponse({"error": "Missing 'endpoint'"}, status_code=400)

    # Normalize endpoint — always needs http://
    ep = body["endpoint"]
    if not ep.startswith("http://") and not ep.startswith("https://"):
        body["endpoint"] = "http://" + ep

    result = router.add_model(key, body)

    config_path = os.environ.get("ANIMA_CONFIG", "config/settings.toml")
    router.persist_to_config(config_path)

    return JSONResponse({"status": "added", "model": result})


@app.post("/models/persist")
async def models_persist():
    """Persist current model config to settings.toml."""
    router = getattr(_cli, "router", None) if _cli else None
    if not router:
        return JSONResponse({"error": "No router"}, status_code=400)
    config_path = os.environ.get("ANIMA_CONFIG", "config/settings.toml")
    router.persist_to_config(config_path)
    return JSONResponse({"status": "persisted"})


@app.post("/models/{key}/remove")
async def models_remove(key: str):
    """Remove a model endpoint."""
    if not _cli:
        return JSONResponse({"error": "ANIMA not initialized"}, status_code=400)
    router = getattr(_cli, "router", None)
    if not router:
        return JSONResponse({"error": "No router"}, status_code=400)

    success = router.remove_model(key)
    if not success:
        return JSONResponse(
            {"error": f"Cannot remove '{key}' — not found or last model"},
            status_code=400,
        )

    config_path = os.environ.get("ANIMA_CONFIG", "config/settings.toml")
    router.persist_to_config(config_path)

    return JSONResponse({"status": "removed", "model": key})


@app.post("/models/{key}/update")
async def models_update(key: str, request: Request):
    """Update model config fields.

    Body: {"endpoint": "...", "name": "...", "enabled": true, ...}
    """
    if not _cli:
        return JSONResponse({"error": "ANIMA not initialized"}, status_code=400)
    router = getattr(_cli, "router", None)
    if not router:
        return JSONResponse({"error": "No router"}, status_code=400)

    body = await request.json()

    # Normalize endpoint
    if body.get("endpoint"):
        ep = body["endpoint"]
        if not ep.startswith("http://") and not ep.startswith("https://"):
            body["endpoint"] = "http://" + ep

    success = router.update_model(key, body)
    if not success:
        return JSONResponse({"error": f"Model '{key}' not found"}, status_code=404)

    config_path = os.environ.get("ANIMA_CONFIG", "config/settings.toml")
    router.persist_to_config(config_path)

    return JSONResponse({"status": "updated", "model": router.models[key].to_dict()})


# ── Static files ────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


@app.get("/plugin-dashboard/{plugin_name}")
async def plugin_dashboard(plugin_name: str):
    """Serve a plugin's dashboard HTML by name."""
    if not _blade_runner:
        return JSONResponse({"error": "Not initialized"}, status_code=503)
    plugin = _blade_runner.plugins.get(plugin_name)
    if not plugin or not plugin.plugin:
        return JSONResponse({"error": f"Plugin '{plugin_name}' not found"}, status_code=404)
    # Try multiple locations: dashboard/templates/, dashboard/, plugin root
    search_paths = [
        os.path.join(plugin.plugin.plugin_dir, "dashboard", "templates", f"{plugin_name}_dashboard.html"),
        os.path.join(plugin.plugin.plugin_dir, "dashboard", "templates", "dashboard.html"),
        os.path.join(plugin.plugin.plugin_dir, "dashboard.html"),
    ]
    for path in search_paths:
        if os.path.isfile(path):
            return FileResponse(path)
    return JSONResponse({"error": f"No dashboard template for '{plugin_name}'"}, status_code=404)


@app.get("/plugin-dashboard/{file_path:path}")
async def plugin_static_file(file_path: str):
    """Serve static files from plugin directories (graph.html, tree.html, etc.).

    Resolves relative links from plugin dashboards. The path is split as
    plugin_name/filename — e.g., /plugin-dashboard/research/graph.html
    serves plugins/research/graph.html.
    """
    if not _blade_runner:
        return JSONResponse({"error": "Not initialized"}, status_code=503)

    # Split into plugin name and remaining path
    parts = file_path.split("/", 1)
    if len(parts) < 2:
        return JSONResponse({"error": "Not found"}, status_code=404)

    plugin_name, rel_path = parts
    plugin = _blade_runner.plugins.get(plugin_name)
    if not plugin or not plugin.plugin:
        return JSONResponse({"error": f"Plugin '{plugin_name}' not found"}, status_code=404)

    # Security: only serve .html, .css, .js files from plugin dir
    if not rel_path.endswith(('.html', '.css', '.js', '.json')):
        return JSONResponse({"error": "Forbidden file type"}, status_code=403)

    # Search in plugin root and dashboard subdirectory
    search_paths = [
        os.path.join(plugin.plugin.plugin_dir, rel_path),
        os.path.join(plugin.plugin.plugin_dir, "dashboard", rel_path),
        os.path.join(plugin.plugin.plugin_dir, "dashboard", "templates", rel_path),
    ]
    for path in search_paths:
        full = os.path.normpath(path)
        # Prevent directory traversal
        if not full.startswith(os.path.normpath(plugin.plugin.plugin_dir)):
            continue
        if os.path.isfile(full):
            return FileResponse(full)

    return JSONResponse({"error": f"File not found: {rel_path}"}, status_code=404)


# ---------------------------------------------------------------------------
# Plugin orchestrator control — ONE authority, core controls everything
# ---------------------------------------------------------------------------

@app.post("/api/plugins/{plugin_id}/orchestrator/start")
async def plugin_orchestrator_start(plugin_id: str):
    """Start a plugin's orchestrator (Layer 2 — plugin-specific work)."""
    if not _blade_runner:
        return JSONResponse({"error": "Not initialized"}, status_code=503)
    plugin = _blade_runner.plugins.get(plugin_id)
    if not plugin:
        return JSONResponse({"error": f"Plugin '{plugin_id}' not found"}, status_code=404)
    plugin.start_plugin_orchestrator()
    return {"ok": True, "plugin": plugin_id, "status": "started"}


@app.post("/api/plugins/{plugin_id}/orchestrator/stop")
async def plugin_orchestrator_stop(plugin_id: str):
    """Stop a plugin's orchestrator."""
    if not _blade_runner:
        return JSONResponse({"error": "Not initialized"}, status_code=503)
    plugin = _blade_runner.plugins.get(plugin_id)
    if not plugin:
        return JSONResponse({"error": f"Plugin '{plugin_id}' not found"}, status_code=404)
    plugin.stop_plugin_orchestrator()
    return {"ok": True, "plugin": plugin_id, "status": "stopped"}


@app.get("/api/plugins/{plugin_id}/orchestrator/status")
async def plugin_orchestrator_status(plugin_id: str):
    """Get plugin orchestrator status."""
    if not _blade_runner:
        return JSONResponse({"error": "Not initialized"}, status_code=503)
    plugin = _blade_runner.plugins.get(plugin_id)
    if not plugin:
        return JSONResponse({"error": f"Plugin '{plugin_id}' not found"}, status_code=404)
    return plugin.plugin_orchestrator_status()


# ---------------------------------------------------------------------------
# Plugin worker proxy — forwards /api/plugins/{id}/* to worker processes
# ---------------------------------------------------------------------------

# Worker registry: plugin_name → "http://127.0.0.1:{port}"
_plugin_workers = {}


def register_plugin_worker(plugin_name: str, endpoint: str):
    """Register a plugin worker endpoint for proxying."""
    _plugin_workers[plugin_name] = endpoint
    logger.info(f"Plugin worker registered: {plugin_name} → {endpoint}")


@app.api_route("/api/plugins/{plugin_id}/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def plugin_proxy(plugin_id: str, path: str, request: Request):
    """Proxy requests to plugin workers. Users only talk to core (8900)."""
    worker_url = _plugin_workers.get(plugin_id)
    if not worker_url:
        return JSONResponse(
            {"error": f"No worker registered for plugin '{plugin_id}'"},
            status_code=404,
        )

    from core.proxy import forward_request
    return await forward_request(worker_url, path, request)

