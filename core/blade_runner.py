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

"""Blade Runner — multi-plugin orchestration on a single ANIMA core.

Plugins are "blades" slotted into the engine. The blade runner cycles
through enabled plugins, giving each a full cycle before moving to
the next. One router, one model pool, one port.

Usage:
    runner = BladeRunner(core_config)
    runner.load_plugins()       # discover and initialize all plugins
    runner.start()              # begin round-robin cycling
    runner.stop()               # graceful shutdown

Architecture:
    BladeRunner
    ├── ModelRouter (shared — ONE router for all plugins)
    ├── EmbeddingEngine (shared)
    ├── PluginInstance (per plugin)
    │   ├── SemanticMemory (own DB)
    │   ├── EpisodicMemory (own DB)
    │   ├── EvolutionEngine (own)
    │   ├── Orchestrator (own)
    │   └── CuriosityMemory (own)
    └── CycleManager
        └── round-robin or single-blade mode
"""

import logging
import os
import threading
import time
from pathlib import Path

import toml

logger = logging.getLogger("blade_runner")


class SafeInferenceProxy:
    """Plugin inference proxy — never crashes, tracks everything.

    Every plugin's self.inference points here. All requests go through
    to the router which queues and serves them. No requests are skipped
    or rejected — the router's _acquire_engine blocks until a model is
    available. Scoring handles fairness.

    Plugins call self.inference.request() for structured results, or
    self.inference.generate_with_messages() for legacy string returns.
    """

    # Reason codes for structured results
    OK = "ok"
    TIMEOUT = "timeout"
    NO_MODEL = "no_model"
    ERROR = "error"

    STOPPED = "stopped"

    def __init__(self, router, plugin_instance):
        self._router = router
        self._plugin = plugin_instance
        self._request_counter = 0
        self._killed = False  # kill switch — stops all inference immediately

    def __getattr__(self, name):
        return getattr(self._router, name)

    def _make_result(self, ok, response="", reason="ok", latency_ms=0,
                     request_id="", model=""):
        return {
            "ok": ok,
            "response": response,
            "reason": reason,
            "latency_ms": latency_ms,
            "id": request_id,
            "model": model,
        }

    def _next_id(self):
        self._request_counter += 1
        return f"{self._plugin.name}:{self._request_counter}"

    def request(self, messages, task=None, task_desc=None,
                max_tokens=None, temperature=None, timeout=600):
        """Send inference request. Blocks until served. Never throws.

        The router queues this request and assigns it to the best
        available model when capacity exists. No skipping, no rejecting.

        Returns:
            {"ok": bool, "response": str, "reason": str,
             "latency_ms": int, "id": str, "model": str}
        """
        rid = self._next_id()
        pi = self._plugin

        # Kill switch — immediately refuse all inference when stopped
        if self._killed:
            return self._make_result(False, reason=self.STOPPED, request_id=rid)

        pi._cycle_task_count += 1
        start = time.time()

        try:
            response = self._router.generate_with_messages(
                messages, max_tokens=max_tokens, temperature=temperature,
                timeout=timeout, task=task, task_desc=task_desc,
            )
            latency = int((time.time() - start) * 1000)

            if response is None:
                return self._make_result(False, reason=self.NO_MODEL,
                                         latency_ms=latency, request_id=rid)

            return self._make_result(True, response=response, latency_ms=latency,
                                     request_id=rid)

        except RuntimeError as e:
            latency = int((time.time() - start) * 1000)
            if "No models online" in str(e):
                logger.warning(f"[{rid}] no models online: {e}")
                return self._make_result(False, reason=self.NO_MODEL,
                                         latency_ms=latency, request_id=rid)
            logger.warning(f"[{rid}] error: {e}")
            return self._make_result(False, reason=self.ERROR,
                                     latency_ms=latency, request_id=rid)
        except Exception as e:
            latency = int((time.time() - start) * 1000)
            logger.warning(f"[{rid}] error: {e}")
            return self._make_result(False, reason=self.ERROR,
                                     latency_ms=latency, request_id=rid)

    def generate_with_messages(self, messages, **kwargs):
        """Legacy interface. Blocks until served. Returns string.

        All existing code that calls self.inference.generate_with_messages()
        goes through the router's queue. Returns empty string only on
        real errors (all models offline).
        """
        task = kwargs.pop("task", None)
        task_desc = kwargs.pop("task_desc", None)
        result = self.request(messages, task=task, task_desc=task_desc, **kwargs)
        if not result["ok"]:
            logger.info(f"[{result['id']}] inference failed: {result['reason']}")
        return result["response"]

    def generate(self, prompt, system_context="", max_tokens=None, task=None):
        """Legacy generate. Blocks until served. Returns string."""
        messages = []
        if system_context:
            messages.append({"role": "system", "content": system_context})
        messages.append({"role": "user", "content": prompt})
        result = self.request(messages, task=task, max_tokens=max_tokens)
        return result["response"]


class PluginInstance:
    """One loaded plugin with its own memory systems and orchestrator."""

    def __init__(self, name, plugin_obj, config, data_dir, shared_router, shared_embeddings):
        self.name = name
        self.plugin = plugin_obj
        self.config = config
        self.data_dir = data_dir
        self.enabled = True
        self._running = False

        # Shared resources
        self.router = shared_router
        self.embeddings = shared_embeddings

        # Per-plugin resources (initialized in load())
        self.semantic = None
        self.episodic = None
        self.reflective = None
        self.curiosity = None
        self.evolution = None
        self.orchestrator = None
        self.inference = SafeInferenceProxy(shared_router, self)  # budget-enforced
        self.retrieval = None
        self.ner = None
        self.scheduler = None
        self.explorations = None
        self.exploration_engine = None

        # Chat state (per-plugin)
        self.conversation_history = []
        self.current_episode_id = None
        self.last_turn_id = None
        self.turn_count = 0
        self._sleeping = False
        self._web_mode = True

        # Budget enforcement — per-plugin task limits
        self._cycle_task_count = 0
        self._max_tasks_per_cycle = config.get("blade", {}).get("max_tasks_per_cycle", 50)
        self._max_inflight = config.get("blade", {}).get("max_inflight", 8)

    def safe_generate(self, messages, **kwargs):
        """Inference that never throws. Returns None on failure.

        This is the ONLY inference path plugins should use. It:
        - Checks budget before dispatching
        - Catches all exceptions (timeout, model unavailable, etc.)
        - Tracks failures for observability
        - Returns None on any error — caller must handle None
        """
        # Budget check — reject if plugin is over its cycle limit
        if self._cycle_task_count >= self._max_tasks_per_cycle:
            logger.info(f"Plugin '{self.name}' hit task cap ({self._max_tasks_per_cycle}/cycle)")
            return None

        # Inflight check
        inflight = self.router._plugin_inflight.get(self.name, 0) if self.router else 0
        if inflight >= self._max_inflight:
            logger.info(f"Plugin '{self.name}' at inflight cap ({self._max_inflight})")
            return None

        try:
            self._cycle_task_count += 1
            return self.router.generate_with_messages(messages, **kwargs)
        except Exception as e:
            logger.warning(f"Plugin '{self.name}' inference failed: {e}")
            return None

    def get_pressure(self):
        """Check system pressure. Plugins should call this before heavy work.

        Returns float 0.0 (idle) to 1.0 (saturated).
        """
        if self.router and hasattr(self.router, '_system_pressure'):
            return self.router._system_pressure()
        return 0.0

    def load(self):
        """Initialize per-plugin memory systems."""
        from memory.episodic import EpisodicMemory
        from memory.curiosity import CuriosityMemory
        from memory.reflective import ReflectiveMemory
        from reflection.evolution import EvolutionEngine

        from memory.semantic_rust import SemanticMemory

        # Set data dir for this plugin
        os.environ["ANIMA_DATA_DIR"] = str(self.data_dir)

        # Initialize memory systems with plugin's own DBs
        self.semantic = SemanticMemory(self.config)
        self.semantic.initialize()
        self.episodic = EpisodicMemory(self.config)
        self.episodic.initialize()
        self.reflective = ReflectiveMemory(self.config)
        self.reflective.initialize()
        self.curiosity = CuriosityMemory(self.config)
        self.curiosity.initialize()

        # Evolution engine — uses shared router for inference
        self.evolution = EvolutionEngine(
            self.config, self.router, self.embeddings,
            self.episodic, self.semantic, self.reflective,
            curiosity=self.curiosity,
        )

        # Register plugin tables
        if hasattr(self.plugin, "register_tables"):
            self.plugin.register_tables(self.semantic.db_conn)

        # Get orchestrator
        orch_class = None
        if hasattr(self.plugin, "get_orchestrator_class"):
            orch_class = self.plugin.get_orchestrator_class()

        if orch_class:
            # Orchestrator needs different args depending on type
            import inspect
            params = inspect.signature(orch_class.__init__).parameters
            if "exploration_engine" in params:
                self.orchestrator = orch_class(
                    self.evolution, self.embeddings,
                    self.curiosity, None, self.config,
                )
            else:
                self.orchestrator = orch_class(
                    self.evolution, self.embeddings,
                    self.curiosity, self.config,
                )

        # Fire on_start
        if hasattr(self.plugin, "on_start"):
            self.plugin.on_start(self)

        logger.info(f"Plugin '{self.name}' loaded — data: {self.data_dir}")

    def run_cycle(self):
        """Run one full cycle: ANIMA lifecycle + plugin tick."""
        if not self.orchestrator:
            logger.warning(f"Plugin '{self.name}' has no orchestrator — skipping cycle")
            return {}

        try:
            self._running = True
            self._cycle_task_count = 0  # reset budget each cycle
            # Fair scheduling — tell the router which plugin is dispatching
            from core.router import ModelRouter
            ModelRouter.set_active_plugin(self.name)

            logger.info(f"=== Blade '{self.name}' — cycle starting ===")

            # Layer 1: ANIMA lifecycle (corpus, dreams, edges)
            stats = self.orchestrator.run_cycle()

            # Layer 2: Plugin-specific work (game building, research exploration, etc.)
            if self._orch_running and hasattr(self.plugin, "tick"):
                try:
                    self.plugin.tick(self)
                except Exception as e:
                    logger.warning(f"Plugin '{self.name}' tick failed: {e}")

            logger.info(f"=== Blade '{self.name}' — cycle complete ===")

            if hasattr(self.plugin, "on_cycle_complete"):
                cycle_num = getattr(self.orchestrator, "_cycle_count", 0)
                self.plugin.on_cycle_complete(self, cycle_num, stats)

            return stats
        except Exception as e:
            logger.error(f"Blade '{self.name}' cycle failed: {e}")
            return {"error": str(e)}
        finally:
            self._running = False
            from core.router import ModelRouter
            ModelRouter.clear_active_plugin()

    # Plugin orchestrator control (Layer 2)
    _orch_running = False

    def start_plugin_orchestrator(self):
        """Start the plugin's own orchestrator (Layer 2)."""
        if self._orch_running:
            return
        self._orch_running = True
        # Re-enable inference (kill switch cleared)
        if hasattr(self, 'inference') and hasattr(self.inference, '_killed'):
            self.inference._killed = False
        if hasattr(self.plugin, "start_orchestrator"):
            self.plugin.start_orchestrator(self)
        logger.info(f"Plugin '{self.name}' orchestrator started")

    def stop_plugin_orchestrator(self):
        """Stop the plugin's orchestrator. Kill switch stops all inference immediately."""
        self._orch_running = False
        # Kill switch — any in-flight or queued inference returns immediately
        if hasattr(self, 'inference') and hasattr(self.inference, '_killed'):
            self.inference._killed = True
        if hasattr(self.plugin, "stop_orchestrator"):
            self.plugin.stop_orchestrator(self)
        logger.info(f"Plugin '{self.name}' orchestrator stopped — inference killed")

    def plugin_orchestrator_status(self):
        """Get plugin orchestrator status."""
        if hasattr(self.plugin, "orchestrator_status"):
            status = self.plugin.orchestrator_status()
            status["running"] = self._orch_running
            return status
        return {"running": self._orch_running, "cycle": 0, "phase": "idle"}

    @property
    def status(self):
        beliefs = 0
        edges = 0
        try:
            if self.semantic:
                beliefs = self.semantic.get_belief_count()
                row = self.semantic.db_conn.execute(
                    "SELECT COUNT(*) FROM belief_links WHERE COALESCE(active,1)=1"
                ).fetchone()
                edges = row[0] if row else 0
        except Exception:
            pass

        return {
            "name": self.name,
            "enabled": self.enabled,
            "running": self._running,
            "beliefs": beliefs,
            "edges": edges,
            "has_orchestrator": self.orchestrator is not None,
            "plugin_orchestrator": self.plugin_orchestrator_status(),
        }

    def shutdown(self):
        if hasattr(self.plugin, "on_stop"):
            self.plugin.on_stop(self)
        if self.semantic:
            self.semantic.close()


class BladeRunner:
    """Manages multiple plugin instances on a single ANIMA core."""

    def __init__(self, core_config_path=None):
        # Load core config
        if core_config_path:
            self.core_config = toml.load(core_config_path)
        else:
            config_path = os.environ.get(
                "ANIMA_CONFIG",
                os.path.join(os.path.dirname(__file__), "..", "config", "settings.toml"),
            )
            self.core_config = toml.load(os.path.normpath(config_path))

        self.plugins = {}  # name -> PluginInstance
        self.router = None
        self.embeddings = None
        self.scheduler = None

    def initialize_shared(self):
        """Initialize shared resources — router and embeddings."""
        from core.router import ModelRouter
        from core.embeddings import EmbeddingEngine

        # One router for all plugins
        self.router = ModelRouter(self.core_config)
        if self.router.models:
            self.router.load()
            model_names = [m.name for m in self.router.models.values()]
            logger.info(f"Router: {len(model_names)} models — {', '.join(model_names)}")
        else:
            logger.warning("No models configured")

        # One embedding engine for all plugins
        self.embeddings = EmbeddingEngine(self.core_config)
        self.embeddings.load()
        logger.info("Embeddings loaded")

        # Task scheduler — bridges queue to router
        from core.task_scheduler import TaskScheduler
        # Try to get Rust engine for queue support
        rust_engine = None
        try:
            import anima_core
            rust_engine = anima_core.Engine({"memory": {"sqlite_path": ":memory:"}})
            logger.info("Task scheduler using Rust queue (Q0/Q1/Q2)")
        except Exception:
            logger.info("Task scheduler using Python fallback queue")
        self.scheduler = TaskScheduler(self.router, rust_engine)
        self.scheduler.start()
        # Wire scheduler into router so generate_with_messages uses queue
        self.router.scheduler = self.scheduler

    def load_plugins(self, only=None):
        """Discover and load all plugins (or specific ones).

        Args:
            only: list of plugin names to load, or None for all
        """
        from core.plugin_loader import PluginLoader

        plugin_dir = os.path.join(os.path.dirname(__file__), "..", "plugins")
        loader = PluginLoader(plugin_dir)
        loader.discover()

        for plugin_obj in loader.all():
            name = plugin_obj.name

            if only and name not in only:
                continue

            # Build plugin config — merge core config with plugin-specific
            plugin_config_path = os.path.join(
                plugin_obj.plugin_dir, "config", "settings.toml"
            )
            if os.path.isfile(plugin_config_path):
                plugin_config = toml.load(plugin_config_path)
            else:
                plugin_config = dict(self.core_config)

            # Core globals — ALWAYS from core config, plugins never override
            CORE_GLOBALS = ["models", "routing", "hardware", "web",
                            "consolidation", "soak_test"]
            for section in CORE_GLOBALS:
                if section in self.core_config:
                    plugin_config[section] = self.core_config[section]

            # Tasks: plugin defines its own, core fills gaps
            if "tasks" not in plugin_config and "tasks" in self.core_config:
                plugin_config["tasks"] = self.core_config["tasks"]

            # Retrieval weights: core provides defaults, plugin can override
            if "retrieval" in self.core_config and "retrieval" not in plugin_config:
                plugin_config["retrieval"] = self.core_config["retrieval"]

            # Plugin data directory — MUST be under plugin dir, never external
            data_dir = Path(plugin_obj.plugin_dir) / "data"
            data_dir.mkdir(parents=True, exist_ok=True)

            # SAFETY: force DB paths to be relative (under plugin data dir)
            # Never allow absolute paths that point outside the plugin
            memory_cfg = plugin_config.get("memory", {})
            for key in ("sqlite_path", "chroma_persist_dir"):
                val = memory_cfg.get(key, "")
                if val and os.path.isabs(val):
                    logger.warning(
                        f"Plugin '{name}': absolute {key} rejected: {val} "
                        f"— using plugin data dir instead"
                    )
                    if "sqlite" in key:
                        memory_cfg[key] = "data/sqlite/persistence.db"
                    else:
                        memory_cfg[key] = "data/chroma"
            plugin_config["memory"] = memory_cfg

            # Governance — resolve preset to capability booleans, discard preset
            _GOV_PRESETS = {
                "monarchy": {"allow_auto_accept": False, "allow_auto_corrections": False, "allow_auto_lessons": False},
                "llm":      {"allow_auto_accept": True,  "allow_auto_corrections": False, "allow_auto_lessons": False},
                "auto":     {"allow_auto_accept": True,  "allow_auto_corrections": True,  "allow_auto_lessons": True},
            }
            gov_manifest = plugin_obj.manifest.get("governance", {})
            gov_preset = gov_manifest.get("preset", "monarchy")
            if gov_preset not in _GOV_PRESETS:
                logger.warning(
                    f"Plugin '{name}': invalid governance preset '{gov_preset}', "
                    f"defaulting to monarchy"
                )
                gov_preset = "monarchy"
            gov = dict(_GOV_PRESETS[gov_preset])
            # Explicit overrides take precedence over preset
            for gov_key in ("allow_auto_accept", "allow_auto_corrections", "allow_auto_lessons"):
                if gov_key in gov_manifest:
                    gov[gov_key] = bool(gov_manifest[gov_key])
            plugin_config["_governance"] = gov
            logger.info(
                f"Plugin '{name}': governance resolved — "
                f"auto_accept={gov['allow_auto_accept']}, "
                f"auto_corrections={gov['allow_auto_corrections']}, "
                f"auto_lessons={gov['allow_auto_lessons']}"
            )

            instance = PluginInstance(
                name=name,
                plugin_obj=plugin_obj,
                config=plugin_config,
                data_dir=data_dir,
                shared_router=self.router,
                shared_embeddings=self.embeddings,
            )
            instance.load()
            # All plugins are enabled. Drop in and load, period.
            instance.enabled = True
            self.plugins[name] = instance

        # Merge all plugin task definitions into the shared router
        # so cross-plugin dispatches (e.g. coding planning via game config) work
        for name, instance in self.plugins.items():
            plugin_tasks = instance.config.get("tasks", {})
            for task_name, task_cfg in plugin_tasks.items():
                if isinstance(task_cfg, dict) and task_name not in self.router.tasks:
                    self.router.tasks[task_name] = task_cfg
                    logger.info(f"Task '{task_name}' registered from plugin '{name}'")

        logger.info(f"Loaded {len(self.plugins)} plugins: {list(self.plugins.keys())}")
        return self.plugins

    def enable_plugin(self, name):
        if name in self.plugins:
            self.plugins[name].enabled = True
            self._persist_blade_state(name, True)

    # Cycling, enable/disable, blade state persistence all removed.
    # ANIMA loads plugins and provides routing. Plugins control themselves.
    # Plugins control their own start/stop/cycles from their dashboards.

    @property
    def status(self):
        return {
            "plugins": {
                name: p.status for name, p in self.plugins.items()
            },
            "models": len(self.router.models) if self.router else 0,
        }

    def shutdown(self):
        if self.scheduler:
            self.scheduler.stop()
        for name, plugin in self.plugins.items():
            plugin.shutdown()
        logger.info("All plugins shut down")
