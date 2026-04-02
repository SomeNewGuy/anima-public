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

"""Plugin discovery, loading, and lifecycle management.

Scans the plugins/ directory for plugin.toml manifests, imports each
plugin's Python module, and provides registration hooks for endpoints,
tables, prompts, orchestrators, and lifecycle events.

Usage:
    loader = PluginLoader(plugins_dir="plugins")
    loader.discover()
    plugin = loader.get(mode)         # get plugin by product mode
    plugin.register_endpoints(app)    # FastAPI route registration
    plugin.register_tables(db_conn)   # plugin-specific schema
"""

import importlib
import importlib.util
import logging
import os
import sys

logger = logging.getLogger("core.plugin_loader")

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


def _load_toml(path):
    """Load a TOML file, return dict."""
    if tomllib:
        with open(path, "rb") as f:
            return tomllib.load(f)
    # Fallback: minimal parser for simple plugin.toml files
    data = {}
    current_section = data
    section_name = None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("["):
                section_name = line.strip("[]").strip()
                parts = section_name.split(".")
                current_section = data
                for part in parts:
                    current_section = current_section.setdefault(part, {})
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                # Handle integers
                try:
                    val = int(val)
                except ValueError:
                    pass
                current_section[key] = val
    return data


class AnimaPlugin:
    """Base class for ANIMA plugins.

    Plugins can override any method. The default implementations are no-ops,
    so a minimal plugin only needs to exist — it doesn't have to override
    anything until it's ready.

    Attributes:
        name: Plugin identifier (from plugin.toml)
        version: Semver string
        mode: Product mode string (matches [product] mode in settings.toml)
        plugin_dir: Absolute path to the plugin directory
        manifest: Raw dict from plugin.toml
    """

    def __init__(self, name, version, mode, plugin_dir, manifest):
        self.name = name
        self.version = version
        self.mode = mode
        self.plugin_dir = plugin_dir
        self.manifest = manifest

    def register_tables(self, db_conn):
        """Create plugin-specific tables. Called once on engine init.

        Override to execute CREATE TABLE IF NOT EXISTS statements for
        tables that only this plugin needs (e.g., hypothesis_queue for
        research, task_states for coding).

        Args:
            db_conn: SQLite connection (same as semantic.db_conn)
        """
        pass

    def register_endpoints(self, app, get_engine):
        """Register FastAPI routes for this plugin.

        Override to mount an APIRouter on the app. The get_engine callable
        returns the PersistenceCLI instance for accessing semantic memory,
        inference, config, etc.

        Args:
            app: FastAPI application instance
            get_engine: callable() → PersistenceCLI
        """
        pass

    def get_prompts(self):
        """Return extraction prompts for this plugin's product mode.

        Returns dict with optional keys:
            extraction_prompt: str template for document extraction
            system_prompt: str for system role
            batch_prompt: str template for batch extraction

        Templates may use {filename}, {file_type}, {content}, {max_beliefs}.
        Returns empty dict to use core defaults.
        """
        return {}

    def get_filters(self):
        """Return additional hard-reject patterns for triage filtering.

        Returns dict with optional keys:
            hard_reject_patterns: list[str] — additional patterns to reject
            domain_filter: callable(statement) → bool — True to reject

        Returns empty dict to use core defaults only.
        """
        return {}

    def get_orchestrator_class(self):
        """Return the orchestrator class for this plugin's lifecycle loop.

        Returns a class (not instance) that will be instantiated by the
        engine with (evolution_engine, embeddings, curiosity, config) args.
        Returns None to use the base DMSOrchestrator.
        """
        return None

    def on_start(self, engine):
        """Called after the engine is fully initialized and ready."""
        pass

    def on_stop(self, engine):
        """Called before the engine shuts down."""
        pass

    def on_cycle_complete(self, engine, cycle_number, stats):
        """Called after each orchestrator cycle completes."""
        pass

    def on_belief_added(self, engine, belief_id, belief_data):
        """Called after a new belief is added to the graph."""
        pass

    def on_dream_complete(self, engine, dream_results):
        """Called after a dream synthesis pass completes."""
        pass

    # ------------------------------------------------------------------
    # Plugin orchestrator — controlled by core, not self-managed
    # ------------------------------------------------------------------

    def start_orchestrator(self, engine):
        """Start the plugin's own orchestrator (game building, research
        exploration, coding dispatch). Called by core — not by dashboards.

        Override to initialize plugin-specific work loop.
        The tick() method will be called by the blade runner each cycle.
        """
        pass

    def stop_orchestrator(self, engine):
        """Stop the plugin's orchestrator. Called by core on shutdown or disable."""
        pass

    def orchestrator_status(self) -> dict:
        """Return plugin orchestrator state for dashboard display.

        Returns dict with at minimum:
            running: bool
            cycle: int
            phase: str
        """
        return {"running": False, "cycle": 0, "phase": "idle"}

    def tick(self, engine):
        """Execute one step of plugin-specific work.

        Called by blade runner after each ANIMA lifecycle cycle.
        Must be non-blocking — do one unit of work and return.

        Override to implement: game profile identification, research
        exploration, coding task dispatch, etc.
        """
        pass


class PluginLoader:
    """Discovers and loads plugins from the plugins directory."""

    def __init__(self, plugins_dir=None):
        if plugins_dir is None:
            # Default: plugins/ relative to repo root
            repo_root = os.path.normpath(
                os.path.join(os.path.dirname(__file__), "..")
            )
            plugins_dir = os.path.join(repo_root, "plugins")
        self.plugins_dir = plugins_dir
        self.plugins = {}  # mode → AnimaPlugin instance
        self._by_name = {}  # name → AnimaPlugin instance

    def discover(self):
        """Scan plugins directory for plugin.toml manifests.

        Each subdirectory with a plugin.toml is loaded. The plugin's
        Python module (plugin.py) is imported if it exists, otherwise
        a base AnimaPlugin is instantiated from the manifest alone.
        """
        if not os.path.isdir(self.plugins_dir):
            logger.warning(f"Plugins directory not found: {self.plugins_dir}")
            return

        for entry in sorted(os.scandir(self.plugins_dir), key=lambda e: e.name):
            if not entry.is_dir():
                continue

            manifest_path = os.path.join(entry.path, "plugin.toml")
            if not os.path.exists(manifest_path):
                continue

            try:
                plugin = self._load_plugin(entry.path, manifest_path)
                self.plugins[plugin.mode] = plugin
                self._by_name[plugin.name] = plugin
                logger.info(
                    f"Plugin discovered: {plugin.name} v{plugin.version} "
                    f"(mode={plugin.mode}, dir={entry.name})"
                )
            except Exception as e:
                logger.error(f"Failed to load plugin from {entry.name}: {e}")

        logger.info(
            f"Plugin discovery complete: {len(self.plugins)} plugins "
            f"({', '.join(self._by_name.keys())})"
        )

    def _load_plugin(self, plugin_dir, manifest_path):
        """Load a single plugin from its directory."""
        manifest = _load_toml(manifest_path)
        plugin_meta = manifest.get("plugin", {})

        name = plugin_meta.get("name", os.path.basename(plugin_dir))
        version = plugin_meta.get("version", "0.0.0")
        mode = plugin_meta.get("mode", name)

        # Try to import plugin.py from the plugin directory
        plugin_py = os.path.join(plugin_dir, "plugin.py")
        if os.path.exists(plugin_py):
            # Add plugin dir to sys.path temporarily for imports
            if plugin_dir not in sys.path:
                sys.path.insert(0, plugin_dir)

            spec = importlib.util.spec_from_file_location(
                f"plugins.{name}.plugin", plugin_py
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Look for a class that extends AnimaPlugin
            plugin_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type)
                        and issubclass(attr, AnimaPlugin)
                        and attr is not AnimaPlugin):
                    plugin_class = attr
                    break

            if plugin_class:
                return plugin_class(
                    name=name, version=version, mode=mode,
                    plugin_dir=plugin_dir, manifest=manifest,
                )

        # No plugin.py or no class found — use base
        return AnimaPlugin(
            name=name, version=version, mode=mode,
            plugin_dir=plugin_dir, manifest=manifest,
        )

    def get(self, mode):
        """Get plugin by product mode. Returns None if not found."""
        return self.plugins.get(mode)

    def get_by_name(self, name):
        """Get plugin by name. Returns None if not found."""
        return self._by_name.get(name)

    def all(self):
        """Return all discovered plugins."""
        return list(self.plugins.values())

    def register_all_endpoints(self, app, get_engine):
        """Register endpoints for all discovered plugins on the FastAPI app."""
        for plugin in self.plugins.values():
            try:
                plugin.register_endpoints(app, get_engine)
                logger.info(f"Registered endpoints for plugin: {plugin.name}")
            except Exception as e:
                logger.error(
                    f"Failed to register endpoints for {plugin.name}: {e}"
                )

    def register_all_tables(self, db_conn):
        """Create plugin-specific tables for all discovered plugins."""
        for plugin in self.plugins.values():
            try:
                plugin.register_tables(db_conn)
                logger.debug(f"Registered tables for plugin: {plugin.name}")
            except Exception as e:
                logger.error(
                    f"Failed to register tables for {plugin.name}: {e}"
                )

    def get_prompts(self, mode):
        """Get extraction prompts for a product mode.

        Returns the plugin's prompts if available, empty dict otherwise.
        """
        plugin = self.plugins.get(mode)
        if plugin:
            return plugin.get_prompts()
        return {}

    def get_filters(self, mode):
        """Get additional triage filters for a product mode."""
        plugin = self.plugins.get(mode)
        if plugin:
            return plugin.get_filters()
        return {}

    def get_orchestrator_class(self, mode):
        """Get the orchestrator class for a product mode.

        Returns the class or None (caller falls back to default).
        """
        plugin = self.plugins.get(mode)
        if plugin:
            return plugin.get_orchestrator_class()
        return None

    def fire_hook(self, hook_name, *args, **kwargs):
        """Fire a lifecycle hook on all plugins.

        Args:
            hook_name: Method name on AnimaPlugin (e.g., "on_start")
            *args, **kwargs: Passed to the hook method
        """
        for plugin in self.plugins.values():
            method = getattr(plugin, hook_name, None)
            if method and callable(method):
                try:
                    method(*args, **kwargs)
                except Exception as e:
                    logger.error(
                        f"Hook {hook_name} failed for {plugin.name}: {e}"
                    )
