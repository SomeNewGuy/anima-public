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


"""ANIMA Plugin — my-plugin.

Replace this docstring with a description of what your plugin does.
Rename the class and update plugin.toml to match.
"""

import logging

from core.plugin_loader import AnimaPlugin

logger = logging.getLogger("plugins.my-plugin")


class MyPlugin(AnimaPlugin):
    """Plugin implementation.

    Override only the methods you need. All defaults are safe no-ops.

    Lifecycle:
        1. register_tables(db_conn)   — called once on engine init
        2. register_endpoints(app)    — called once for FastAPI routes
        3. on_start(engine)           — engine fully ready
        4. tick(engine)               — called each blade cycle (non-blocking)
        5. on_stop(engine)            — shutdown

    Inference contract:
        ALWAYS use engine.inference.request() — never call router directly.

        result = engine.inference.request(
            messages,
            task="extraction",    # TaskDescriptor string or object
            policy="balanced",    # fast | balanced | heavy | critical
        )

        if not result["ok"]:
            # result["reason"]: budget_exceeded | pressure_skip |
            #                   timeout | no_model | error
            return

        response = result["response"]

        Policies:
            fast     — skip if pressure > 0.7 (extraction, tagging)
            balanced — skip if pressure > 0.85 (triage, chat, analysis)
            heavy    — skip if pressure > 0.5 (dreams, synthesis, planning)
            critical — almost always run, degrades at > 0.95 (operator input)
    """

    def register_tables(self, db_conn):
        """Create plugin-specific tables. Called once on engine init."""
        pass

    def register_endpoints(self, app, get_engine):
        """Register FastAPI routes for this plugin."""
        pass

    def get_orchestrator_class(self):
        """Return orchestrator class, or None for no orchestrator.
        If your plugin needs document ingestion, copy the ingestion/
        directory from an existing plugin into your own plugin folder
        and import from there. Never import from another plugin.
        """
        return None

    def on_start(self, engine):
        logger.info("my-plugin started")

    def on_stop(self, engine):
        logger.info("my-plugin stopped")

    def tick(self, engine):
        """One unit of plugin work per blade cycle.

        Must be non-blocking. Do one step and return.

        Example:
            result = engine.inference.request(
                [{"role": "user", "content": "Extract entities from: ..."}],
                task="extraction",
                policy="fast",
            )
            if not result["ok"]:
                return  # system busy or budget hit — try next cycle

            entities = parse(result["response"])
        """
        pass

    def start_orchestrator(self, engine):
        """Start continuous work loop (called from plugin dashboard)."""
        pass

    def stop_orchestrator(self, engine):
        """Stop continuous work loop."""
        pass

    def orchestrator_status(self) -> dict:
        return {"running": False, "cycle": 0, "phase": "idle"}
