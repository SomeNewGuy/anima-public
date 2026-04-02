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

"""
Plugin template — copy this folder, rename it, and start building.

Quick start:
  1. cp -r plugins/_template plugins/my-plugin
  2. Edit plugins/my-plugin/plugin.toml (set name, mode, description)
  3. Edit plugins/my-plugin/plugin.py (this file)
  4. ./anima start — your plugin loads automatically

Your plugin gets:
  - Isolated SQLite + ChromaDB (via engine.semantic)
  - Shared model router (via engine.router)
  - Belief system (add, search, link, dream)
  - Curiosity gaps, approval queue
  - Its own dashboard endpoints (mount on FastAPI)

See core/plugin_loader.py for the full AnimaPlugin interface.
"""

from core.plugin_loader import AnimaPlugin


class MyPlugin(AnimaPlugin):
    """Drop-in plugin for ANIMA. All methods are optional — override what you need."""

    def register_tables(self, db_conn):
        """Create plugin-specific tables. Called once on startup.

        Example:
            db_conn.executescript('''
                CREATE TABLE IF NOT EXISTS my_tasks (
                    id TEXT PRIMARY KEY,
                    description TEXT,
                    status TEXT DEFAULT 'pending',
                    created_at TEXT
                );
            ''')
        """
        pass

    def register_endpoints(self, app, get_engine):
        """Mount FastAPI routes for your plugin's API and dashboard.

        Example:
            from fastapi import APIRouter
            router = APIRouter(prefix="/my-plugin", tags=["my-plugin"])

            @router.get("/status")
            async def status():
                engine = get_engine()
                return {"beliefs": engine.semantic.get_belief_count()}

            app.include_router(router)
        """
        pass

    def get_prompts(self):
        """Override extraction prompts for your domain.

        Returns dict with optional keys:
            extraction_prompt: str — how to extract beliefs from documents
            system_prompt: str — system role for conversations
            batch_prompt: str — batch extraction template

        Templates can use: {filename}, {file_type}, {content}, {max_beliefs}
        Return {} to use core defaults.
        """
        return {}

    def get_orchestrator_class(self):
        """Return a custom orchestrator class for your plugin's lifecycle.

        The orchestrator drives your plugin's work loop (scan, ingest, dream, etc.).
        If your plugin needs document ingestion, copy the ingestion/
        directory into your own plugin folder and import from there.
        Never import from another plugin.
        Return None for no orchestrator.
        """
        return None

    def on_start(self, engine):
        """Engine is fully ready. Start any background work here.

        engine.semantic — belief store (add_belief, search_beliefs, etc.)
        engine.router — model inference (generate_with_messages)
        engine.config — your plugin's config dict
        """
        pass

    def on_stop(self, engine):
        """Engine shutting down. Clean up gracefully."""
        pass

    def on_belief_added(self, engine, belief_id, belief_data):
        """Called after a new belief is added. React to new knowledge."""
        pass

    def on_dream_complete(self, engine, dream_results):
        """Called after dream synthesis. React to new connections."""
        pass
