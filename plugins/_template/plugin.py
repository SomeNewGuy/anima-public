# ANIMA Platform
# Copyright (c) 2026 Gerald Teeple
#
# This file is part of ANIMA and is licensed under the
# GNU Affero General Public License v3.0 (AGPL-3.0).
# See the LICENSE file for details.
#
# Commercial licensing available — see NOTICE file.

"""
Plugin template — extend AnimaPlugin and override the methods you need.
See docs/PLUGINS.md for the full plugin development guide.
"""


class MyPlugin:
    """Drop-in plugin for ANIMA. All methods are optional."""

    def register_tables(self, db_conn):
        """Called once on engine init. Create plugin-specific tables."""
        pass

    def register_endpoints(self, app, get_engine):
        """Called once to mount FastAPI routes on the app."""
        pass

    def get_prompts(self):
        """Override extraction and system prompts for this plugin."""
        return {}

    def get_filters(self):
        """Add domain-specific triage filters."""
        return {}

    def get_orchestrator_class(self):
        """Return a custom orchestrator class, or None for the default."""
        return None

    def on_start(self, engine):
        """Engine is fully ready. Start any background work here."""
        pass

    def on_stop(self, engine):
        """Engine shutting down. Clean up gracefully."""
        pass

    def tick(self, engine):
        """Called each cycle after Layer 1 completes. Do plugin work here."""
        pass
