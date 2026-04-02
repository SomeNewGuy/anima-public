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

"""Research project profiles — isolated research contexts on shared infrastructure.

Each profile owns its own database, corpus, config, and hypothesis pipeline.
Profiles share the model pool (router) and engine code.

Single mode (default): one active profile, zero overhead.
Multi mode: round-robin cycle rotation across active profiles.

Usage:
    manager = ProfileManager(base_dir="~/anima_research/projects")
    manager.load_profiles()
    profile = manager.get_active()  # returns current profile context
    manager.switch("alzheimers")    # swap dashboard context
    manager.rotate()                # next profile in round-robin
"""

import json
import logging
import os
import shutil
from datetime import datetime, timezone

logger = logging.getLogger("core.profiles")


class ResearchProfile:
    """A single research project context."""

    def __init__(self, name, base_dir):
        self.name = name
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, "data")
        self.corpus_dir = os.path.join(base_dir, "corpus")
        self.config_path = os.path.join(base_dir, "config", "settings.toml")
        self.mode = "active"  # active | paused | archived
        self.cycle_count = 0
        self.created_at = None
        self.last_cycle_at = None

    def ensure_dirs(self):
        """Create directory structure for this profile."""
        os.makedirs(os.path.join(self.data_dir, "sqlite"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "chroma"), exist_ok=True)
        os.makedirs(os.path.join(self.data_dir, "backups"), exist_ok=True)
        os.makedirs(self.corpus_dir, exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, "config"), exist_ok=True)
        os.makedirs(os.path.join(self.base_dir, "logs"), exist_ok=True)

    def to_dict(self):
        return {
            "name": self.name,
            "base_dir": self.base_dir,
            "data_dir": self.data_dir,
            "corpus_dir": self.corpus_dir,
            "config_path": self.config_path,
            "mode": self.mode,
            "cycle_count": self.cycle_count,
            "created_at": self.created_at,
            "last_cycle_at": self.last_cycle_at,
        }

    @staticmethod
    def from_dict(d):
        p = ResearchProfile(d["name"], d["base_dir"])
        p.mode = d.get("mode", "active")
        p.cycle_count = d.get("cycle_count", 0)
        p.created_at = d.get("created_at")
        p.last_cycle_at = d.get("last_cycle_at")
        return p


class ProfileManager:
    """Manages multiple research profiles on shared infrastructure."""

    def __init__(self, base_dir=None):
        self.base_dir = base_dir or os.path.expanduser("~/anima_research/projects")
        self.profiles = {}  # name -> ResearchProfile
        self.active_name = None
        self._rotation_index = 0
        self._manifest_path = os.path.join(self.base_dir, "profiles.json")

    def load_profiles(self):
        """Load profile manifest from disk."""
        if not os.path.isfile(self._manifest_path):
            return
        try:
            with open(self._manifest_path, "r") as f:
                data = json.load(f)
            for pd in data.get("profiles", []):
                p = ResearchProfile.from_dict(pd)
                self.profiles[p.name] = p
            self.active_name = data.get("active")
            self._rotation_index = data.get("rotation_index", 0)
            logger.info(
                f"Loaded {len(self.profiles)} profiles, "
                f"active: {self.active_name}"
            )
        except Exception as e:
            logger.warning(f"Failed to load profiles: {e}")

    def save_profiles(self):
        """Persist profile manifest to disk."""
        os.makedirs(self.base_dir, exist_ok=True)
        data = {
            "profiles": [p.to_dict() for p in self.profiles.values()],
            "active": self.active_name,
            "rotation_index": self._rotation_index,
        }
        with open(self._manifest_path, "w") as f:
            json.dump(data, f, indent=2)

    def create_profile(self, name, template_config=None):
        """Create a new research profile with empty DB and corpus.

        Args:
            name: Profile name (used as directory name)
            template_config: Path to settings.toml to copy, or None for defaults

        Returns ResearchProfile.
        """
        if name in self.profiles:
            raise ValueError(f"Profile '{name}' already exists")

        safe_name = name.lower().replace(" ", "_").replace("/", "_")
        profile_dir = os.path.join(self.base_dir, safe_name)
        profile = ResearchProfile(name, profile_dir)
        profile.created_at = datetime.now(timezone.utc).isoformat()
        profile.ensure_dirs()

        # Copy or create config
        if template_config and os.path.isfile(template_config):
            shutil.copy2(template_config, profile.config_path)
        else:
            # Create minimal config
            with open(profile.config_path, "w") as f:
                f.write(f'# ANIMA Research Profile: {name}\n')
                f.write(f'# Created: {profile.created_at}\n\n')
                f.write('[model]\ncontext_window = 16384\n\n')
                f.write(f'[memory]\n')
                f.write(f'sqlite_path = "{profile.data_dir}/sqlite/persistence.db"\n')
                f.write(f'chroma_persist_dir = "{profile.data_dir}/chroma"\n\n')
                f.write(f'[extraction]\n')
                f.write(f'datafiles_dir = "{profile.corpus_dir}"\n')

        self.profiles[name] = profile
        if not self.active_name:
            self.active_name = name
        self.save_profiles()

        logger.info(f"Profile created: {name} at {profile_dir}")
        return profile

    def get_active(self):
        """Get the currently active profile."""
        if self.active_name and self.active_name in self.profiles:
            return self.profiles[self.active_name]
        return None

    def switch(self, name):
        """Switch active profile."""
        if name not in self.profiles:
            raise ValueError(f"Profile '{name}' not found")
        self.active_name = name
        self.save_profiles()
        logger.info(f"Switched to profile: {name}")
        return self.profiles[name]

    def get_active_profiles(self):
        """Get all profiles in 'active' mode for rotation."""
        return [p for p in self.profiles.values() if p.mode == "active"]

    def rotate(self):
        """Get next profile in round-robin rotation.

        Returns the next active profile, or None if no active profiles.
        """
        active = self.get_active_profiles()
        if not active:
            return None
        self._rotation_index = self._rotation_index % len(active)
        profile = active[self._rotation_index]
        self._rotation_index = (self._rotation_index + 1) % len(active)
        self.active_name = profile.name
        self.save_profiles()
        return profile

    def archive_profile(self, name):
        """Archive a profile — remove from rotation, preserve data."""
        if name in self.profiles:
            self.profiles[name].mode = "archived"
            if self.active_name == name:
                # Switch to next active
                active = self.get_active_profiles()
                self.active_name = active[0].name if active else None
            self.save_profiles()
            logger.info(f"Profile archived: {name}")

    def pause_profile(self, name):
        """Pause a profile — skip rotation, retain state."""
        if name in self.profiles:
            self.profiles[name].mode = "paused"
            self.save_profiles()

    def resume_profile(self, name):
        """Resume a paused profile."""
        if name in self.profiles:
            self.profiles[name].mode = "active"
            self.save_profiles()

    def list_profiles(self):
        """List all profiles with status."""
        return [p.to_dict() for p in self.profiles.values()]
