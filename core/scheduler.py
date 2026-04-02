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

"""Consolidation Scheduler — decouples consolidation timing from the soak loop.

Supports 4 trigger modes, all configurable via settings.toml [consolidation]:
  - cycle-count: every N exploration cycles (current ANIMA behavior)
  - time-interval: wall clock interval in minutes
  - event-driven: fires on explicit event notification
  - manual: fires only on explicit trigger() call

The consolidation engine (evolution.consolidate) is never modified —
the scheduler only controls *when* the consolidate_fn callable is invoked.
"""

import logging
import time

logger = logging.getLogger("scheduler")


class ConsolidationScheduler:
    """Controls when consolidation fires. Does not own consolidation logic."""

    def __init__(self, config, consolidate_fn, episode_fn=None):
        """
        Args:
            config: Full settings dict (from settings.toml).
            consolidate_fn: Callable that runs consolidation. No args.
            episode_fn: Callable that creates a new episode (for future
                        time-interval/event-driven episode boundary handling).
                        Not used by cycle-count mode.
        """
        self.consolidate_fn = consolidate_fn
        self.episode_fn = episode_fn

        # Read [consolidation] section, fall back to legacy [soak_test] config
        consol_cfg = config.get("consolidation", {})
        if consol_cfg:
            self.mode = consol_cfg.get("mode", "cycle-count")
            self.interval = consol_cfg.get("interval", 4)
            self.events = consol_cfg.get("events", [])
        else:
            # Legacy fallback — read from [soak_test].consolidation_interval
            soak_cfg = config.get("soak_test", {})
            legacy_interval = soak_cfg.get("consolidation_interval", 0)
            if legacy_interval:
                self.mode = "cycle-count"
                self.interval = legacy_interval
                self.events = []
                logger.warning(
                    "Consolidation config: using legacy [soak_test].consolidation_interval=%d. "
                    "Migrate to [consolidation] section in settings.toml.",
                    legacy_interval,
                )
            else:
                # No consolidation config at all — disabled
                self.mode = "manual"
                self.interval = 0
                self.events = []
                logger.info("Consolidation scheduler: no config found, defaulting to manual mode")

        # Time-interval state
        self._last_consolidation_time = time.monotonic()

        # Cycle-count state
        self._cycle_count = 0

        logger.info(
            "Consolidation scheduler initialized: mode=%s, interval=%s, events=%s",
            self.mode, self.interval, self.events,
        )

    def notify_cycle(self, cycle_number=None):
        """Notify the scheduler that an exploration cycle completed.

        For cycle-count mode, fires consolidation every `interval` cycles.
        For time-interval mode, checks the clock.
        For event-driven and manual modes, this is a no-op.

        Args:
            cycle_number: Optional explicit cycle number. If None, uses
                          internal counter.

        Returns:
            True if consolidation fired, False otherwise.
        """
        if self.mode == "cycle-count":
            if cycle_number is not None:
                effective_cycle = cycle_number
            else:
                self._cycle_count += 1
                effective_cycle = self._cycle_count

            if self.interval and effective_cycle % self.interval == 0:
                return self._fire(f"cycle-count (cycle {effective_cycle})")
            return False

        elif self.mode == "time-interval":
            return self.check_timer()

        # event-driven and manual: cycles don't trigger consolidation
        return False

    def check_timer(self):
        """Check if wall-clock interval has elapsed. Fires if so.

        Returns:
            True if consolidation fired, False otherwise.
        """
        if self.mode != "time-interval" or not self.interval:
            return False

        elapsed_minutes = (time.monotonic() - self._last_consolidation_time) / 60
        if elapsed_minutes >= self.interval:
            return self._fire(f"time-interval ({elapsed_minutes:.1f}min elapsed)")
        return False

    def notify_event(self, event_type):
        """Notify the scheduler of a named event.

        For event-driven mode, fires consolidation if event_type matches
        the configured events list.

        Args:
            event_type: String event name (e.g. "batch_complete", "seed_complete").

        Returns:
            True if consolidation fired, False otherwise.
        """
        if self.mode != "event-driven":
            return False

        if event_type in self.events:
            return self._fire(f"event-driven ({event_type})")
        return False

    def trigger(self):
        """Explicitly trigger consolidation. Always fires regardless of mode.

        Returns:
            True if consolidation fired, False on error.
        """
        return self._fire("manual trigger")

    def should_consolidate(self, cycle_number=None):
        """Query whether consolidation would fire, without side effects.

        Returns:
            True if the next notify_cycle/check_timer/trigger would fire.
        """
        if self.mode == "cycle-count":
            if cycle_number is not None:
                return bool(self.interval and cycle_number % self.interval == 0)
            next_cycle = self._cycle_count + 1
            return bool(self.interval and next_cycle % self.interval == 0)

        elif self.mode == "time-interval":
            if not self.interval:
                return False
            elapsed_minutes = (time.monotonic() - self._last_consolidation_time) / 60
            return elapsed_minutes >= self.interval

        elif self.mode == "manual":
            return True  # manual always fires when triggered

        return False

    def _fire(self, reason):
        """Execute consolidation via the provided callable.

        Returns:
            True if consolidation completed, False on error.
        """
        logger.info("Consolidation scheduler firing: %s", reason)
        try:
            self.consolidate_fn()
            self._last_consolidation_time = time.monotonic()
            return True
        except Exception as e:
            logger.error("Consolidation scheduler: consolidate_fn failed: %s", e)
            return False
