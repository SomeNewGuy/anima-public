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

"""Priority work queue for ANIMA router.

Persistent background workers pull tasks by priority and dispatch
through the router. Models stay busy, high priority jumps ahead.

Priority levels:
    0 = CRITICAL  — escalations, blocking architecture decisions
    1 = HIGH      — code review, planning
    2 = NORMAL    — extraction, triage
    3 = BULK      — code generation, test writing

Usage:
    POST /queue/submit  — returns task_id instantly
    GET  /queue/{id}    — poll for result
    GET  /queue/status  — queue depth + active workers
"""

import json
import logging
import re
import time
import uuid
from queue import PriorityQueue, Empty
from threading import Thread, Lock

logger = logging.getLogger("core.queue")


class TaskQueue:
    """Priority queue with persistent background workers."""

    def __init__(self, router, context_provider=None):
        """
        Args:
            router: ModelRouter instance for dispatching
            context_provider: optional callable(query_text) → list of belief strings
                              Used when include_context=True on a submitted task.
        """
        self.router = router
        self.context_provider = context_provider
        self._queue = PriorityQueue()
        self._results = {}
        self._active = {}       # task_id → worker_id
        self._workers = []
        self._running = False
        self._lock = Lock()
        self._submitted = 0
        self._completed = 0
        self._errors = 0

    def submit(self, prompt, system_prompt="", task="extraction",
               priority=2, max_tokens=4096, temperature=0.1,
               include_context=False, context_query="", metadata=None):
        """Submit a task to the queue. Returns task_id immediately."""
        task_id = str(uuid.uuid4())[:12]
        item = {
            "task_id": task_id,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "task": task,
            "priority": priority,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "include_context": include_context,
            "context_query": context_query,
            "metadata": metadata or {},
            "submitted_at": time.time(),
        }

        # PriorityQueue sorts ascending — lower priority number = processed first
        # Timestamp as tiebreaker = FIFO within same priority
        self._queue.put((priority, time.time(), item))

        with self._lock:
            self._results[task_id] = {"status": "pending", "priority": priority}
            self._submitted += 1

        self._ensure_workers()
        logger.info(f"Queue submit: {task_id} task={task} priority={priority}")
        return task_id

    def get_result(self, task_id):
        """Get result for a task. Returns status + response if complete."""
        with self._lock:
            return dict(self._results.get(task_id, {"status": "unknown"}))

    def get_status(self):
        """Queue health: depth, active workers, completion stats."""
        with self._lock:
            pending = self._queue.qsize()
            active = len(self._active)
            return {
                "pending": pending,
                "active": active,
                "workers": len(self._workers),
                "submitted": self._submitted,
                "completed": self._completed,
                "errors": self._errors,
                "running": self._running,
            }

    def cancel(self, task_id):
        """Cancel a pending task. Cannot cancel running tasks."""
        with self._lock:
            result = self._results.get(task_id)
            if result and result.get("status") == "pending":
                self._results[task_id] = {"status": "cancelled"}
                return True
        return False

    def stop(self):
        """Stop all workers. Pending tasks remain in queue."""
        self._running = False
        for w in self._workers:
            w.join(timeout=5)
        self._workers = []
        logger.info("Queue workers stopped")

    def _ensure_workers(self):
        """Start worker threads if not already running."""
        if self._running:
            return
        self._running = True

        count = sum(
            1 for m in self.router.models.values()
            if m.online and m.enabled and m.parallel
        )
        count = max(count, 1)

        for i in range(count):
            t = Thread(target=self._worker_loop, args=(i,), daemon=True)
            t.start()
            self._workers.append(t)

        logger.info(f"Queue started {count} workers")

    def _worker_loop(self, worker_id):
        """Worker pulls highest priority task, dispatches, stores result."""
        while self._running:
            try:
                priority, ts, item = self._queue.get(timeout=5)
            except Empty:
                continue

            task_id = item["task_id"]

            # Check if cancelled while queued
            with self._lock:
                if self._results.get(task_id, {}).get("status") == "cancelled":
                    continue
                self._results[task_id] = {
                    "status": "running",
                    "worker": worker_id,
                    "priority": priority,
                    "started_at": time.time(),
                }
                self._active[task_id] = worker_id

            # Build messages
            system_parts = []
            if item["system_prompt"]:
                system_parts.append(item["system_prompt"])

            # Optional context injection from belief graph
            beliefs_injected = 0
            if item["include_context"] and self.context_provider:
                try:
                    query = item["context_query"] or item["prompt"]
                    beliefs = self.context_provider(query)
                    if beliefs:
                        context_lines = ["[RELEVANT PROJECT KNOWLEDGE — use as context:]"]
                        for b in beliefs[:15]:
                            context_lines.append(f"  {b}")
                        system_parts.append("\n".join(context_lines))
                        beliefs_injected = len(beliefs[:15])
                except Exception as e:
                    logger.debug(f"Context injection failed: {e}")

            messages = []
            if system_parts:
                messages.append({"role": "system", "content": "\n\n".join(system_parts)})
            messages.append({"role": "user", "content": item["prompt"]})

            # Dispatch through router
            start = time.time()
            try:
                response = self.router.generate_with_messages(
                    messages,
                    max_tokens=item["max_tokens"],
                    temperature=item["temperature"],
                    task=item["task"],
                )
                elapsed = time.time() - start

                # Clean response artifacts
                response = re.sub(r'<think>.*?</think>', '', response.strip(), flags=re.DOTALL)
                response = re.sub(r'</?answer>', '', response).strip()
                response = response.replace('/no_think', '').strip()

                with self._lock:
                    self._results[task_id] = {
                        "status": "complete",
                        "response": response,
                        "task": item["task"],
                        "priority": priority,
                        "latency_ms": int(elapsed * 1000),
                        "beliefs_injected": beliefs_injected,
                        "metadata": item.get("metadata", {}),
                    }
                    self._completed += 1

                logger.info(
                    f"Queue complete: {task_id} task={item['task']} "
                    f"priority={priority} elapsed={elapsed:.1f}s worker={worker_id}"
                )

            except Exception as e:
                elapsed = time.time() - start
                with self._lock:
                    self._results[task_id] = {
                        "status": "error",
                        "error": str(e),
                        "task": item["task"],
                        "priority": priority,
                        "latency_ms": int(elapsed * 1000),
                        "metadata": item.get("metadata", {}),
                    }
                    self._errors += 1

                logger.warning(
                    f"Queue error: {task_id} task={item['task']} "
                    f"error={e} worker={worker_id}"
                )

            finally:
                with self._lock:
                    self._active.pop(task_id, None)
