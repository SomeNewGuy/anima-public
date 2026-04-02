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

"""Task Scheduler — bridges the Rust queue to the Python router.

The scheduler runs a background loop that:
1. Pulls tasks from the Rust TieredQueue (Q0→Q1→Q2)
2. Scores available models via Router V3
3. Dispatches to the best model
4. Writes results back to the queue

Plugins submit work via submit() and poll via get_result().
submit_and_wait() provides blocking backward compat.

Architecture:
    Plugin → submit() → Rust TieredQueue
                              ↓
    Scheduler loop → next_task() → V3 score → dispatch → complete/fail
                              ↓
    Plugin ← get_result() ← result dict
"""

import logging
import threading
import time

logger = logging.getLogger("core.task_scheduler")


class TaskScheduler:
    """Background task dispatcher. One per ANIMA instance."""

    def __init__(self, router, rust_engine=None):
        self.router = router
        self.engine = rust_engine
        self._running = False
        self._thread = None
        self._poll_interval = 0.05  # 50ms
        self._workers = []
        # Dynamic worker count: match actual model capacity
        self._max_workers = self._compute_worker_count()
        self._worker_sem = threading.Semaphore(self._max_workers)
        self._stats = {
            "dispatched": 0,
            "completed": 0,
            "failed": 0,
            "no_model": 0,
        }
        # Python fallback queue (if Rust not available)
        self._py_queue = []
        self._py_results = {}
        self._py_lock = threading.Lock()

    def _compute_worker_count(self):
        """Derive max workers from actual model capacity. Minimum 2."""
        if not self.router or not self.router.models:
            return 4  # safe default before models load
        total = sum(2 if getattr(m, 'parallel', False) else 1
                    for m in self.router.models.values()
                    if getattr(m, 'enabled', True))
        return max(2, total)

    @property
    def has_rust_queue(self):
        return self.engine is not None and hasattr(self.engine, 'queue_next_task')

    def start(self):
        """Start the background dispatch loop."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._dispatch_loop, daemon=True, name="task-scheduler"
        )
        self._thread.start()
        logger.info("Task scheduler started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("Task scheduler stopped")

    def submit(self, messages, task="extraction", priority=2,
               max_tokens=None, temperature=None, timeout=180,
               plugin_id=""):
        """Submit a task. Returns task_id immediately."""
        # Split messages into system prompt + user content
        sys_prompt = ""
        prompt_parts = []
        for m in messages:
            if m["role"] == "system":
                sys_prompt = m["content"]
            else:
                prompt_parts.append(f"{m['role']}: {m['content']}")
        prompt = "\n".join(prompt_parts)

        if self.has_rust_queue:
            task_id = self.engine.queue_submit(
                prompt, task, priority,
                sys_prompt or None,
                max_tokens or 2048,
                temperature or 0.7,
            )
        else:
            import uuid
            task_id = str(uuid.uuid4())[:12]
            with self._py_lock:
                self._py_queue.append({
                    "task_id": task_id,
                    "messages": messages,
                    "task": task,
                    "priority": priority,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "timeout": timeout,
                    "plugin_id": plugin_id,
                    "submitted_at": time.time(),
                })
                self._py_results[task_id] = {"status": "pending"}

        return task_id

    def get_result(self, task_id):
        """Get task result. Returns dict with status/response/error/latency_ms."""
        if self.has_rust_queue:
            return self.engine.queue_result(task_id)
        with self._py_lock:
            return self._py_results.get(task_id, {"status": "unknown"})

    def submit_and_wait(self, messages, task="extraction", priority=2,
                        max_tokens=None, temperature=None, timeout=180,
                        plugin_id=""):
        """Submit and block until result. Drop-in for generate_with_messages()."""
        task_id = self.submit(
            messages, task, priority, max_tokens, temperature, timeout, plugin_id,
        )
        deadline = time.time() + timeout
        while time.time() < deadline:
            result = self.get_result(task_id)
            status = result.get("status", "unknown")
            if status == "complete":
                return result.get("response", "")
            elif status == "error":
                raise RuntimeError(
                    f"Task failed: {result.get('error', 'unknown')}"
                )
            time.sleep(self._poll_interval)

        raise TimeoutError(f"Task {task_id} timed out after {timeout}s")

    def get_status(self):
        """Queue + scheduler health for dashboard."""
        queue_status = {}
        if self.has_rust_queue:
            queue_status = self.engine.queue_status()
        else:
            with self._py_lock:
                queue_status = {"pending": len(self._py_queue)}
        return {
            "running": self._running,
            "queue": queue_status,
            "stats": dict(self._stats),
        }

    # ------------------------------------------------------------------
    # Dispatch loop — the scheduler brain
    # ------------------------------------------------------------------

    def _dispatch_loop(self):
        """Pull tasks and dispatch to models. Multiple concurrent dispatches."""
        while self._running:
            # Try to grab a worker slot (non-blocking)
            if not self._worker_sem.acquire(timeout=self._poll_interval):
                continue  # all workers busy, wait

            task = self._pop_next_task()
            if task is None:
                self._worker_sem.release()
                time.sleep(self._poll_interval)
                continue

            # Dispatch in a worker thread
            t = threading.Thread(
                target=self._dispatch_task, args=(task,),
                daemon=True, name=f"dispatch-{task.get('task_id', '?')[:8]}",
            )
            t.start()

    def _pop_next_task(self):
        """Get next task from Rust or Python queue."""
        if self.has_rust_queue:
            return self.engine.queue_next_task()
        with self._py_lock:
            if not self._py_queue:
                return None
            self._py_queue.sort(key=lambda x: (x["priority"], x["submitted_at"]))
            return self._py_queue.pop(0)

    def _dispatch_task(self, task):
        """Execute a single task — runs in worker thread."""
        try:
            task_id = task.get("task_id", "")
            is_rust = self.has_rust_queue and "task_class" in task

            if is_rust:
                self._dispatch_rust_task(task)
            else:
                self._dispatch_py_task(task)
        finally:
            self._worker_sem.release()

    def _dispatch_rust_task(self, task):
        """Dispatch a task from the Rust queue.

        Sets plugin context so the router's fairness system tracks this
        task against the correct plugin.
        """
        task_id = task["task_id"]
        task_class = task["task_class"]
        prompt = task["prompt"]
        system_prompt = task.get("system_prompt", "")
        plugin_id = task.get("plugin_id", "")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        from core.task_presets import resolve_task
        descriptor = resolve_task(task_class)

        typical = getattr(descriptor, 'typical_tokens', 1024)
        timeout = max(240, typical // 2)

        self._stats["dispatched"] += 1
        start = time.time()

        # Set plugin context for fairness tracking
        from core.router import ModelRouter
        if plugin_id:
            ModelRouter.set_active_plugin(plugin_id)
        try:
            response = self.router.generate_with_messages(
                messages, max_tokens=2048, temperature=0.7,
                timeout=timeout, task_desc=descriptor,
            )
            latency_ms = int((time.time() - start) * 1000)
            self.engine.queue_complete_task(task_id, response, latency_ms)
            self._stats["completed"] += 1
        except Exception as e:
            self.engine.queue_fail_task(task_id, str(e))
            self._stats["failed"] += 1
            logger.warning(f"Task {task_id} failed: {e}")
        finally:
            if plugin_id:
                ModelRouter.clear_active_plugin()

    def _dispatch_py_task(self, task):
        """Dispatch from Python fallback queue.

        Sets plugin context so the router's fairness system tracks this
        task against the correct plugin.
        """
        task_id = task["task_id"]
        messages = task["messages"]
        task_name = task["task"]
        plugin_id = task.get("plugin_id", "")

        self._stats["dispatched"] += 1
        start = time.time()

        from core.router import ModelRouter
        if plugin_id:
            ModelRouter.set_active_plugin(plugin_id)
        try:
            response = self.router.generate_with_messages(
                messages,
                max_tokens=task.get("max_tokens"),
                temperature=task.get("temperature"),
                timeout=task.get("timeout", 180),
                task=task_name,
            )
            latency_ms = int((time.time() - start) * 1000)
            with self._py_lock:
                self._py_results[task_id] = {
                    "status": "complete",
                    "response": response,
                    "latency_ms": latency_ms,
                }
            self._stats["completed"] += 1
        except Exception as e:
            with self._py_lock:
                self._py_results[task_id] = {
                    "status": "error",
                    "error": str(e),
                }
            self._stats["failed"] += 1
            logger.warning(f"Task {task_id} failed: {e}")
        finally:
            if plugin_id:
                ModelRouter.clear_active_plugin()
