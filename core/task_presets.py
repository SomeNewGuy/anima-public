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

"""Task Presets — descriptor constants for model routing.

Plugins pick a preset or compose custom descriptors. The router ONLY sees
TaskDescriptor objects — never task name strings.

Usage:
    from core.task_presets import DEEP_REASONING, with_overrides

    # Use preset directly
    result = router.generate_with_messages(messages, task_desc=DEEP_REASONING)

    # Override for specific needs
    PLANNING = with_overrides(DEEP_REASONING, min_context=32000, typical_tokens=6000)
    result = router.generate_with_messages(messages, task_desc=PLANNING)

Presets are frozen dataclasses — immutable constants. Composition via
with_overrides() creates new instances without mutation.

The router never imports this file. Presets have zero coupling to routing.
"""

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class TaskDescriptor:
    """Describes what a task NEEDS from a model. Router scores against this."""
    task_class: str = "extraction"          # extraction | reasoning | synthesis | generation | evaluation
    min_reasoning: float = 0.1             # minimum reasoning_score (0.0-1.0)
    min_context: int = 4096                # minimum context window (tokens)
    latency: str = "medium"                # low | medium | high
    typical_tokens: int = 1024             # expected response size
    parallel_ok: bool = True               # can run alongside other tasks
    determinism: bool = False              # prefer stable/repeatable output
    label: str | None = None               # observability only — router ignores this
    prefer: str = "speed"                  # speed | quality — affects scoring bias
    pause_only: bool = False               # if no model qualifies, pause (don't fallback)
    safe_fallback: bool = True             # allow any online model as last resort
    thinking: bool = False                 # requires thinking/chain-of-thought model

    def to_dict(self) -> dict:
        return asdict(self)


def with_overrides(base: TaskDescriptor, **kwargs) -> TaskDescriptor:
    """Create a new descriptor from a base with overrides. Never mutates."""
    return TaskDescriptor(**{**base.to_dict(), **kwargs})


# =========================================================================
# CANONICAL PRESETS — keep this list small and universal
# =========================================================================

# ---- FAST / CHEAP ----
# Use for: entity extraction, classification, simple parsing, tagging
FAST_EXTRACTION = TaskDescriptor(
    task_class="extraction",
    min_reasoning=0.1,
    min_context=4096,
    latency="low",
    typical_tokens=300,
    prefer="speed",
    safe_fallback=True,
    label="fast_extraction",
)

FAST_CLASSIFICATION = TaskDescriptor(
    task_class="evaluation",
    min_reasoning=0.2,
    min_context=4096,
    latency="low",
    typical_tokens=200,
    prefer="speed",
    safe_fallback=True,
    label="fast_classification",
)

# ---- BALANCED ----
# Use for: triage, exploration, reinforcement, chat, moderate analysis
BALANCED_REASONING = TaskDescriptor(
    task_class="reasoning",
    min_reasoning=0.4,
    min_context=8192,
    latency="medium",
    typical_tokens=1500,
    prefer="quality",
    label="balanced_reasoning",
)

BALANCED_SYNTHESIS = TaskDescriptor(
    task_class="synthesis",
    min_reasoning=0.6,
    min_context=12000,
    latency="medium",
    typical_tokens=2500,
    prefer="quality",
    label="balanced_synthesis",
)

BALANCED_GENERATION = TaskDescriptor(
    task_class="generation",
    min_reasoning=0.4,
    min_context=8192,
    latency="medium",
    typical_tokens=2000,
    prefer="quality",
    safe_fallback=True,
    label="balanced_generation",
)

# ---- HEAVY / QUALITY ----
# Use for: planning, architecture, hypothesis evaluation, deep analysis
DEEP_REASONING = TaskDescriptor(
    task_class="reasoning",
    min_reasoning=0.7,
    min_context=16000,
    latency="medium",
    typical_tokens=4000,
    parallel_ok=False,
    prefer="quality",
    label="deep_reasoning",
)

LONG_CONTEXT_ANALYSIS = TaskDescriptor(
    task_class="reasoning",
    min_reasoning=0.7,
    min_context=64000,
    latency="high",
    typical_tokens=3000,
    parallel_ok=False,
    prefer="quality",
    label="long_context",
)

HIGH_QUALITY_SYNTHESIS = TaskDescriptor(
    task_class="synthesis",
    min_reasoning=0.7,
    min_context=16000,
    latency="medium",
    typical_tokens=5000,
    parallel_ok=False,
    prefer="quality",
    pause_only=True,
    label="hq_synthesis",
)

# ---- CODE-SPECIFIC ----
# Use for: code generation, review, testing, documentation
CODE_GENERATION = TaskDescriptor(
    task_class="generation",
    min_reasoning=0.5,
    min_context=12000,
    latency="medium",
    typical_tokens=2000,
    prefer="quality",
    label="code_generation",
)

CODE_REVIEW = TaskDescriptor(
    task_class="evaluation",
    min_reasoning=0.7,
    min_context=16000,
    latency="medium",
    typical_tokens=1500,
    determinism=True,
    prefer="quality",
    label="code_review",
)

# ---- CRITICAL / STABLE ----
# Use for: triage decisions, belief validation, approval evaluation
DETERMINISTIC_EVAL = TaskDescriptor(
    task_class="evaluation",
    min_reasoning=0.5,
    min_context=8000,
    latency="medium",
    typical_tokens=1000,
    determinism=True,
    parallel_ok=False,
    prefer="quality",
    label="deterministic_eval",
)


# =========================================================================
# LEGACY COMPAT — map old task name strings to presets
# Used ONLY during migration. Callers should switch to descriptors directly.
# =========================================================================

LEGACY_PRESETS = {
    # Core ANIMA tasks
    "extraction": FAST_EXTRACTION,
    "entities": FAST_EXTRACTION,
    "corpus_matching": FAST_EXTRACTION,
    "triage": BALANCED_REASONING,
    "reinforcement": BALANCED_REASONING,
    "exploration": BALANCED_REASONING,
    "chat": BALANCED_GENERATION,
    "dreams": HIGH_QUALITY_SYNTHESIS,
    "triplet": HIGH_QUALITY_SYNTHESIS,
    "hypothesis": HIGH_QUALITY_SYNTHESIS,
    # Coding plugin tasks
    "planning": DEEP_REASONING,
    "architecture": with_overrides(HIGH_QUALITY_SYNTHESIS, min_context=24000, label="architecture"),
    "code_generation": CODE_GENERATION,
    "code_review": CODE_REVIEW,
    "code_test": with_overrides(FAST_EXTRACTION, min_context=4096, label="code_test"),
    "documentation": with_overrides(BALANCED_GENERATION, min_context=16000, label="documentation"),
    "audit": with_overrides(DEEP_REASONING, min_context=32000, label="audit"),
}


def resolve_task(task) -> TaskDescriptor:
    """Convert a task argument to a TaskDescriptor.

    Accepts:
        - TaskDescriptor object (pass through)
        - str (legacy name → lookup in LEGACY_PRESETS, fallback to FAST_EXTRACTION)
        - dict (construct TaskDescriptor from fields)
        - None (returns FAST_EXTRACTION)

    This is the ONLY place string→descriptor conversion happens.
    Plugin developers should use descriptors directly.
    """
    if isinstance(task, TaskDescriptor):
        return task
    if isinstance(task, dict):
        return TaskDescriptor(**{k: v for k, v in task.items() if k in TaskDescriptor.__dataclass_fields__})
    if isinstance(task, str):
        return LEGACY_PRESETS.get(task, FAST_EXTRACTION)
    return FAST_EXTRACTION
