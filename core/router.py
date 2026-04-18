"""Multi-model routing — distributes inference tasks across available LLM endpoints.

Routes by capability vectors: models have measured scores (reasoning, speed,
stability, creativity). Tasks declare requirements via TaskDescriptor.
Router scores all candidates and picks the best match.

NO roles. NO tiers in routing logic. Tiers exist only as UI labels.

Solo mode: single model, zero overhead, backward compatible.
Multi mode: V3 scoring → dispatch per task type.

Governance principle: parallelize volume work, serialize judgment work.
Pause judgment tasks if quality model unavailable — never substitute unsafely.
"""

import logging
import threading
import time
from datetime import datetime, timezone

import requests

logger = logging.getLogger("core.router")

# Thread-local storage for tracking which plugin is dispatching
_active_plugin = threading.local()

# ---------------------------------------------------------------------------
# Task system — purely plugin-driven via [tasks.*] in settings.toml
#
# The engine defines NO tasks. Plugins define ALL tasks. Each task has:
#   task_class: extraction/reasoning/synthesis/generation/evaluation
#   min_reasoning: 0.0-1.0 — minimum reasoning_score
#   min_context: minimum context window (optional)
#   prefer: quality/speed
#   pause_only: bool — don't fallback, wait for right model
#   safe_fallback: bool — use any online model if none matched
#
# If a task isn't in the config, the router uses safe defaults.
# ---------------------------------------------------------------------------

# Legacy reasoning→min_reasoning mapping (for old configs during migration)
LEGACY_REASONING_MAP = {"high": 0.7, "medium": 0.4, "low": 0.1}

# Legacy task_class inference from task name (for old configs)
LEGACY_TASK_CLASS = {
    "extraction": "extraction", "entities": "extraction", "corpus_matching": "extraction",
    "code_test": "extraction",
    "triage": "reasoning", "reinforcement": "reasoning", "exploration": "reasoning",
    "code_review": "reasoning", "planning": "reasoning", "audit": "reasoning",
    "dreams": "synthesis", "triplet": "synthesis", "hypothesis": "synthesis",
    "architecture": "synthesis",
    "chat": "generation", "code_generation": "generation", "documentation": "generation",
}

# ---------------------------------------------------------------------------
# Known model registry — pre-populated capability profiles.
# API models get these directly (no probing). Local models start with
# registry values and refine via calibration.
# ---------------------------------------------------------------------------
KNOWN_MODELS = {
    # Anthropic
    "claude-opus": {
        "reasoning_score": 0.95, "instruction_score": 0.95, "stability_score": 0.95,
        "creativity_score": 0.90, "tokens_per_sec": 30.0, "latency_ms": 2000,
        "cost_per_token": 0.000075, "context_max": 200000,
    },
    "claude-sonnet": {
        "reasoning_score": 0.85, "instruction_score": 0.90, "stability_score": 0.90,
        "creativity_score": 0.80, "tokens_per_sec": 60.0, "latency_ms": 1000,
        "cost_per_token": 0.000015, "context_max": 200000,
    },
    "claude-haiku": {
        "reasoning_score": 0.60, "instruction_score": 0.80, "stability_score": 0.85,
        "creativity_score": 0.60, "tokens_per_sec": 100.0, "latency_ms": 500,
        "cost_per_token": 0.000005, "context_max": 200000,
    },
    # OpenAI
    "gpt-4o": {
        "reasoning_score": 0.85, "instruction_score": 0.85, "stability_score": 0.85,
        "creativity_score": 0.80, "tokens_per_sec": 80.0, "latency_ms": 800,
        "cost_per_token": 0.000025, "context_max": 128000,
    },
    "gpt-4o-mini": {
        "reasoning_score": 0.55, "instruction_score": 0.75, "stability_score": 0.80,
        "creativity_score": 0.55, "tokens_per_sec": 120.0, "latency_ms": 400,
        "cost_per_token": 0.0000075, "context_max": 128000,
    },
    # Local — Hunyuan 80B (soak-validated)
    "hunyuan": {
        "reasoning_score": 0.80, "instruction_score": 0.75, "stability_score": 0.75,
        "creativity_score": 0.70, "tokens_per_sec": 10.8, "latency_ms": 3000,
        "cost_per_token": 0.0, "context_max": 32768,
    },
    # Local — GPT OSS 20B
    "gpt-oss-20b": {
        "reasoning_score": 0.55, "instruction_score": 0.65, "stability_score": 0.70,
        "creativity_score": 0.50, "tokens_per_sec": 23.7, "latency_ms": 1500,
        "cost_per_token": 0.0, "context_max": 131072,
    },
    # Local — Qwen Coder 14B
    "qwen-coder-14b": {
        "reasoning_score": 0.45, "instruction_score": 0.70, "stability_score": 0.65,
        "creativity_score": 0.40, "tokens_per_sec": 42.6, "latency_ms": 800,
        "cost_per_token": 0.0, "context_max": 28672,
    },
}


def _lookup_known_model(name, claude_model=None):
    """Try to match a model name to a known registry profile."""
    name_lower = (name or "").lower()
    claude_lower = (claude_model or "").lower()

    # Anthropic models — match by claude_model or name
    if "opus" in claude_lower or "opus" in name_lower:
        return KNOWN_MODELS["claude-opus"]
    if "sonnet" in claude_lower or "sonnet" in name_lower:
        return KNOWN_MODELS["claude-sonnet"]
    if "haiku" in claude_lower or "haiku" in name_lower:
        return KNOWN_MODELS["claude-haiku"]

    # OpenAI
    if "gpt-4o-mini" in name_lower:
        return KNOWN_MODELS["gpt-4o-mini"]
    if "gpt-4o" in name_lower or "gpt-4" in name_lower:
        return KNOWN_MODELS["gpt-4o"]

    # Local models
    if "hunyuan" in name_lower:
        return KNOWN_MODELS["hunyuan"]
    if "gpt" in name_lower and "oss" in name_lower:
        return KNOWN_MODELS["gpt-oss-20b"]
    if "qwen" in name_lower and ("coder" in name_lower or "14b" in name_lower):
        return KNOWN_MODELS["qwen-coder-14b"]

    return None


def _tier_label(reasoning_score):
    """Compute a UI-only tier label from reasoning score. NOT used in routing."""
    if reasoning_score >= 0.7:
        return "large"
    elif reasoning_score >= 0.4:
        return "medium"
    return "small"


class ModelInfo:
    """Represents a configured LLM endpoint with capability vectors."""

    def __init__(self, name, endpoint, context_window=8192,
                 thinking_prefix=False, enabled=True,
                 backend="llama-server", api_key_env=None, claude_model=None,
                 parallel=True,
                 # Capability vectors (0.0-1.0 unless noted)
                 reasoning_score=0.5, instruction_score=0.5,
                 stability_score=0.5, creativity_score=0.5,
                 tokens_per_sec=0.0, latency_ms=2000,
                 cost_per_token=0.0,
                 # Legacy compat — kept for config read, computed to tier label
                 tier=None):
        self.name = name
        self.endpoint = endpoint
        self.context_window = context_window
        self.thinking_prefix = thinking_prefix
        self.enabled = enabled
        self.parallel = parallel
        self.backend = backend
        self.api_key_env = api_key_env
        self.claude_model = claude_model
        self.thinking_tasks = set()

        # Capability vectors — the ONLY thing routing uses
        self.reasoning_score = reasoning_score
        self.instruction_score = instruction_score
        self.stability_score = stability_score
        self.creativity_score = creativity_score
        self.cost_per_token = cost_per_token
        self.latency_ms = latency_ms

        # UI-only tier label (NOT used in routing)
        self.tier = tier or _tier_label(reasoning_score)

        # Runtime state
        self.online = False
        self.busy = False
        self.last_check = None
        self.last_latency_ms = 0
        self.error_count = 0
        self.consecutive_timeouts = 0  # reset on any success; N in a row = frozen
        self.tokens_per_second = tokens_per_sec  # calibrated generation speed
        self.recent_success_rate = 1.0  # rolling average

    def accepts_task(self, task):
        """Capability-based check: can this model handle this task class?

        Always returns True for enabled+online models — the V3 scorer
        handles the actual ranking. This method exists for backward
        compat with code that calls it as a boolean filter.
        """
        return self.enabled

    def get_profile(self):
        """Return capability vector as a dict (replaces old SUBSTRATE_PROFILES)."""
        return {
            "reasoning_score": self.reasoning_score,
            "instruction_score": self.instruction_score,
            "stability_score": self.stability_score,
            "creativity_score": self.creativity_score,
            "tokens_per_sec": self.tokens_per_second,
            "latency_ms": self.latency_ms,
            "cost_per_token": self.cost_per_token,
            "context_max": self.context_window,
        }

    def to_dict(self):
        return {
            "name": self.name,
            "endpoint": self.endpoint,
            "tier": self.tier,
            "context_window": self.context_window,
            "thinking_prefix": self.thinking_prefix,
            "backend": self.backend,
            "enabled": self.enabled,
            "parallel": self.parallel,
            "online": self.online,
            "busy": self.busy,
            "last_check": self.last_check,
            "last_latency_ms": self.last_latency_ms,
            "error_count": self.error_count,
            "tokens_per_second": self.tokens_per_second,
            "api_key_env": self.api_key_env,
            "claude_model": self.claude_model,
            "profile": self.get_profile(),
            "recent_success_rate": self.recent_success_rate,
        }

    @staticmethod
    def from_config(name, cfg):
        """Create ModelInfo from a config dict (settings.toml [models.X] section).

        Supports both new capability vector format and legacy tier/roles format.
        If capability scores aren't specified, tries known model registry,
        then falls back to estimates from tier.
        """
        model_name = cfg.get("name", name)
        claude_model = cfg.get("claude_model")

        # Try known model registry first
        known = _lookup_known_model(model_name, claude_model)

        # Legacy tier → estimate scores if no explicit capabilities
        legacy_tier = cfg.get("tier", "medium")
        tier_defaults = {
            "large":  {"reasoning_score": 0.75, "instruction_score": 0.75, "stability_score": 0.75, "creativity_score": 0.70},
            "medium": {"reasoning_score": 0.50, "instruction_score": 0.60, "stability_score": 0.60, "creativity_score": 0.50},
            "small":  {"reasoning_score": 0.30, "instruction_score": 0.50, "stability_score": 0.50, "creativity_score": 0.30},
        }
        defaults = tier_defaults.get(legacy_tier, tier_defaults["medium"])

        return ModelInfo(
            name=model_name,
            endpoint=cfg.get("endpoint", "http://127.0.0.1:8080"),
            context_window=cfg.get("context_window", known["context_max"] if known else 8192),
            thinking_prefix=cfg.get("thinking_prefix", False),
            enabled=cfg.get("enabled", True),
            parallel=cfg.get("parallel", cfg.get("backend", "llama-server") != "anthropic"),
            backend=cfg.get("backend", "llama-server"),
            api_key_env=cfg.get("api_key_env"),
            claude_model=claude_model,
            tier=legacy_tier,
            # Capability vectors — explicit config > known registry > tier defaults
            reasoning_score=cfg.get("reasoning_score", known["reasoning_score"] if known else defaults["reasoning_score"]),
            instruction_score=cfg.get("instruction_score", known["instruction_score"] if known else defaults["instruction_score"]),
            stability_score=cfg.get("stability_score", known["stability_score"] if known else defaults["stability_score"]),
            creativity_score=cfg.get("creativity_score", known["creativity_score"] if known else defaults["creativity_score"]),
            tokens_per_sec=cfg.get("tokens_per_second", known["tokens_per_sec"] if known else 0.0),
            latency_ms=cfg.get("latency_ms", known["latency_ms"] if known else 2000),
            cost_per_token=cfg.get("cost_per_token", known["cost_per_token"] if known else 0.0),
        )


# ---------------------------------------------------------------------------
# Auto-detection — probe endpoints to identify backend and model capabilities
# ---------------------------------------------------------------------------

# Tier assignment from parameter count (UI label only — NOT used in routing)
TIER_THRESHOLDS = [
    (40e9, "large"),    # >40B
    (15e9, "medium"),   # 15-40B
    (0, "small"),       # <15B
]

# Thinking prefix model families
THINKING_FAMILIES = {"hunyuan", "qwen-thinking", "qwen3", "deepseek-r1"}


def _parse_param_count(text):
    """Extract parameter count from model name/description. Returns float (bytes) or None."""
    import re
    text = str(text).lower()
    # Match patterns like "80B", "7.5b", "13B", "32b"
    m = re.search(r'(\d+\.?\d*)\s*b\b', text)
    if m:
        return float(m.group(1)) * 1e9
    # Match "params": 80000000000
    m = re.search(r'"?params"?\s*[:=]\s*(\d+)', text)
    if m:
        return float(m.group(1))
    return None


def _tier_from_params(param_count):
    """Assign tier based on parameter count."""
    if param_count is None:
        return "medium"  # safe default
    for threshold, tier in TIER_THRESHOLDS:
        if param_count > threshold:
            return tier
    return "small"


def _detect_thinking_prefix(model_name):
    """Detect if model supports thinking prefix from name."""
    name_lower = (model_name or "").lower()
    return any(family in name_lower for family in THINKING_FAMILIES)


def probe_endpoint(host, port, timeout=5):
    """Probe an inference endpoint to auto-detect backend, model, and capabilities.

    Tries endpoints in order: llama-server → ollama → vllm/OpenAI → tabbyAPI.
    Returns dict with detected info, or None if unreachable.
    """
    base = f"http://{host}:{port}"
    result = {
        "endpoint": base,
        "backend": None,
        "model_name": None,
        "param_count": None,
        "tier": "medium",
        "context_window": 8192,
        "thinking_prefix": False,
        "slot_count": 1,
        "quantization": None,
        "detected": False,
        "auto_name": f"model-{host}-{port}",  # default if name can't be detected
    }

    # --- Try llama-server: /props and /slots ---
    try:
        r = requests.get(f"{base}/props", timeout=timeout)
        if r.status_code == 200:
            props = r.json()
            result["backend"] = "llama-server"
            result["model_name"] = props.get("default_generation_settings", {}).get("model", "unknown")
            result["context_window"] = props.get("default_generation_settings", {}).get("n_ctx", 8192)
            result["detected"] = True

            # Get slot info
            try:
                sr = requests.get(f"{base}/slots", timeout=3)
                if sr.status_code == 200:
                    slots = sr.json()
                    result["slot_count"] = len(slots) if isinstance(slots, list) else 1
            except Exception:
                pass

            # Infer params from model name
            param_count = _parse_param_count(result["model_name"])
            result["param_count"] = param_count
            result["tier"] = _tier_from_params(param_count)
            result["thinking_prefix"] = _detect_thinking_prefix(result["model_name"])
            # Capabilities populated from registry at model creation time

            # Auto-name: use model name if detected, else llama-host-port
            if result["model_name"] and result["model_name"] != "unknown":
                result["auto_name"] = result["model_name"]
            else:
                result["auto_name"] = f"llama-{host}-{port}"

            # Detect thinking prefix by sending a tiny prompt and checking for <think> tags
            if not result["thinking_prefix"]:
                try:
                    test_r = requests.post(
                        f"{base}/v1/chat/completions",
                        json={
                            "messages": [{"role": "user", "content": "Say yes"}],
                            "max_tokens": 20, "temperature": 0.0,
                        },
                        timeout=10,
                    )
                    if test_r.status_code == 200:
                        test_text = test_r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                        if "<think>" in test_text or "/think" in test_text or "<answer>" in test_text:
                            result["thinking_prefix"] = True
                            logger.info(f"Probe {base}: thinking prefix detected from response")
                except Exception:
                    pass

            logger.info(
                f"Probe {base}: llama-server, model={result['model_name']}, "
                f"ctx={result['context_window']}, slots={result['slot_count']}, "
                f"tier={result['tier']}, think={result['thinking_prefix']}"
            )
            return result
    except Exception:
        pass

    # --- Try ollama: /api/tags ---
    try:
        r = requests.get(f"{base}/api/tags", timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            models = data.get("models", [])
            result["backend"] = "ollama"
            result["detected"] = True

            if models:
                model = models[0]  # Use first loaded model
                result["model_name"] = model.get("name", "unknown")
                details = model.get("details", {})
                result["quantization"] = details.get("quantization_level")

                # Try /api/show for full details
                try:
                    show_r = requests.post(
                        f"{base}/api/show",
                        json={"name": result["model_name"]},
                        timeout=5,
                    )
                    if show_r.status_code == 200:
                        show_data = show_r.json()
                        model_info = show_data.get("model_info", {})
                        # Context from model info or parameters
                        for key in model_info:
                            if "context" in key.lower():
                                result["context_window"] = int(model_info[key])
                                break
                        params = show_data.get("parameters", "")
                        if "num_ctx" in params:
                            import re
                            m = re.search(r'num_ctx\s+(\d+)', params)
                            if m:
                                result["context_window"] = int(m.group(1))
                except Exception:
                    pass

                # Infer params from model name or size
                param_count = _parse_param_count(result["model_name"])
                if not param_count and model.get("size"):
                    # Rough estimate: model size / 2 bytes per param (FP16)
                    param_count = model["size"] / 2
                result["param_count"] = param_count
                result["tier"] = _tier_from_params(param_count)
                result["thinking_prefix"] = _detect_thinking_prefix(result["model_name"])
                # Capabilities populated from registry at model creation time

                # Cap ollama context to realistic values based on model size
                # Ollama reports theoretical max, not what fits in RAM
                if result["context_window"] > 32768:
                    if param_count and param_count < 10e9:
                        result["context_window"] = 4096  # small models
                    elif param_count and param_count < 35e9:
                        result["context_window"] = 8192  # medium models
                    else:
                        result["context_window"] = 16384  # large models
                    # Operator can override in the review screen

            result["auto_name"] = result["model_name"] or f"ollama-{host}-{port}"

            logger.info(
                f"Probe {base}: ollama, model={result['model_name']}, "
                f"quant={result['quantization']}, tier={result['tier']}"
            )
            return result
    except Exception:
        pass

    # --- Try vllm / OpenAI-compatible: /v1/models ---
    try:
        r = requests.get(f"{base}/v1/models", timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            models = data.get("data", [])
            result["backend"] = "vllm"
            result["detected"] = True

            if models:
                result["model_name"] = models[0].get("id", "unknown")
                param_count = _parse_param_count(result["model_name"])
                result["param_count"] = param_count
                result["tier"] = _tier_from_params(param_count)
                result["thinking_prefix"] = _detect_thinking_prefix(result["model_name"])
                # Capabilities populated from registry at model creation time

            result["auto_name"] = result["model_name"] or f"vllm-{host}-{port}"
            logger.info(f"Probe {base}: vllm/OpenAI, model={result['model_name']}, tier={result['tier']}")
            return result
    except Exception:
        pass

    # --- Try tabbyAPI: /v1/model ---
    try:
        r = requests.get(f"{base}/v1/model", timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            result["backend"] = "tabbyapi"
            result["detected"] = True
            result["model_name"] = data.get("id", data.get("model", "unknown"))
            result["quantization"] = data.get("parameters", {}).get("quant", None)

            param_count = _parse_param_count(result["model_name"])
            result["param_count"] = param_count
            result["tier"] = _tier_from_params(param_count)
            result["thinking_prefix"] = _detect_thinking_prefix(result["model_name"])
            # Capabilities populated from registry at model creation time

            logger.info(f"Probe {base}: tabbyAPI, model={result['model_name']}, tier={result['tier']}")
            return result
    except Exception:
        pass

    logger.warning(f"Probe {base}: no backend detected")
    return result


# ---------------------------------------------------------------------------
# Backend adapters — same interface as InferenceEngine for non-local backends
# ---------------------------------------------------------------------------

class AnthropicAdapter:
    """Adapter for Anthropic API. Same generate_with_messages interface."""

    def __init__(self, model_id="claude-sonnet-4-20250514", api_key_env="ANTHROPIC_API_KEY"):
        import os
        self.model_id = model_id
        self.api_key_env = api_key_env
        self.api_key = os.environ.get(api_key_env, "")
        self.endpoint = "https://api.anthropic.com/v1/messages"
        self.use_thinking = False  # set per-call by router

    def generate_with_messages(self, messages, max_tokens=None, temperature=None, timeout=180):
        _trace_msg = messages[0].get("content", "")[:60] if messages else "?"
        logger.warning(f"ANTHROPIC API CALL: {self.model_id} | {_trace_msg}")

        # Re-read key in case it was set after init
        if not self.api_key:
            import os
            self.api_key = os.environ.get(self.api_key_env, "")
        if not self.api_key:
            raise RuntimeError(f"{self.api_key_env} not set")

        # Convert messages format: our format uses role/content, Anthropic needs system separated
        system_msg = ""
        api_messages = []
        for m in messages:
            if m["role"] == "system":
                system_msg = m["content"]
            else:
                content = m["content"].replace(" /no_think", "")
                api_messages.append({"role": m["role"], "content": content})

        payload = {
            "model": self.model_id,
            "max_tokens": max_tokens or 1024,
            "messages": api_messages,
        }
        if system_msg:
            payload["system"] = system_msg
        if temperature is not None:
            payload["temperature"] = temperature

        # Extended thinking — enabled per-task by router
        if self.use_thinking:
            payload["thinking"] = {"type": "enabled", "budget_tokens": min(max_tokens or 1024, 4096)}
            # Thinking requires temperature=1 per Anthropic docs
            payload.pop("temperature", None)

        r = requests.post(
            self.endpoint,
            json=payload,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=timeout,
        )

        # Handle billing/rate errors gracefully
        if r.status_code in (401, 403):
            raise RuntimeError("Anthropic API: invalid API key")
        if r.status_code == 402:
            raise RuntimeError("Anthropic API: insufficient balance — add credits at console.anthropic.com")
        if r.status_code == 429:
            raise RuntimeError("Anthropic API: rate limited — retrying may help")
        if r.status_code == 529:
            raise RuntimeError("Anthropic API: overloaded — try again later")
        r.raise_for_status()

        data = r.json()
        content_blocks = data.get("content", [])
        text = " ".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")
        return text.strip()

    def load(self):
        return self

    def unload(self):
        pass


## ClaudeCodeAdapter REMOVED — TOS violation.
## Claude Code CLI cannot be used as a backend for automated inference.
## Use the Anthropic API adapter instead for Claude models.


class ModelRouter:
    """Routes inference tasks to appropriate LLM endpoints.

    Drop-in replacement for InferenceEngine — same interface, but selects
    the right backend per task. In solo mode, wraps a single engine with
    zero overhead.
    """

    def __init__(self, config):
        self.config = config
        self.models = {}          # name -> ModelInfo
        self._engines = {}        # name -> InferenceEngine
        self._lock = threading.Lock()
        self.mode = "solo"        # solo | multi
        self.call_log = {}  # model_name → {task: count} per cycle
        self.token_log = {}  # model_name → estimated total tokens

        # Routing preference: local_first | api_first | balanced
        routing_cfg = config.get("routing", {})
        self.preference = routing_cfg.get("preference", "local_first")

        # Scheduler reference — set by BladeRunner after init.
        # When set, generate_with_messages routes through the queue.
        self.scheduler = None

        # ── Rust scheduler — owns capacity, scoring, health, fairness ──
        import anima_core
        self._core_engine = anima_core.Engine({
            "models": config.get("models", {}),
            "tasks": config.get("tasks", {}),
            "memory": {"sqlite_path": ":memory:"},
        })
        logger.info("Router using Rust scheduler (capacity + scoring + health)")

        # Load task definitions from plugin config — engine defines NONE
        self.tasks = {}
        config_tasks = config.get("tasks", {})
        for task_name, task_cfg in config_tasks.items():
            if isinstance(task_cfg, dict):
                self.tasks[task_name] = task_cfg
        logger.info(f"Loaded {len(self.tasks)} task definitions from config")

        # Load model configs
        self._load_models_from_config(config)

    # ── Fair scheduling helpers ──────────────────────────────────────

    @staticmethod
    def set_active_plugin(name):
        """Set the current plugin for fair scheduling. Called by BladeRunner."""
        _active_plugin.name = name

    @staticmethod
    def clear_active_plugin():
        """Clear the current plugin. Called by BladeRunner after cycle."""
        _active_plugin.name = None

    @staticmethod
    def get_active_plugin():
        """Get the current plugin name, or None."""
        return getattr(_active_plugin, 'name', None)

    def _sync_model_online(self, model_key, online):
        """Sync a model's online status to the Rust scoring engine."""
        if self._core_engine:
            try:
                self._core_engine.update_model_online(model_key, online)
                logger.info(f"Rust sync: {model_key} → {'online' if online else 'offline'}")
            except Exception as e:
                logger.warning(f"Rust sync FAILED for {model_key}: {e}")

    def _system_pressure(self):
        """System-wide load ratio (0.0 = idle, 1.0 = saturated)."""
        # Capacity info from model config (Rust owns actual in-flight counts)
        total_capacity = sum(
            (2 if m.parallel else 1)
            for m in self.models.values() if m.online and m.enabled
        )
        return 0.0 if total_capacity == 0 else 0.5  # approximate — real load is in Rust

    def get_capacity(self, task_desc=None):
        """Query available capacity."""
        available = 0
        total_capacity = 0
        for name, info in self.models.items():
            if not info.online or not info.enabled:
                continue
            available += 1
            total_capacity += (2 if info.parallel else 1)

        return {
            "available_models": available,
            "total_capacity": total_capacity,
            "estimated_wait": "blocked" if available == 0 else "none",
        }

    def _load_models_from_config(self, config):
        """Load model definitions from config."""
        from core.inference import InferenceEngine

        models_cfg = config.get("models", {})
        routing_cfg = config.get("routing", {})
        self.mode = routing_cfg.get("mode", "solo")

        if models_cfg:
            for key, model_cfg in models_cfg.items():
                if not isinstance(model_cfg, dict):
                    continue
                info = ModelInfo.from_config(key, model_cfg)
                self.models[key] = info

                # Create the right engine per backend type
                if info.backend == "anthropic":
                    self._engines[key] = AnthropicAdapter(
                        model_id=info.claude_model or "claude-sonnet-4-20250514",
                        api_key_env=info.api_key_env or "ANTHROPIC_API_KEY",
                    )
                else:
                    # llama-server, ollama, vllm, tabbyapi — all use OpenAI-compatible API
                    engine_config = dict(config)
                    engine_config["hardware"] = dict(config.get("hardware", {}))
                    engine_config["hardware"]["inference_mode"] = "server"
                    engine_config["hardware"]["inference_server"] = info.endpoint
                    engine_config["hardware"]["thinking_prefix"] = info.thinking_prefix
                    # Ollama requires model name in payload
                    if info.backend == "ollama":
                        engine_config["hardware"]["ollama_model"] = info.name
                    self._engines[key] = InferenceEngine(engine_config)

        # No models configured — that's fine, operator adds via dashboard
        if not self.models:
            self.mode = "solo"
            logger.info("No models configured. Add models via Settings.")

    def load(self):
        """Initialize all model connections. Run health checks."""
        for name, engine in self._engines.items():
            info = self.models[name]
            try:
                if info.backend in ("ollama", "anthropic"):
                    # Non-standard backends — use health check instead of engine.load()
                    result = self.health_check(name)
                    if result.get(name, {}).get("online"):
                        info.online = True
                        self._sync_model_online(name, True)
                        logger.info(f"Model '{name}' connected ({info.backend})")
                    else:
                        info.online = False
                        self._sync_model_online(name, False)
                        logger.warning(f"Model '{name}' offline ({info.backend})")
                else:
                    engine.load()
                    info.online = True
                    self._sync_model_online(name, True)
                    logger.info(f"Model '{name}' connected at {info.endpoint}")
            except Exception as e:
                info.online = False
                self._sync_model_online(name, False)
                logger.warning(f"Model '{name}' offline: {e}")

        # Check online models — log, don't block startup.
        # Operators must be able to boot the platform before models are reachable
        # (enable a disabled model from the dashboard, wait for a local server to
        # finish loading, swap endpoints). Router wakes up empty and picks up
        # models as they come online via health-check polling or dashboard actions.
        online = [n for n, m in self.models.items() if m.online]
        if not online and self.models:
            logger.warning(
                f"Router starting with 0/{len(self.models)} models online. "
                f"Dispatches will fail until a model recovers or is re-enabled."
            )
        elif not self.models:
            logger.warning("No models configured — add via dashboard Settings")

        logger.info(
            f"ModelRouter ready: {len(online)}/{len(self.models)} models online, "
            f"mode={self.mode}"
        )

        # Load cached calibration or run fresh
        if self.mode == "multi" and len(online) > 1:
            self._load_cached_calibration()
            uncalibrated = [n for n in online if self.models[n].tokens_per_second == 0.0]
            if uncalibrated:
                logger.info(f"Calibrating {len(uncalibrated)} uncached models")
                for n in uncalibrated:
                    self.calibrate(n)
                self._save_calibration_cache()

        # Start slot monitor — polls real server state to detect idle models
        self._start_slot_monitor()

        return self

    def _start_slot_monitor(self):
        """Background thread that polls server slots every 3 seconds.

        When a model's server is idle (is_processing=false) but Rust thinks
        it's at capacity, sweep expired reservations to free the slot.
        This handles external usage (Stephen) and leaked reservations.
        """
        import requests as _req

        def _monitor():
            while True:
                try:
                    time.sleep(3)
                    for name, info in self.models.items():
                        if not info.online or not info.enabled:
                            continue
                        if info.backend in ("anthropic", "ollama"):
                            continue  # can't check slots on these

                        try:
                            r = _req.get(f"{info.endpoint}/slots", timeout=2)
                            if r.status_code == 200:
                                slots = r.json()
                                if slots and not slots[0].get("is_processing", True):
                                    # Server idle — force-clear if Rust thinks busy
                                    self._core_engine.force_clear_model(name)
                        except Exception:
                            pass
                except Exception:
                    pass

        t = threading.Thread(target=_monitor, name="slot-monitor", daemon=True)
        t.start()
        logger.info("Slot monitor started (3s poll)")

    def _load_cached_calibration(self):
        """Load cached tok/s from config. Skip LLM calls for known models."""
        try:
            import toml, os
            config_path = os.environ.get("ANIMA_CONFIG", "config/settings.toml")
            cfg = toml.load(config_path)
            for key, model_cfg in cfg.get("models", {}).items():
                cached_tps = model_cfg.get("tokens_per_second", 0)
                if cached_tps and key in self.models:
                    self.models[key].tokens_per_second = cached_tps
                    logger.info(f"Calibration cached: '{key}' = {cached_tps} tok/s")
        except Exception as e:
            logger.debug(f"Calibration cache load failed: {e}")

    def _save_calibration_cache(self):
        """Save tok/s to config so next startup skips calibration."""
        try:
            import toml, os
            config_path = os.environ.get("ANIMA_CONFIG", "config/settings.toml")
            cfg = toml.load(config_path)
            for key, info in self.models.items():
                if key in cfg.get("models", {}) and info.tokens_per_second > 0:
                    cfg["models"][key]["tokens_per_second"] = info.tokens_per_second
            with open(config_path, "w") as f:
                toml.dump(cfg, f)
        except Exception as e:
            logger.debug(f"Calibration cache save failed: {e}")

    def calibrate(self, name=None):
        """Run speed calibration on one or all models.

        Sends a short generation request, measures tokens/second.
        Runs once at startup or when a new model is added.
        """
        targets = [name] if name else [
            n for n, info in self.models.items() if info.online
        ]

        for n in targets:
            info = self.models.get(n)
            engine = self._engines.get(n)
            if not info or not engine or not info.online:
                continue

            try:
                # Anthropic/Claude Code: use published specs, no API calls
                if info.backend == "anthropic":
                    # Published approximate speeds from Anthropic docs
                    model_id = (info.claude_model or "").lower()
                    if "haiku" in model_id:
                        info.tokens_per_second = 100.0
                    elif "sonnet" in model_id:
                        info.tokens_per_second = 60.0
                    elif "opus" in model_id:
                        info.tokens_per_second = 30.0
                    else:
                        info.tokens_per_second = 50.0
                    logger.info(f"Calibration: '{n}' ({info.backend}) = {info.tokens_per_second} tok/s (published spec)")
                    continue
                # Local models: send test prompt, measure actual tok/s from API response
                prompt_messages = [{"role": "user", "content": "Count from 1 to 10: 1, 2, 3, /no_think"}]

                if info.backend == "llama-server":
                    # Direct API call to get usage stats
                    start = time.time()
                    r = requests.post(
                        f"{info.endpoint}/v1/chat/completions",
                        json={"messages": prompt_messages, "max_tokens": 50, "temperature": 0.0},
                        timeout=30,
                    )
                    elapsed = time.time() - start
                    if r.status_code == 200:
                        data = r.json()
                        tokens = data.get("usage", {}).get("completion_tokens", 0)
                        if tokens > 0:
                            tps = tokens / max(elapsed, 0.01)
                            info.tokens_per_second = round(tps, 1)
                        else:
                            info.tokens_per_second = 0.0

                elif info.backend == "ollama":
                    # Ollama /api/generate returns token counts
                    start = time.time()
                    r = requests.post(
                        f"{info.endpoint}/api/generate",
                        json={"model": info.name, "prompt": "Count from 1 to 10: 1, 2, 3,", "stream": False},
                        timeout=30,
                    )
                    elapsed = time.time() - start
                    if r.status_code == 200:
                        data = r.json()
                        tokens = data.get("eval_count", 0)
                        if tokens > 0:
                            tps = tokens / max(elapsed, 0.01)
                            info.tokens_per_second = round(tps, 1)
                        else:
                            info.tokens_per_second = 0.0
                else:
                    # Other backends: use generate_with_messages fallback
                    start = time.time()
                    response = engine.generate_with_messages(
                        prompt_messages, max_tokens=50, temperature=0.0, timeout=30,
                    )
                    elapsed = time.time() - start
                    import re
                    response = re.sub(r"<think>.*?</think>", "", response or "", flags=re.DOTALL).strip()
                    token_estimate = max(len(response.split()), 1)
                    tps = token_estimate / max(elapsed, 0.01)
                    info.tokens_per_second = round(tps, 1)

                logger.info(
                    f"Calibration: '{n}' ({info.backend}) = {info.tokens_per_second} tok/s "
                    f"(measured in {elapsed:.1f}s)"
                )
            except Exception as e:
                logger.warning(f"Calibration failed for '{n}' ({info.backend}): {e}")
                info.tokens_per_second = 0.0

    def validate_all(self):
        """Pre-run validation — check all models for health, identity, and consistency.

        Call before autonomous runs. Catches:
        - Offline models (network issues)
        - Substrate swaps (model changed since last check)
        - Billing failures (API key expired, no credits)
        - Role gaps (no model available for critical tasks)

        Returns dict of issues found.
        """
        issues = []

        # Health check all models
        self.health_check()

        # Identity check — detect substrate swaps
        for name, info in self.models.items():
            if not info.online or not info.enabled:
                continue
            if info.backend in ("anthropic"):
                continue  # Can't probe these without spending credits

            probe = self.probe_model(name)
            if probe and probe.get("model_changed"):
                old = probe.get("previous_model", "?")
                new = probe.get("model_name", "?")
                issues.append({
                    "type": "substrate_swap",
                    "model": name,
                    "message": f"Model changed: {old} → {new}",
                })
                # Auto-correct capabilities from registry
                info.name = new
                info.context_window = probe.get("context_window", info.context_window)
                info.thinking_prefix = probe.get("thinking_prefix", info.thinking_prefix)
                known = _lookup_known_model(new)
                if known:
                    info.reasoning_score = known["reasoning_score"]
                    info.instruction_score = known["instruction_score"]
                    info.stability_score = known["stability_score"]
                    info.creativity_score = known["creativity_score"]
                info.tier = probe.get("tier", _tier_label(info.reasoning_score))
                logger.warning(f"Auto-corrected '{name}': {old} → {new}, tier={info.tier}")

        # Check for capability gaps — pause_only tasks must have at least one scorer
        for task_name, task_cfg in self.tasks.items():
            if not task_cfg.get("pause_only"):
                continue
            task_desc = self._parse_task_def(task_name)
            if self._core_engine:
                desc_dict = self._build_task_desc_dict(task_desc)
                result = self._core_engine.select_model_v2(desc_dict)
                handlers = [result] if result else []
                if result:
                    self._core_engine.release_model(result)
            else:
                handlers = [n for n, i in self.models.items() if i.online and i.enabled]
            if not handlers:
                issues.append({
                    "type": "capability_gap",
                    "task": task_name,
                    "message": f"No model meets capability requirements for {task_name} (pause_only)",
                })

        # Check error-disabled models that might have recovered
        for name, info in self.models.items():
            if not info.online and info.error_count >= 3 and info.enabled:
                # Try re-enabling
                check = self.health_check(name)
                if check.get(name, {}).get("online"):
                    info.error_count = 0
                    issues.append({
                        "type": "recovered",
                        "model": name,
                        "message": f"Model '{name}' recovered from error state",
                    })

        if issues:
            logger.info(f"Pre-run validation: {len(issues)} issues — {[i['type'] for i in issues]}")
        else:
            logger.info("Pre-run validation: all models healthy")

        return {"issues": issues, "models_online": sum(1 for i in self.models.values() if i.online)}

    def health_check(self, name=None):
        """Run health check on one or all models."""
        targets = [name] if name else list(self.models.keys())
        results = {}
        for n in targets:
            info = self.models.get(n)
            engine = self._engines.get(n)
            if not info or not engine:
                continue

            start = time.time()
            try:
                # Health check — different endpoints per backend
                if info.backend == "ollama":
                    r = requests.get(f"{info.endpoint}/api/tags", timeout=5)
                    online = r.status_code == 200
                elif info.backend in ("anthropic", "openai"):
                    # API models: assumed always online if enabled.
                    # Only goes offline on billing/auth failure (handled in _handle_engine_error).
                    # NO probing — trust until it fails.
                    online = info.enabled
                else:
                    # llama-server, vllm, tabbyapi (default)
                    r = requests.get(f"{info.endpoint}/health", timeout=5)
                    online = r.status_code == 200

                latency = int((time.time() - start) * 1000)
                info.online = online
                info.last_latency_ms = latency
                info.last_check = datetime.now(timezone.utc).isoformat()
                self._sync_model_online(n, online)

                # Check slots for busy status (llama-server only)
                if info.online and info.backend not in ("ollama", "anthropic"):
                    try:
                        slots_r = requests.get(f"{info.endpoint}/slots", timeout=3)
                        if slots_r.status_code == 200:
                            slots = slots_r.json()
                            info.busy = any(
                                s.get("is_processing", False)
                                for s in slots
                                if isinstance(s, dict)
                            )
                    except Exception:
                        info.busy = False
                elif info.online:
                    info.busy = False  # Ollama manages its own queue

                info.error_count = 0

                # Periodic model identity check — detect substrate swaps
                # Run every 10th health check to avoid overhead
                if not hasattr(info, '_hc_count'):
                    info._hc_count = 0
                info._hc_count += 1
                if info._hc_count % 10 == 0 and info.online:
                    try:
                        probe = self.probe_model(n)
                        if probe and probe.get("model_changed"):
                            info._model_changed = True
                            info._new_model = probe.get("model_name")
                            # Update tier/roles from new model
                            new_name = probe.get("model_name", info.name)
                            info.name = new_name
                            info.context_window = probe.get("context_window", info.context_window)
                            info.thinking_prefix = probe.get("thinking_prefix", info.thinking_prefix)
                            known = _lookup_known_model(new_name)
                            if known:
                                info.reasoning_score = known["reasoning_score"]
                                info.instruction_score = known["instruction_score"]
                                info.stability_score = known["stability_score"]
                                info.creativity_score = known["creativity_score"]
                            info.tier = _tier_label(info.reasoning_score)
                            logger.info(f"Model '{n}' updated: {new_name}, tier={info.tier}")
                    except Exception:
                        pass

            except Exception:
                info.online = False
                info.busy = False
                info.error_count += 1
                info.last_check = datetime.now(timezone.utc).isoformat()
                self._sync_model_online(n, False)

            results[n] = info.to_dict()
        return results

    def _parse_task_def(self, task):
        """Parse task definition — returns a TaskDescriptor.

        Checks config [tasks.*] first, then falls back to resolve_task()
        which handles legacy names and presets.
        """
        from core.task_presets import TaskDescriptor, resolve_task

        # If it's already a descriptor, pass through
        if isinstance(task, TaskDescriptor):
            return task

        # Check config-defined tasks first
        task_def = self.tasks.get(task, {}) if isinstance(task, str) else {}
        if task_def and isinstance(task_def, dict):
            # Config has explicit definition — build descriptor from it
            task_class = task_def.get("task_class")
            min_reasoning = task_def.get("min_reasoning")

            # Legacy format: reasoning=high/medium/low → convert
            if task_class is None:
                task_class = LEGACY_TASK_CLASS.get(task, "extraction")
            if min_reasoning is None:
                legacy_reasoning = task_def.get("reasoning", "low")
                min_reasoning = LEGACY_REASONING_MAP.get(legacy_reasoning, 0.1)

            return TaskDescriptor(
                task_class=task_class,
                min_reasoning=min_reasoning,
                min_context=task_def.get("min_context", task_def.get("context_min", 0)),
                prefer=task_def.get("prefer", "speed"),
                pause_only=task_def.get("pause_only", False),
                safe_fallback=task_def.get("safe_fallback", False),
                thinking=task_def.get("thinking", False),
                label=task if isinstance(task, str) else None,
            )

        # No config definition — use presets
        return resolve_task(task)

    def _build_task_desc_dict(self, descriptor):
        """Convert a TaskDescriptor to a dict for Rust select_model_v2."""
        return {
            "task_class": getattr(descriptor, 'task_class', 'extraction'),
            "min_tier": getattr(descriptor, 'min_tier', 'small'),
            "min_context": getattr(descriptor, 'min_context', 0),
            "thinking": getattr(descriptor, 'thinking', False),
            "execution": getattr(descriptor, 'execution', 'stateless'),
            "latency": getattr(descriptor, 'latency', 'medium'),
            "determinism": getattr(descriptor, 'determinism', False),
            "parallel_ok": getattr(descriptor, 'parallel_ok', True),
            "typical_tokens": getattr(descriptor, 'typical_tokens', 1024),
            "priority": getattr(descriptor, 'priority', 2),
            "prefer": getattr(descriptor, 'prefer', 'speed'),
            "plugin_id": getattr(descriptor, 'plugin_id', '') or self.get_active_plugin() or '',
        }

    def get_engine(self, task=None, estimated_tokens=None, task_desc=None):
        """Get an engine for a task (informational — no reservation).

        For actual dispatch, use _acquire_engine which reserves via Rust.
        This method is for backward-compat queries like get_model_name().
        """
        if self.mode == "solo" or (task is None and task_desc is None):
            if not self._engines:
                return None
            return self._engines[next(iter(self._engines))]

        # Try Rust scoring (non-reserving — use select_model_v2 + release)
        if self._core_engine:
            from core.task_presets import resolve_task
            descriptor = resolve_task(task_desc or task)
            desc_dict = self._build_task_desc_dict(descriptor)
            selected = self._core_engine.select_model_v2(desc_dict)
            if selected:
                self._core_engine.release_model(selected)
                if selected in self._engines:
                    return self._engines[selected]

        # Fallback — any online model
        for name, info in self.models.items():
            if info.online and info.enabled:
                return self._engines.get(name)
        return None

    def get_model_name(self, task=None):
        """Get the name of the model that would handle this task."""
        if self.mode == "solo" or task is None:
            if not self.models:
                return None
            return self.models[next(iter(self.models))].name

        engine = self.get_engine(task=task)
        if engine:
            name = self._engine_name(engine)
            if name and name in self.models:
                return self.models[name].name
        return None

    # -----------------------------------------------------------------------
    # Drop-in InferenceEngine interface — delegates to the appropriate engine
    # -----------------------------------------------------------------------

    def _estimate_tokens(self, messages, max_tokens=None):
        """Estimate total tokens (prompt + response) for context check."""
        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        prompt_tokens = prompt_chars // 3
        response_tokens = max_tokens or 1024
        return prompt_tokens + response_tokens

    def _release(self, reservation_id):
        """Release a reservation via Rust. Idempotent."""
        if reservation_id:
            try:
                self._core_engine.release_reservation(reservation_id)
            except Exception as e:
                logger.warning(f"Reservation release failed (rid={reservation_id}): {e}")

    def _record_result(self, reservation_id, success, latency_ms):
        """Record result + release via Rust. Preferred completion path."""
        if reservation_id:
            try:
                self._core_engine.record_result_v2(reservation_id, success, latency_ms)
            except Exception as e:
                logger.warning(f"Record result failed (rid={reservation_id}): {e}")
                # Fallback — try raw release
                try:
                    self._core_engine.release_reservation(reservation_id)
                except Exception:
                    pass

    def _log_dispatch(self, model_name, task_label, messages, max_tokens):
        """Thread-safe dispatch logging."""
        if not model_name:
            return
        with self._lock:
            self.call_log.setdefault(model_name, {})
            self.call_log[model_name][task_label] = self.call_log[model_name].get(task_label, 0) + 1
            est = self._estimate_tokens(messages, max_tokens)
            self.token_log[model_name] = self.token_log.get(model_name, 0) + est

    def _configure_thinking(self, engine, descriptor):
        """Enable thinking on Anthropic adapters if descriptor requires it."""
        if isinstance(engine, AnthropicAdapter):
            name = self._engine_name(engine)
            info = self.models.get(name)
            if getattr(descriptor, 'thinking', False):
                model_id = (info.claude_model or "").lower() if info else ""
                engine.use_thinking = "sonnet" in model_id or "opus" in model_id
            else:
                engine.use_thinking = False

    def _acquire_engine(self, descriptor, estimated_tokens, timeout):
        """Block until a model with capacity is available, then reserve it.

        Uses Rust wait_and_reserve — blocks on a condvar signaled by every
        reservation release. Plugin-fair: starvation boost ensures no plugin
        is permanently starved. No Python polling loop needed.

        Returns (engine, model_name, reservation_id).
        Caller MUST call _release(reservation_id) or
        _record_result(reservation_id, ...) when done.
        """
        label = getattr(descriptor, 'label', None) or descriptor.task_class

        if self._core_engine:
            desc_dict = self._build_task_desc_dict(descriptor)

            # Rust blocks with condvar — releases GIL so other threads proceed
            # timeout_ms=0 would wait forever, use 10 minutes as safety
            timeout_ms = int(timeout * 1000) if timeout else 600_000
            sel = self._core_engine.wait_and_reserve(desc_dict, timeout_ms)

            if sel is not None:
                model_key = sel["model"]
                rid = sel["reservation_id"]
                if model_key in self._engines:
                    return self._engines[model_key], model_key, rid
                else:
                    self._core_engine.release_reservation(rid)
                    # Shouldn't happen — model in Rust but not Python
                    raise RuntimeError(f"Model '{model_key}' has no Python engine")

            # Timed out — check if models are online
            any_online = any(m.online and m.enabled for m in self.models.values())
            if not any_online:
                raise RuntimeError(f"No models online for '{label}'")
            raise RuntimeError(f"Timeout acquiring model for '{label}' after {timeout}s")

    def generate(self, prompt, system_context="", max_tokens=None, task=None):
        from core.task_presets import resolve_task
        descriptor = resolve_task(task)
        # Preset default; explicit override is a code-smell and logged below
        if max_tokens is None:
            max_tokens = descriptor.max_output_tokens
        else:
            logger.debug(
                f"max_tokens literal override: {max_tokens} (task={task}, "
                f"preset default={descriptor.max_output_tokens}). "
                f"Call site should omit max_tokens and use task= only."
            )
        estimated = (len(prompt) + len(system_context)) // 3 + max_tokens
        engine, model_name, rid = self._acquire_engine(descriptor, estimated, 180)
        _start = time.time()
        try:
            result = engine.generate(prompt, system_context, max_tokens)
            self._record_result(rid, True, (time.time() - _start) * 1000)
            return result
        except Exception as e:
            self._handle_engine_error(model_name, e)
            raise
        finally:
            self._release(rid)  # idempotent — safe even after _record_result

    def generate_streaming(self, prompt, system_context="", max_tokens=None, task=None):
        from core.task_presets import resolve_task
        descriptor = resolve_task(task)
        if max_tokens is None:
            max_tokens = descriptor.max_output_tokens
        engine, model_name, rid = self._acquire_engine(descriptor, 0, 180)
        try:
            yield from engine.generate_streaming(prompt, system_context, max_tokens)
        finally:
            self._release(rid)

    def generate_with_messages(self, messages, max_tokens=None, temperature=None,
                               timeout=180, task=None, task_desc=None):
        """Dispatch messages to the best available model.

        Rust scheduler owns capacity. Python acquires, dispatches, records result.
        Reservation ID guarantees release even on crash (Rust sweep recovers).

        max_tokens: if None, uses descriptor.max_output_tokens (the preset default).
        Explicit literals are logged at DEBUG — call sites should pass task= only.
        """
        from core.task_presets import resolve_task
        descriptor = resolve_task(task_desc or task)

        # Preset default; explicit override is a code-smell and logged
        if max_tokens is None:
            max_tokens = descriptor.max_output_tokens
        else:
            logger.debug(
                f"max_tokens literal override: {max_tokens} (task={task}, "
                f"preset default={descriptor.max_output_tokens}). "
                f"Call site should omit max_tokens and use task= only."
            )

        # Auto-scale timeout based on max_tokens. Caller-passed timeout is a
        # floor — router enforces minimum based on realistic throughput.
        # MIN_TOKENS_PER_SEC=5 is conservative: covers slow reasoning models
        # (MiniMax ~10 tok/s) with 2x safety margin. +60s buffer for prompt
        # processing, network latency, model acquire wait.
        MIN_TOKENS_PER_SEC = 5
        min_required_timeout = (max_tokens // MIN_TOKENS_PER_SEC) + 60
        if timeout < min_required_timeout:
            logger.debug(
                f"Timeout auto-scaled: {timeout}s → {min_required_timeout}s "
                f"(max_tokens={max_tokens}, floor={MIN_TOKENS_PER_SEC} tok/s). "
                f"Task={task}. Caller-passed timeout was too short for budget."
            )
            timeout = min_required_timeout

        estimated_input = self._estimate_tokens(messages, 0)
        estimated = estimated_input + max_tokens
        task_label = getattr(descriptor, 'label', None) or task or descriptor.task_class

        # Acquire — blocks until Rust grants a slot
        engine, model_name, rid = self._acquire_engine(descriptor, estimated, timeout)
        _dispatch_start = time.time()

        # Context budget check — input + output must fit in 90% of the
        # SELECTED MODEL's actual context window. Previous implementation
        # checked against descriptor.min_context (a routing floor), which
        # rejected valid requests that fit in the model but exceeded the
        # floor. Now checked post-selection against real capacity.
        model_info = self.models.get(model_name)
        ctx_budget = model_info.context_window if model_info else 32768
        if estimated_input + max_tokens > ctx_budget * 0.9:
            self._record_result(rid, False, 0)
            rid = 0
            raise RuntimeError(
                f"Context budget exceeded: input={estimated_input} + "
                f"max_output={max_tokens} > 90% of context={ctx_budget} "
                f"(model={model_name}, task={task}). Reduce input or use "
                f"a model with larger context window."
            )

        # Configure thinking if needed
        self._configure_thinking(engine, descriptor)

        # Log dispatch
        self._log_dispatch(model_name, task_label, messages, max_tokens)

        # Dispatch
        try:
            result = engine.generate_with_messages(messages, max_tokens, temperature, timeout)
            # Success — record result (releases reservation + feeds health metrics)
            latency_ms = (time.time() - _dispatch_start) * 1000
            self._record_result(rid, True, latency_ms)
            rid = 0  # prevent double-release in finally
            if model_name and self.models.get(model_name):
                m = self.models[model_name]
                m.error_count = 0
                m.consecutive_timeouts = 0
                m.recent_success_rate = m.recent_success_rate * 0.95 + 0.05
            return result

        except Exception as primary_error:
            self._handle_engine_error(model_name, primary_error)

            # Release primary BEFORE trying fallback
            self._release(rid)
            rid = 0

            # Fallback — acquire a different model via Rust
            fb_rid = 0
            try:
                fb_engine, fb_name, fb_rid = self._acquire_engine(descriptor, estimated, timeout)
                if self._engine_name(fb_engine) == self._engine_name(engine):
                    # Got the same model — skip fallback
                    self._release(fb_rid)
                    raise primary_error

                logger.info(f"Failover: {self._engine_name(engine)} → {fb_name} for '{task_label}'")
                self._configure_thinking(fb_engine, descriptor)
                fb_start = time.time()
                result = fb_engine.generate_with_messages(messages, max_tokens, temperature, timeout)
                self._record_result(fb_rid, True, (time.time() - fb_start) * 1000)
                if fb_name and self.models.get(fb_name):
                    fb_m = self.models[fb_name]
                    fb_m.error_count = 0
                    fb_m.consecutive_timeouts = 0
                fb_rid = 0
                return result
            except RuntimeError:
                # No models online for fallback
                raise primary_error
            except Exception:
                self._release(fb_rid)
                raise primary_error

        finally:
            # Defensive — idempotent release ensures no leak
            self._release(rid)

    def dispatch_batch(self, items, max_workers=None):
        """Work-stealing batch dispatch. Models pull tasks as they free up.

        Args:
            items: list of dicts, each with:
                messages: list of message dicts
                task: task type string
                max_tokens: optional
                timeout: optional
                _id: optional identifier passed through to result
            max_workers: thread pool size (default: number of online models)

        Returns list of dicts in same order:
            {_id, result, error, model_name, elapsed_s}

        Fast models naturally process more items. No pre-assignment.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import queue

        if not items:
            return []

        # Build work queue
        work = queue.Queue()
        for idx, item in enumerate(items):
            work.put((idx, item))

        online_count = sum(1 for m in self.models.values() if m.online and m.enabled and m.parallel)
        workers = max_workers or max(online_count, 1)
        results = [None] * len(items)

        def _worker():
            while True:
                try:
                    idx, item = work.get_nowait()
                except queue.Empty:
                    return

                task = item.get("task")
                messages = item.get("messages", [])
                max_tokens = item.get("max_tokens")
                timeout = item.get("timeout", 180)
                item_id = item.get("_id", idx)

                start = time.time()
                try:
                    result = self.generate_with_messages(
                        messages, max_tokens=max_tokens,
                        timeout=timeout, task=task,
                    )
                    elapsed = time.time() - start
                    results[idx] = {
                        "_id": item_id,
                        "result": result,
                        "error": None,
                        "elapsed_s": round(elapsed, 2),
                    }
                except Exception as e:
                    elapsed = time.time() - start
                    results[idx] = {
                        "_id": item_id,
                        "result": None,
                        "error": str(e),
                        "elapsed_s": round(elapsed, 2),
                    }

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_worker) for _ in range(workers)]
            for f in futures:
                f.result()  # wait for all workers

        return results

    def reset_call_log(self):
        """Reset per-cycle call log."""
        self.call_log = {}

    def _engine_name(self, engine):
        """Find the name of an engine instance."""
        for name, eng in self._engines.items():
            if eng is engine:
                return name
        return None

    def _handle_engine_error(self, name, error):
        """Track errors. Only disable on auth/billing — never on timeouts."""
        if not name:
            return
        info = self.models.get(name)
        if not info:
            return
        error_str = str(error).lower()
        # Rate limit — backoff, don't count
        if "rate limit" in error_str:
            logger.info(f"Model '{name}' rate limited — will retry")
            return
        # Timeout — single timeouts are fine (slow models on big tasks).
        # But N-in-a-row means the box is frozen, not slow. Reset on any success.
        if "timeout" in error_str or "timed out" in error_str:
            info.consecutive_timeouts += 1
            if info.consecutive_timeouts >= 3:
                info.online = False
                self._sync_model_online(name, False)
                logger.warning(
                    f"Model '{name}' offline after {info.consecutive_timeouts} consecutive timeouts — likely frozen"
                )
            else:
                logger.info(
                    f"Model '{name}' timed out ({info.consecutive_timeouts}/3 consecutive)"
                )
            return
        # Immediate disable ONLY on auth/billing errors
        if any(s in error_str for s in ("insufficient balance", "invalid api key", "not set", "unauthorized", "authentication")):
            info.online = False
            self._sync_model_online(name, False)
            logger.warning(f"Model '{name}' disabled (billing/auth): {error}")
            return
        # Connection refused / unreachable / reset — model actually down
        if any(s in error_str for s in ("connection", "unreachable", "refused", "reset", "broken pipe")):
            info.error_count += 1
            if info.error_count >= 2:
                info.online = False
                self._sync_model_online(name, False)
                logger.warning(f"Model '{name}' offline after {info.error_count} connection failures")
            return
        # Other errors — log but don't disable
        info.error_count += 1
        logger.warning(f"Model '{name}' error ({info.error_count}): {error}")

    # _get_fallback removed — fallback logic is now inline in generate_with_messages
    # using V3 scoring with the resolved descriptor. No separate method needed.

    def generate_with_messages_streaming(self, messages, max_tokens=None, task=None):
        from core.task_presets import resolve_task
        descriptor = resolve_task(task)
        engine, model_name, rid = self._acquire_engine(descriptor, 0, 180)
        try:
            yield from engine.generate_with_messages_streaming(messages, max_tokens)
        finally:
            self._release(rid)

    def unload(self):
        for engine in self._engines.values():
            engine.unload()

    # -----------------------------------------------------------------------
    # Model management (called from dashboard endpoints)
    # -----------------------------------------------------------------------

    def add_model(self, key, model_cfg):
        """Add a new model at runtime. Auto-detects if only endpoint provided."""
        from core.inference import InferenceEngine

        # Auto-detect if minimal config (just endpoint, no name/tier)
        endpoint = model_cfg.get("endpoint", "")
        if endpoint and not model_cfg.get("name"):
            try:
                # Parse host:port from endpoint URL
                from urllib.parse import urlparse
                parsed = urlparse(endpoint)
                host = parsed.hostname or "localhost"
                port = parsed.port or 8080
                probe = probe_endpoint(host, port)
                if probe and probe.get("detected"):
                    # Merge probe results — operator overrides take precedence
                    for field in ["name", "tier", "context_window", "thinking_prefix", "backend"]:
                        probe_key = "model_name" if field == "name" else field
                        if field not in model_cfg and probe.get(probe_key) is not None:
                            model_cfg[field] = probe[probe_key]
                    logger.info(
                        f"Auto-detected: {probe.get('backend')} / "
                        f"{probe.get('model_name')} / tier={probe.get('tier')}"
                    )
            except Exception as e:
                logger.debug(f"Auto-detection failed for {endpoint}: {e}")

        info = ModelInfo.from_config(key, model_cfg)
        self.models[key] = info

        engine_config = dict(self.config)
        engine_config["hardware"] = dict(self.config.get("hardware", {}))
        engine_config["hardware"]["inference_mode"] = "server"
        engine_config["hardware"]["inference_server"] = info.endpoint
        engine_config["hardware"]["thinking_prefix"] = info.thinking_prefix
        self._engines[key] = InferenceEngine(engine_config)

        # Attempt connection
        try:
            if info.backend == "ollama":
                result = self.health_check(key)
                info.online = result.get(key, {}).get("online", False)
            else:
                self._engines[key].load()
                info.online = True
        except Exception:
            info.online = False

        if len(self.models) > 1:
            self.mode = "multi"

        # Calibrate new model speed and cache
        if info.online:
            self.calibrate(key)
            self._save_calibration_cache()

        logger.info(f"Model '{key}' added: {info.endpoint} (online={info.online})")
        return info.to_dict()

    def probe_model(self, key):
        """Re-probe an existing model to detect changes (model swap, restart)."""
        info = self.models.get(key)
        if not info:
            return None
        try:
            from urllib.parse import urlparse
            parsed = urlparse(info.endpoint)
            host = parsed.hostname or "localhost"
            port = parsed.port or 8080
            probe = probe_endpoint(host, port)
            if probe and probe.get("detected"):
                old_name = info.name
                new_name = probe.get("model_name", "unknown")
                if old_name != new_name and new_name != "unknown":
                    logger.warning(
                        f"MODEL CHANGED: '{key}' was '{old_name}', "
                        f"now '{new_name}' at {info.endpoint}"
                    )
                    probe["model_changed"] = True
                    probe["previous_model"] = old_name
                else:
                    probe["model_changed"] = False
            return probe
        except Exception as e:
            logger.debug(f"Re-probe failed for {key}: {e}")
            return None

    def remove_model(self, key):
        """Remove a model."""
        if key not in self.models:
            return False

        self.models.pop(key, None)
        engine = self._engines.pop(key, None)
        if engine:
            engine.unload()

        if len(self.models) <= 1:
            self.mode = "solo"

        logger.info(f"Model '{key}' removed")
        return True

    def set_enabled(self, key, enabled):
        """Enable/disable a model without removing its config."""
        if key in self.models:
            info = self.models[key]
            info.enabled = enabled
            # Rust scheduler only tracks online — hide disabled models by
            # flipping their Rust-side online state. Actual online is preserved
            # in Python and restored when the model is re-enabled.
            self._sync_model_online(key, enabled and info.online)
            logger.info(f"Model '{key}' {'enabled' if enabled else 'disabled'}")
            return True
        return False

    def update_model(self, key, updates):
        """Update model config fields."""
        info = self.models.get(key)
        if not info:
            return False

        if "endpoint" in updates:
            info.endpoint = updates["endpoint"]
            if info.backend == "anthropic":
                # API backend — endpoint is informational, adapter hardcodes the URL.
                # Never probe API models; assumed online until a billing/auth call fails.
                if not isinstance(self._engines.get(key), AnthropicAdapter):
                    self._engines[key] = AnthropicAdapter(
                        model_id=info.claude_model or "claude-sonnet-4-20250514",
                        api_key_env=info.api_key_env or "ANTHROPIC_API_KEY",
                    )
                info.online = True
                self._sync_model_online(key, True)
                logger.info(f"Model '{key}' endpoint updated to {info.endpoint} (anthropic, no probe)")
            else:
                # Recreate engine with new endpoint and reconnect
                from core.inference import InferenceEngine
                engine_config = dict(self.config)
                engine_config["hardware"] = dict(self.config.get("hardware", {}))
                engine_config["hardware"]["inference_mode"] = "server"
                engine_config["hardware"]["inference_server"] = info.endpoint
                engine_config["hardware"]["thinking_prefix"] = info.thinking_prefix
                self._engines[key] = InferenceEngine(engine_config)
                try:
                    self._engines[key].load()
                    info.online = True
                    logger.info(f"Model '{key}' reconnected at {info.endpoint}")
                except Exception as e:
                    info.online = False
                    logger.warning(f"Model '{key}' failed to connect at {info.endpoint}: {e}")

        if "tier" in updates:
            info.tier = updates["tier"]
        if "context_window" in updates:
            info.context_window = updates["context_window"]
        if "thinking_prefix" in updates:
            info.thinking_prefix = updates["thinking_prefix"]
        if "enabled" in updates:
            info.enabled = updates["enabled"]
            self._sync_model_online(key, info.enabled and info.online)
        if "parallel" in updates:
            info.parallel = updates["parallel"]
        if "name" in updates:
            info.name = updates["name"]
        if "backend" in updates:
            info.backend = updates["backend"]
        if "claude_model" in updates:
            info.claude_model = updates["claude_model"]
            # Update capability vectors from known registry
            known = _lookup_known_model(info.name, updates["claude_model"])
            if known:
                info.reasoning_score = known["reasoning_score"]
                info.instruction_score = known["instruction_score"]
                info.stability_score = known["stability_score"]
                info.creativity_score = known["creativity_score"]
                info.tokens_per_second = known["tokens_per_sec"]
                info.tier = _tier_label(info.reasoning_score)
            # Update the adapter's model ID
            engine = self._engines.get(key)
            if isinstance(engine, AnthropicAdapter):
                engine.model_id = updates["claude_model"]
        # Capability vector updates
        for cap in ("reasoning_score", "instruction_score", "stability_score", "creativity_score"):
            if cap in updates:
                setattr(info, cap, float(updates[cap]))

        logger.info(f"Model '{key}' updated: {updates}")

        # Always try to connect after save — brings offline models online
        if info.enabled and not info.online and key in self._engines:
            try:
                self._engines[key].load()
                info.online = True
                logger.info(f"Model '{key}' now online at {info.endpoint}")
            except Exception:
                pass  # stays offline, will retry on next health check

        return True

    def get_status(self):
        """Full router status for dashboard."""
        return {
            "mode": self.mode,
            "routing_preference": getattr(self, 'preference', 'local_first'),
            "models": {k: v.to_dict() for k, v in self.models.items()},
            "task_routing": self._get_task_routing(),
            "call_log": self.call_log,
            "token_log": self.token_log,
        }

    def _get_task_routing(self):
        """Show which model handles each task type."""
        tasks = sorted(self.tasks.keys())
        routing = {}
        for task in tasks:
            task_desc = self._parse_task_def(task)
            handlers = []
            for name, info in self.models.items():
                if not info.enabled:
                    continue
                handlers.append({
                    "model": name,
                    "online": info.online,
                    "tier": info.tier,
                })
            routing[task] = {
                "handlers": handlers,
                "task_class": getattr(task_desc, 'task_class', '?'),
                "min_reasoning": getattr(task_desc, 'min_reasoning', 0),
                "pause_only": getattr(task_desc, 'pause_only', False),
                "safe_fallback": getattr(task_desc, 'safe_fallback', True),
                "prefer": getattr(task_desc, 'prefer', 'speed'),
            }
        return routing

    def persist_to_config(self, config_path):
        """Save current model config to settings.toml."""
        import toml

        try:
            with open(config_path, "r") as f:
                existing = toml.load(f)
        except Exception:
            existing = {}

        # Clear old models section
        existing.pop("models", None)

        # Write current models
        existing["models"] = {}
        for key, info in self.models.items():
            model_cfg = {
                "name": info.name,
                "endpoint": info.endpoint,
                "tier": info.tier,
                "context_window": info.context_window,
                "thinking_prefix": info.thinking_prefix,
                "backend": info.backend,
                "enabled": info.enabled,
                "parallel": info.parallel,
                "reasoning_score": info.reasoning_score,
                "instruction_score": info.instruction_score,
                "stability_score": info.stability_score,
                "creativity_score": info.creativity_score,
            }
            if info.tokens_per_second > 0:
                model_cfg["tokens_per_second"] = info.tokens_per_second
            if info.api_key_env:
                model_cfg["api_key_env"] = info.api_key_env
            if info.claude_model:
                model_cfg["claude_model"] = info.claude_model
            existing["models"][key] = model_cfg

        existing["routing"] = {
            "mode": self.mode,
        }

        with open(config_path, "w") as f:
            toml.dump(existing, f)

        # Verify write — read back and check model count matches
        try:
            with open(config_path, "r") as f:
                verify = toml.load(f)
            written = len(verify.get("models", {}))
            expected = len(self.models)
            if written != expected:
                logger.error(
                    f"Config persist MISMATCH: wrote {written} models, "
                    f"expected {expected}. File may be corrupt."
                )
            else:
                logger.info(f"Model config persisted to {config_path} ({written} models verified)")
        except Exception as ve:
            logger.error(f"Config persist verification failed: {ve}")
