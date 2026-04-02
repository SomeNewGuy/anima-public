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

"""Persistence CLI — conversation interface with memory persistence.

Commands:
    /quit, /exit        — go to sleep (end session)
    /important          — flag current conversation as high importance
    /feedback [u/d]     — thumbs up or down on last response
    /search [query]     — force a web search (manual override)
    /memory             — show memory status
    /corrections        — show recent corrections
    /beliefs            — show current beliefs (numbered)
    /beliefs remove N   — delete belief by number
    /beliefs review     — review and selectively delete beliefs
    /beliefs clear      — delete all beliefs (with confirmation)
    /freetime           — trigger an exploration window (autonomous topic exploration)
    /explorations       — view recent explorations
    /reconsolidate      — re-run sleep consolidation on recent conversations
    /paste              — enter paste mode (type /end to send)
    /help               — show commands
"""

import atexit
import os
import re
import select
import sys
import signal
import time
import logging

import toml

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.inference import InferenceEngine
from core.router import ModelRouter
from core.embeddings import EmbeddingEngine
from core.tools import detect_tool_call, execute_tool
from core.signals import detect_confusion
from memory.episodic import EpisodicMemory
from memory.semantic_rust import SemanticMemory
from memory.reflective import ReflectiveMemory
from memory.curiosity import CuriosityMemory
from memory.retrieval import ContextReconstructionEngine
from memory.explorations import ExplorationMemory
from memory.state_monitor import StateMonitor
from core.scheduler import ConsolidationScheduler
from reflection.evolution import EvolutionEngine
try:
    from reflection.exploration import ExplorationEngine
except ImportError:
    ExplorationEngine = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "..", "data", "persistence.log")),
    ],
)
logger = logging.getLogger("cli")


def load_config():
    config_path = os.environ.get(
        "ANIMA_CONFIG",
        os.path.join(os.path.dirname(__file__), "..", "config", "settings.toml"),
    )
    return toml.load(os.path.normpath(config_path))


# LaTeX cleanup — Qwen sometimes wraps text in LaTeX notation
_LATEX_TEXT_RE = re.compile(
    r'\$\\text\{([^}]*)\}\$'       # $\text{content}$  → content
)
_LATEX_CMD_RE = re.compile(
    r'\\(?:text|textbf|textit|emph|mathrm)\{([^}]*)\}'  # \textbf{content} → content
)
_LATEX_BARE_MATH_RE = re.compile(
    r'\$([^$\\]+)\$'               # $simple text$ → simple text (no backslashes inside)
)


_THINK_RE = re.compile(r'<think>.*?</think>\s*', re.DOTALL)


def strip_latex(text):
    """Remove LaTeX markup from model output, preserving the content inside."""
    text = _LATEX_TEXT_RE.sub(r'\1', text)
    text = _LATEX_CMD_RE.sub(r'\1', text)
    text = _LATEX_BARE_MATH_RE.sub(r'\1', text)
    return text


def strip_think(text):
    """Remove Qwen3 <think>...</think> blocks and /no_think artifacts from model output."""
    text = _THINK_RE.sub('', text)
    text = text.replace('/no_think', '').replace(' /no_think', '')
    return text


class PersistenceCLI:
    def __init__(self):
        self.config = load_config()
        self.inference = None
        self.embeddings = None
        self.episodic = None
        self.semantic = None
        self.reflective = None
        self.curiosity = None
        self.retrieval = None
        self.ner = None
        self.evolution = None
        self.explorations = None
        self.exploration_engine = None
        self.scheduler = None
        self.state_monitor = None
        self.current_episode_id = None
        self.last_turn_id = None
        self.turn_count = 0
        self.conversation_history = []  # accumulated (role, content) pairs for this session
        self._sleeping = False
        self._web_mode = False
        self._pending_notifications = []
        # Nap state
        self._nap_count = 0
        self._nap_summaries = []
        # Fatigue state
        self._confusion_count = 0
        self._fatigue_denial_turn = 0
        # Exploration window state
        self._last_exploration_turn = 0

    def _print(self, *args, **kwargs):
        """Print only when not in web mode."""
        if not self._web_mode:
            print(*args, **kwargs)

    def wake(self):
        """Wake up — initialize all components, restore long-term memory."""
        self._print(f"\n  Persistence — Waking up...\n")

        # Memory systems
        self._print("  Loading memory systems...", flush=True)
        self.episodic = EpisodicMemory(self.config)
        self.episodic.initialize()
        self.semantic = SemanticMemory(self.config)
        self.semantic.initialize()
        self.reflective = ReflectiveMemory(self.config)
        self.reflective.initialize()
        self.curiosity = CuriosityMemory(self.config)
        self.curiosity.initialize()
        self.explorations = ExplorationMemory(self.config)
        self.explorations.initialize()

        # Embedding model
        self._print("  Loading embedding model...", flush=True)
        self.embeddings = EmbeddingEngine(self.config)
        self.embeddings.load()

        # NER (load once, reuse)
        from core.ner import EntityExtractor
        self.ner = EntityExtractor()
        self.ner.load()

        # Document retrieval (DMS layer — optional, only when datafiles_dir configured)
        doc_retrieval = None
        if self.config.get("extraction", {}).get("datafiles_dir"):
            from plugins._template.ingestion.document_retrieval import DocumentRetrieval
            doc_retrieval = DocumentRetrieval(
                self.semantic.db_conn, self.config, semantic_memory=self.semantic,
            )

        # Context reconstruction engine
        self.retrieval = ContextReconstructionEngine(
            self.config, self.episodic, self.semantic, self.reflective, self.embeddings,
            curiosity_memory=self.curiosity,
            exploration_memory=self.explorations,
            document_retrieval=doc_retrieval,
        )

        # Inference engine — ModelRouter wraps single or multiple InferenceEngines
        self.router = ModelRouter(self.config)
        if self.router.models:
            if self.router.mode == "multi":
                model_names = [m.name for m in self.router.models.values()]
                self._print(f"  Multi-model routing: {', '.join(model_names)}", flush=True)
            else:
                info = next(iter(self.router.models.values()))
                self._print(f"  Connecting to inference server ({info.endpoint})...", flush=True)
            self.router.load()
        else:
            self._print("  No models configured — add via dashboard Settings", flush=True)
        # Backward compat: self.inference points to router (same interface)
        self.inference = self.router

        # Evolution engine — sleep consolidation and belief formation
        self.evolution = EvolutionEngine(
            self.config, self.inference, self.embeddings,
            self.episodic, self.semantic, self.reflective,
            curiosity=self.curiosity,
        )

        # Exploration engine — autonomous topic exploration (optional, research plugin)
        self.exploration_engine = None
        if ExplorationEngine and self.config.get("exploration", {}).get("enabled", False):
            self.exploration_engine = ExplorationEngine(
                self.config, self.inference, self.curiosity, self.semantic,
                self.explorations,
            )

        # Consolidation scheduler — decoupled trigger logic
        self.scheduler = ConsolidationScheduler(
            self.config,
            consolidate_fn=lambda: self._sleep_cycle(extraction_context="exploration"),
            episode_fn=lambda: self.episodic.create_episode(),
        )

        # State monitor — silent observability layer (separate telemetry.db)
        self.state_monitor = StateMonitor(
            self.config, self.episodic, self.semantic,
            self.curiosity, self.explorations,
            embeddings=self.embeddings,
        )
        self.state_monitor.initialize()

        # Give retrieval engine access to state telemetry (read-only)
        self.retrieval.state_monitor = self.state_monitor

        # Give exploration engine access to state monitor for window telemetry
        if self.exploration_engine:
            self.exploration_engine.state_monitor = self.state_monitor
            self.exploration_engine.embeddings = self.embeddings

        # Process any missed sleep from crashed/interrupted sessions
        if self.router and getattr(self.router, 'models', None):
            self.evolution.process_missed_sleep(queue_mode=self._web_mode)
        else:
            logger.warning("No models configured — skipping missed sleep consolidation")

        # Start a new episode
        self.current_episode_id = self.episodic.create_episode()
        self.turn_count = 0
        self.conversation_history = []

        # Log wake state
        if self.state_monitor:
            self.state_monitor.log_state("wake", episode_id=self.current_episode_id)

        # Status display
        inference_label = "multi-model" if self.router.mode == "multi" else ("GPU server" if self.config.get("hardware", {}).get("inference_mode") == "server" else "CPU local")
        ep_count = self.episodic.get_episode_count()
        belief_count = self.semantic.get_belief_count()
        obs_count = self.reflective.get_observation_count()

        self._print(f"\n  Ready. Inference: {inference_label} | Tools: search, calculate")
        status_parts = [
            f"{ep_count} episodes",
            f"{belief_count} beliefs",
            f"{obs_count} observations",
        ]
        self._print(f"  Memory: {' | '.join(status_parts)}")
        if ep_count <= 1:
            self._print(f"  This is a fresh start. Memory will build over time.")

        # Surface recent explorations
        if self.explorations:
            recent_exp = self.explorations.get_accepted(limit=3)
            if recent_exp:
                count = len(recent_exp)
                topic_preview = recent_exp[0]["trigger_text"][:60]
                self._print(f"\n  Recent explorations: {count} topic(s).")
                self._print(f"  Latest: \"{topic_preview}\"")

        self._print(f"  Type /help for commands.\n")

    def run(self):
        """Main conversation loop — waking life."""
        self.wake()

        prompt = self.config["interface"]["prompt"]

        while True:
            try:
                user_input = self._get_input_with_timeout(prompt)
            except (EOFError, KeyboardInterrupt):
                self.sleep()
                break

            if user_input is None:
                # Idle timeout — go to sleep
                print("\n  (Idle timeout — going to sleep...)")
                self.sleep()
                break

            if not user_input:
                continue

            # Paste mode: accumulate lines until /end
            if user_input.lower() == "/paste":
                pasted = self._collect_paste()
                if pasted:
                    if not self._process_turn(pasted):
                        break
                continue

            # Handle commands
            if user_input.startswith("/"):
                if self._handle_command(user_input):
                    continue
                else:
                    break  # /quit or /exit

            # Process conversation turn
            if not self._process_turn(user_input):
                break  # fatigue-initiated sleep

    def _collect_paste(self):
        """Collect multi-line paste input until /end."""
        print("  Paste mode. Type /end on its own line to send.\n")
        lines = []
        while True:
            try:
                line = input("... ")
            except (EOFError, KeyboardInterrupt):
                print("\n  Paste cancelled.\n")
                return None
            if line.strip().lower() == "/end":
                break
            lines.append(line)
        text = "\n".join(lines).strip()
        if not text:
            print("  Empty paste, nothing sent.\n")
            return None
        char_count = len(text)
        line_count = len(lines)
        print(f"  Pasted {line_count} lines ({char_count} chars).\n")
        return text

    def _get_input_with_timeout(self, prompt):
        """Read input with optional idle timeout. Returns None on timeout."""
        timeout_min = self.config.get("interface", {}).get("idle_timeout_minutes", 0)
        if timeout_min <= 0:
            return input(prompt).strip()

        timeout_sec = timeout_min * 60
        sys.stdout.write(prompt)
        sys.stdout.flush()
        ready, _, _ = select.select([sys.stdin], [], [], timeout_sec)
        if ready:
            return sys.stdin.readline().rstrip("\n").strip()
        return None  # timeout

    def _process_turn(self, user_input, force_search=None, capture_output=False):
        """Process a single conversation turn with optional tool use loop.

        When capture_output=True (web mode): suppress terminal printing, return
        dict with {response, notifications} instead of bool.
        """
        self._pending_notifications = []
        self._explore_done_this_turn = False
        # Store user turn
        entities = self.ner.extract_names(user_input)
        user_turn_id = self.episodic.add_turn(
            self.current_episode_id, "user", user_input, entities=entities
        )

        # Reconstruct context (exclude current episode from retrieval)
        context = self.retrieval.reconstruct(
            user_input, current_episode_id=self.current_episode_id
        )
        system_context = context["system_context"]

        # Inject nap context so the model knows what was discussed before context refresh
        if self._nap_summaries:
            nap_lines = ["[NAP CONTEXT — Earlier in this conversation, before a context refresh:]"]
            for ns in self._nap_summaries:
                nap_lines.append(f"- {ns.get('summary', 'No summary')}")
            system_context += "\n\n" + "\n".join(nap_lines)

        # If forced search, inject results before first generation
        if force_search:
            if not capture_output:
                print(f"  Searching: {force_search}...", flush=True)
            search_results = execute_tool("search", force_search)
            system_context += f"\n\n{search_results}"
            logger.info(f"Manual search triggered: {force_search}")

        # Build message history — system prompt + full conversation so far
        messages = [{"role": "system", "content": system_context}]

        # Include prior turns from this session so the model can see
        # what it and the operator have already said (conversation continuity).
        # Budget: keep conversation history within ~60% of remaining context
        # after system prompt. Rough estimate: 4 chars ≈ 1 token.
        context_window = self.config["model"]["context_window"]
        system_tokens = len(system_context) // 4
        response_reserve = self.config["model"]["max_response_tokens"]
        history_budget = context_window - system_tokens - response_reserve - 100
        history_budget_chars = max(0, history_budget * 4)

        # Walk backwards through history to keep most recent turns
        history_to_include = []
        chars_used = len(user_input)
        for role, content in reversed(self.conversation_history):
            chars_used += len(content)
            if chars_used > history_budget_chars:
                break
            history_to_include.insert(0, (role, content))

        for role, content in history_to_include:
            messages.append({"role": role, "content": content})

        # Add the current user message
        messages.append({"role": "user", "content": user_input + " /no_think"})

        start_time = time.time()

        # Complexity gate — short conversational messages should never trigger tools.
        # Base model can parrot tool examples from system prompt on simple inputs.
        _stripped_input = user_input.strip().lower().rstrip("!?.:;,)")
        _is_simple = (
            len(user_input.split()) <= 6
            or _stripped_input in (
                "hi", "hey", "hello", "hi an", "hey an", "good morning",
                "good evening", "good night", "thanks", "thank you", "ok",
                "yes", "no", "sure", "bye", "goodbye", "how are you",
            )
            or context.get("query_class") == "emotional"
        )

        # Tool use loop — max 3 iterations to prevent infinite loops
        max_tool_rounds = 3
        for tool_round in range(max_tool_rounds):
            # Generate response (non-streaming for first pass to check for tool calls)
            response = self.inference.generate_with_messages(messages, max_tokens=1024)
            response_text = strip_think(strip_latex(response.strip()))

            # Check for tool call — skip detection on simple messages
            if _is_simple:
                tool_name, tool_query = None, None
            else:
                tool_name, tool_query = detect_tool_call(response_text)

            if tool_name is None:
                # No tool call — this is the final response.
                # Strip any tool call artifacts the model may have generated
                # (base model can parrot tool examples even when gated)
                if _is_simple:
                    response_text = re.sub(
                        r'\[TOOL:.*?\]\s*\[QUERY:.*?\]', '', response_text
                    ).strip()
                if not capture_output:
                    print()
                    print(response_text)
                    print()
                break
            else:
                # Tool call detected — execute it
                if not capture_output:
                    print(f"\n  [{tool_name}] {tool_query}...", flush=True)

                # Intercept explore tool — run exploration window or single topic
                # Only allow one explore per turn to prevent loop
                _explore_done = getattr(self, '_explore_done_this_turn', False)
                if tool_name == "explore" and self.exploration_engine and not _explore_done:
                    self._explore_done_this_turn = True
                    _explore_query = tool_query.strip().lower()
                    _is_session = _explore_query in (
                        "session", "full", "window", "exploration window",
                        "explore", "start session", "start exploration",
                    ) or not tool_query.strip()

                    if _is_session:
                        # Full exploration window — multiple topics
                        results = self.exploration_engine.explore_session(
                            episode_id=self.current_episode_id
                        )
                        if results:
                            tool_result = self._format_exploration_report(results)
                            self._last_exploration_turn = self.turn_count
                        else:
                            tool_result = '[EXPLORATION WINDOW: No topics available to explore.]'
                    else:
                        # Single topic exploration
                        result = self.exploration_engine.explore(
                            episode_id=self.current_episode_id
                        )
                        if result:
                            parts = [f'[EXPLORATION COMPLETE: "{tool_query}"]']
                            if result.get("confirmed"):
                                parts.append(f'Confirmed: {result["confirmed"]}')
                            if result.get("inferred"):
                                parts.append(f'Inferred: {result["inferred"]}')
                            if result.get("uncertain"):
                                parts.append(f'Uncertain: {result["uncertain"]}')
                            if result.get("open_questions"):
                                parts.append(f'Open questions: {"; ".join(result["open_questions"])}')
                            tool_result = "\n".join(parts)
                        else:
                            tool_result = f'[EXPLORATION: No results for "{tool_query}"]'
                elif tool_name == "explore" and _explore_done:
                    tool_result = '[EXPLORATION: Already completed this turn. Report your findings.]'
                else:
                    tool_result = execute_tool(tool_name, tool_query)

                # Add the model's tool call and results to message history
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": tool_result})

                logger.info(f"Tool round {tool_round + 1}: {tool_name}({tool_query})")

                # Continue loop — model will generate again with tool results
        else:
            # Hit max tool rounds
            if not capture_output:
                print()
                print(response_text)
                print()
            logger.warning(f"Hit max tool rounds ({max_tool_rounds})")

        elapsed = time.time() - start_time
        full_response = response_text

        # Accumulate conversation history for multi-turn continuity
        self.conversation_history.append(("user", user_input))
        self.conversation_history.append(("assistant", full_response))

        # Store assistant turn (final response only, not intermediate tool calls)
        self.last_turn_id = self.episodic.add_turn(
            self.current_episode_id, "assistant", full_response
        )
        self.turn_count += 1

        # Real-time curiosity: detect gaps and resolve questions (no inference calls)
        self._detect_gaps(user_input, full_response)
        self._check_resolutions(user_input, full_response)

        # Micro-consolidation (lightweight note-taking during waking life)
        if self.evolution:
            self.evolution.micro_consolidate(
                self.current_episode_id, self.turn_count
            )

        # Log telemetry
        logger.info(
            f"Turn {self.turn_count}: "
            f"class={context['query_class']} "
            f"context_tokens~={context['budget_used']} "
            f"response_chars={len(full_response)} "
            f"time={elapsed:.1f}s"
        )

        # State monitor — silent, no output
        if self.state_monitor:
            self.state_monitor.log_state("turn", episode_id=self.current_episode_id)

        # Capacity check — nap if conversation history is getting large
        self._check_capacity()

        # Fatigue check — propose nap or sleep if confusion rate is high
        if not self._check_fatigue():
            if capture_output:
                return {"response": full_response, "notifications": list(self._pending_notifications)}
            return False

        # Auto-trigger exploration window if conditions warrant (web mode only)
        self._check_exploration_trigger()

        if capture_output:
            return {"response": full_response, "notifications": list(self._pending_notifications)}
        return True

    # -------------------------------------------------------------------
    # Real-time curiosity — gap detection and resolution (no inference)
    # -------------------------------------------------------------------

    def _detect_gaps(self, user_input, response):
        """Detect knowledge gaps from hedging/deferral in the response.

        Pure heuristic — no inference call. If the response contains confusion
        signals, extract the topic from user_input via NER and store as an
        open question.
        """
        cur_cfg = self.config.get("curiosity", {})
        if not cur_cfg.get("auto_detect_gaps", True):
            return
        if not self.curiosity or not self.ner:
            return

        is_confused, signal_type, matched = detect_confusion(response)
        if is_confused:
            self._confusion_count += 1
        if not is_confused:
            return

        # Extract topic from user input
        entities = self.ner.extract_names(user_input)

        # Build question text — must be a genuine question, not verbatim user input.
        # Only create a gap when we can form a real question:
        # 1. User input is already a question (ends with ?)
        # 2. We extracted entities and can ask "What is {entity}?"
        # If neither applies, the hedging alone isn't an actionable gap.
        user_is_question = user_input.rstrip().endswith("?")

        if entities:
            topic = entities[0]
            question_text = f"What is {topic}?"
        elif user_is_question and len(user_input) < 120:
            topic = user_input
            question_text = user_input
        else:
            # User made a statement, ANIMA hedged — not an actionable gap
            logger.info(
                f"Gap detection: hedging on statement, no entities extracted. "
                f"Skipping: '{user_input[:60]}'"
            )
            return

        # Check for duplicate open questions on same topic
        existing = self.curiosity.get_open_questions(limit=20)
        for eq in existing:
            if eq["question"].lower() == question_text.lower():
                logger.info(f"Gap detection: duplicate question skipped: {question_text[:60]}")
                return
            # Check topic_tags overlap
            eq_tags = {t.lower() for t in eq.get("topic_tags", [])}
            if topic.lower() in eq_tags:
                logger.info(f"Gap detection: topic already tracked: {topic}")
                return

        # Deferral = high priority (hard gap), hedging = medium (soft uncertainty)
        priority = "high" if signal_type == "deferral" else "medium"
        topic_tags = [e.lower() for e in entities] if entities else []

        self.curiosity.add_question(
            question=question_text,
            question_type="knowledge_gap",
            context=f"Detected {signal_type}: '{matched}' in response",
            topic_tags=topic_tags,
            source_episode=self.current_episode_id,
            priority=priority,
            embedder=self.embeddings,
        )
        logger.info(
            f"Gap detected [{signal_type}]: '{question_text[:60]}' "
            f"(signal: '{matched}', priority: {priority})"
        )

    def _check_resolutions(self, user_input, response):
        """Auto-resolve open questions when the response is confident on a matching topic.

        Pure heuristic — no inference call. If open questions exist on topics
        matching the current input, and the response has no confusion signals,
        resolve them and create beliefs from the answers.
        """
        cur_cfg = self.config.get("curiosity", {})
        if not cur_cfg.get("auto_resolve", True):
            return
        if not self.curiosity or not self.ner:
            return

        # Only resolve if response is confident (no confusion signals)
        is_confused, _, _ = detect_confusion(response)
        if is_confused:
            return

        # Extract topics from user input
        entities = self.ner.extract_names(user_input)
        if not entities:
            return

        # Find open questions matching current topics
        matched_questions = self.curiosity.get_questions_by_topics(
            entities, status="open", limit=3
        )
        if not matched_questions:
            return

        show_notification = cur_cfg.get("resolution_notification", True)
        conf_threshold = cur_cfg.get("gap_confidence_threshold", 0.3)

        for q in matched_questions:
            # Use user_input as the answer (operator provided the info)
            answer = user_input
            self.curiosity.resolve_question(
                q["id"], answer, answered_by="conversation"
            )

            # Create/reinforce belief from the answer
            if self.evolution:
                match = self.evolution._find_similar_belief(answer)
                if match:
                    belief_id, similarity = match
                    self.evolution.semantic.update_belief(
                        belief_id,
                        new_confidence=min(0.9, conf_threshold + 0.4),
                        reason=f"Resolved question: {q['question'][:60]}",
                        episode_id=self.current_episode_id,
                    )
                else:
                    from core.proposals import BeliefProposal
                    proposal = BeliefProposal(
                        statement=answer,
                        confidence=conf_threshold + 0.3,
                        source="conversation",
                        source_episode=self.current_episode_id,
                        extraction_context="operator",
                        supporting_evidence=[
                            f"Resolved question: {q['question'][:100]}"
                        ],
                    )
                    triage_result = self.evolution.gateway.submit(proposal)
                    if triage_result.decision == "accept":
                        self.evolution.semantic.add_belief(
                            statement=answer,
                            confidence=conf_threshold + 0.3,
                            source_episode=self.current_episode_id,
                            supporting_evidence=[
                                f"Resolved question: {q['question'][:100]}"
                            ],
                        )

            if show_notification:
                snippet = q["question"][:60]
                msg = f"Learned: {snippet}"
                if self._web_mode:
                    self._pending_notifications.append(msg)
                else:
                    print(f"  ({msg})")

            logger.info(
                f"Auto-resolved question: '{q['question'][:60]}' "
                f"answer: '{answer[:60]}'"
            )

    # -------------------------------------------------------------------
    # Nap — mid-session context refresh
    # -------------------------------------------------------------------

    def _check_capacity(self):
        """Check if conversation history is approaching context window capacity."""
        nap_cfg = self.config.get("nap", {})
        if not nap_cfg.get("enabled", True):
            return

        threshold = nap_cfg.get("capacity_threshold", 0.70)
        context_window = self.config["model"]["context_window"]

        history_tokens = sum(len(c) for _, c in self.conversation_history) // 4
        if context_window > 0 and history_tokens / context_window >= threshold:
            self._do_nap()

    def _do_nap(self):
        """Execute a nap — consolidate and trim conversation history."""
        nap_cfg = self.config.get("nap", {})
        notify = nap_cfg.get("notify", True)
        retain = nap_cfg.get("post_nap_retain_turns", 2)

        if notify:
            if self._web_mode:
                self._pending_notifications.append("Napping — consolidating context...")
            else:
                print("  (Napping — consolidating context...)", flush=True)

        summary = None
        if self.evolution:
            summary = self.evolution.nap(self.current_episode_id)

        if summary:
            self._nap_summaries.append(summary)

        # Trim conversation history — keep last N turn pairs (each pair = 2 entries)
        keep_entries = retain * 2
        if len(self.conversation_history) > keep_entries:
            self.conversation_history = self.conversation_history[-keep_entries:]

        self._nap_count += 1

        if notify:
            if self._web_mode:
                self._pending_notifications.append("Awake — context refreshed.")
            else:
                print("  (Awake — context refreshed.)", flush=True)

        logger.info(
            f"Nap #{self._nap_count}: trimmed to {len(self.conversation_history)} entries, "
            f"summary={'yes' if summary else 'no'}"
        )

    # -------------------------------------------------------------------
    # Fatigue — self-initiated sleep via confusion monitoring
    # -------------------------------------------------------------------

    def _assess_fatigue(self):
        """Assess whether confusion rate warrants proposing nap or sleep.

        Returns "ok", "nap", or "sleep".
        """
        fat_cfg = self.config.get("fatigue", {})
        # Web mode always enables fatigue (no interactive gate to protect against)
        if not self._web_mode and not fat_cfg.get("enabled", False):
            return "ok"
        if self.turn_count < fat_cfg.get("min_turns_before_check", 5):
            return "ok"

        # Cooldown: skip if operator denied a proposal within last 3 turns
        if self.turn_count - self._fatigue_denial_turn < 3:
            return "ok"

        window = fat_cfg.get("window_turns", 6)
        nap_threshold = fat_cfg.get("nap_confusion_rate", 0.5)
        sleep_threshold = fat_cfg.get("sleep_confusion_rate", 0.7)

        # Scan recent assistant responses for confusion signals
        recent_assistant = [
            content for role, content in self.conversation_history[-(window * 2):]
            if role == "assistant"
        ]
        if not recent_assistant:
            return "ok"

        confused_count = sum(1 for r in recent_assistant if detect_confusion(r)[0])
        rate = confused_count / len(recent_assistant)

        # Escalation: if already napped and still confused, propose sleep
        if self._nap_count > 0 and rate >= nap_threshold:
            return "sleep"
        if rate >= sleep_threshold:
            return "sleep"
        if rate >= nap_threshold:
            return "nap"
        return "ok"

    def _sleep_cycle(self, extraction_context="operator"):
        """Autonomous sleep cycle for continuous operation — consolidate + start fresh episode.

        Unlike sleep(), this does NOT close memory systems or unload inference.
        It runs consolidation on the current episode, then starts a new one.
        extraction_context: 'operator' or 'exploration' — controls belief triage thresholds.
        """
        if not self.current_episode_id or self.turn_count == 0:
            return

        self._pending_notifications.append("Sleeping — consolidating...")
        logger.info(f"Sleep cycle starting for episode {self.current_episode_id[:8]}")

        # Consolidate current episode
        if self.evolution:
            try:
                self.evolution.consolidate(
                    self.current_episode_id, interactive=False, queue_mode=True,
                    extraction_context=extraction_context,
                )
            except Exception as e:
                logger.warning(f"Sleep cycle consolidation failed: {e}")
                try:
                    self.episodic.mark_episode_consolidated(self.current_episode_id)
                except Exception:
                    pass

        # Log sleep state
        if self.state_monitor:
            self.state_monitor.log_state("sleep", episode_id=self.current_episode_id)

        # Start fresh episode — carry forward the context type
        self.current_episode_id = self.episodic.create_episode(
            context_type=extraction_context,
        )
        self.turn_count = 0
        self.conversation_history = []
        self._nap_count = 0
        self._nap_summaries = []
        self._confusion_count = 0

        # Log wake state
        if self.state_monitor:
            self.state_monitor.log_state("wake", episode_id=self.current_episode_id)

        self._pending_notifications.append("Awake — new cycle started.")
        logger.info(f"Sleep cycle complete. New episode: {self.current_episode_id[:8]}")

    def _check_fatigue(self):
        """Propose nap or sleep if fatigue is detected. Returns True to continue, False to exit."""
        assessment = self._assess_fatigue()
        if assessment == "ok":
            return True

        # Web mode: autonomous nap/sleep without prompts
        if self._web_mode:
            if assessment == "nap":
                self._do_nap()
            elif assessment == "sleep":
                # Use exploration context if in soak mode
                ctx = "operator"
                if hasattr(self, 'exploration_engine') and getattr(self.exploration_engine, '_soak_mode', False):
                    ctx = "exploration"
                self._sleep_cycle(extraction_context=ctx)
            return True

        if assessment == "nap":
            print("\n  (I'm having trouble staying coherent. A nap might help.)")
            try:
                answer = input("  Nap? [y/n] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = "n"

            if answer in ("y", "yes"):
                self._do_nap()
                return True
            else:
                self._fatigue_denial_turn = self.turn_count
                return True

        if assessment == "sleep":
            print("\n  (I'm struggling — sleep might help more than continuing.)")
            try:
                answer = input("  Sleep? [y/n] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = "n"

            if answer in ("y", "yes"):
                self.sleep()
                return False
            else:
                self._fatigue_denial_turn = self.turn_count
                return True

        return True

    # -------------------------------------------------------------------
    # Exploration windows — autonomous topic exploration
    # -------------------------------------------------------------------

    def _format_exploration_report(self, results):
        """Build system message from exploration results."""
        parts = [
            f"[SYSTEM: You just completed an exploration window. You explored "
            f"{len(results)} topic(s). Below are your findings classified by "
            f"evidence strength. Report what you found and what remains "
            f"uncertain. Do NOT end with a question.]\n"
        ]
        for i, result in enumerate(results, 1):
            search_label = "web search" if result["search_used"] else "your own reasoning"
            priority = result.get("priority_ai", "normal")
            priority_tag = f" [{priority.upper()}]" if priority != "normal" else ""
            parts.append(f"--- Exploration {i} (via {search_label}){priority_tag} ---")
            parts.append(f"Topic: {result['trigger_text']}")
            if result.get("confirmed"):
                parts.append(f"Confirmed: {result['confirmed']}")
            if result.get("inferred"):
                parts.append(f"Inferred: {result['inferred']}")
            if result.get("uncertain"):
                parts.append(f"Uncertain: {result['uncertain']}")
            if result.get("open_questions"):
                parts.append(f"Open questions: {'; '.join(result['open_questions'][:3])}")
            parts.append("")
        return "\n".join(parts)

    def _run_exploration_window(self):
        """Run an exploration window. Used by both /freetime command and auto-trigger."""
        try:
            results = self.exploration_engine.explore_session(
                episode_id=self.current_episode_id
            )
            if results:
                if not self._web_mode:
                    # Print summary of each exploration
                    for i, result in enumerate(results, 1):
                        search_label = "searched" if result["search_used"] else "internal"
                        print(f"  [{i}] {result['trigger_text'][:70]}")
                        print(f"      Confidence: {result['internal_confidence']:.2f} ({search_label})")
                        if result["queries"]:
                            print(f"      Queries: {', '.join(result['queries'][:3])}")
                    print(f"\n  {len(results)} exploration(s) complete.\n")

                system_msg = self._format_exploration_report(results)
                self._process_turn(system_msg, capture_output=self._web_mode)
                self._last_exploration_turn = self.turn_count
            else:
                if not self._web_mode:
                    print("  Nothing to explore — no open gaps or topics.\n")
        except Exception as e:
            logger.warning(f"Exploration window failed: {e}")
            if not self._web_mode:
                print(f"  Exploration failed: {e}\n")

    def _check_exploration_trigger(self):
        """Auto-trigger exploration window if conditions warrant. Web mode only."""
        if not self._web_mode or self._sleeping:
            return
        if not self.exploration_engine:
            return

        exp_cfg = self.config.get("exploration", self.config.get("freetime", {}))
        if not exp_cfg.get("auto_trigger", True):
            return

        # Don't explore if we just explored (minimum turn gap)
        min_gap = exp_cfg.get("auto_trigger_min_turn_gap", 3)
        if self.turn_count - self._last_exploration_turn < min_gap:
            return

        curiosity_threshold = exp_cfg.get("auto_trigger_curiosity_threshold", 0.3)

        # Combined signals from last state
        try:
            import sqlite3 as _sqlite3
            base = os.path.join(os.path.dirname(__file__), "..")
            tel_path = os.path.normpath(os.path.join(base, "data", "telemetry.db"))
            conn = _sqlite3.connect(tel_path)
            conn.row_factory = _sqlite3.Row
            row = conn.execute(
                "SELECT curiosity_pressure, coherence_confidence FROM state_log "
                "ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            conn.close()
            if not row:
                return

            curiosity = row["curiosity_pressure"] or 0
            coherence = row["coherence_confidence"] or 0

            # Conditions: curiosity building + coherence healthy
            if curiosity >= curiosity_threshold and coherence >= 0.70:
                logger.info(
                    f"Auto-triggering exploration window: "
                    f"curiosity={curiosity:.2f}, coherence={coherence:.2f}"
                )
                self._run_exploration_window()
        except Exception:
            pass

    def _handle_command(self, command):
        """Handle a CLI command. Returns True to continue, False to exit."""
        parts = command.split(maxsplit=1)
        cmd = parts[0].lower()

        if cmd in ("/quit", "/exit"):
            self.sleep()
            return False

        elif cmd == "/help":
            print(__doc__)

        elif cmd == "/important":
            if self.current_episode_id:
                self.episodic.update_episode_metadata(
                    self.current_episode_id, importance=1.0
                )
                print("  Flagged current conversation as high importance.\n")

        elif cmd == "/feedback":
            fb = parts[1].strip() if len(parts) > 1 else None
            if fb in ("u", "up"):
                self.reflective.record_feedback(
                    "up", episode_id=self.current_episode_id, turn_id=self.last_turn_id
                )
                print("  Recorded: thumbs up.\n")
            elif fb in ("d", "down"):
                self.reflective.record_feedback(
                    "down", episode_id=self.current_episode_id, turn_id=self.last_turn_id
                )
                print("  Recorded: thumbs down.\n")
            else:
                print("  Usage: /feedback u (thumbs up) or /feedback d (thumbs down)\n")

        elif cmd == "/search":
            query = parts[1].strip() if len(parts) > 1 else None
            if not query:
                print("  Usage: /search <query>\n")
            else:
                # Force search and let model respond with results
                self._process_turn(
                    f"Search for and tell me about: {query}",
                    force_search=query,
                )

        elif cmd == "/memory":
            ep_count = self.episodic.get_episode_count()
            turn_count = self.episodic.get_turn_count()
            belief_count = self.semantic.get_belief_count()
            obs_count = self.reflective.get_observation_count()
            print(f"  Episodes: {ep_count}")
            print(f"  Turns stored: {turn_count}")
            print(f"  Beliefs: {belief_count}")
            print(f"  Self-observations: {obs_count}")
            if self.explorations:
                exp_total = self.explorations.get_exploration_count()
                exp_pending = self.explorations.get_exploration_count(status="preliminary")
                print(f"  Explorations: {exp_total} ({exp_pending} pending)")
            print()

        elif cmd == "/corrections":
            corrections = self.episodic.get_corrections()
            if not corrections:
                print("  No corrections recorded yet.\n")
            else:
                for c in corrections[:10]:
                    print(f"  - Was: {c['original_position'][:80]}")
                    print(f"    Now: {c['corrected_position'][:80]}")
                    print(f"    By: {c['corrected_by']}\n")

        elif cmd == "/beliefs":
            subcmd = parts[1].strip().lower() if len(parts) > 1 else ""
            beliefs = self.semantic.search_beliefs()

            if not beliefs:
                print("  No beliefs formed yet.\n")
            elif subcmd == "clear":
                count = len(beliefs)
                try:
                    confirm = input(
                        f"  Delete all {count} beliefs? This cannot be undone. [y/n] "
                    ).strip().lower()
                except (EOFError, KeyboardInterrupt):
                    confirm = "n"
                if confirm in ("y", "yes"):
                    self.semantic.delete_all_beliefs()
                    print(f"  Deleted {count} beliefs.\n")
                else:
                    print("  Cancelled.\n")
            elif subcmd.startswith("remove"):
                # /beliefs remove N — delete by index
                arg = subcmd[6:].strip()
                if not arg:
                    # Check if there's more after "remove" in the original parts
                    full_arg = parts[1].strip() if len(parts) > 1 else ""
                    arg = full_arg.split(None, 1)[1] if " " in full_arg else ""
                try:
                    idx = int(arg) - 1  # 1-indexed for display
                    if 0 <= idx < len(beliefs):
                        b = beliefs[idx]
                        self.semantic.delete_belief(b["id"])
                        print(f"  Deleted belief {idx + 1}: {b['statement'][:80]}\n")
                    else:
                        print(f"  Invalid index. Use /beliefs to see numbered list.\n")
                except ValueError:
                    print(f"  Usage: /beliefs remove N (where N is the belief number)\n")
            elif subcmd == "review":
                print("  Review beliefs — [k]eep or [d]elete each:\n")
                deleted = 0
                for i, b in enumerate(beliefs):
                    print(f"  {i + 1}. [{b['confidence']:.1f}] {b['statement'][:100]}")
                    try:
                        ans = input("    [k]eep / [d]elete? ").strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        print("\n  Interrupted. Skipping remaining.")
                        break
                    if ans in ("d", "delete"):
                        self.semantic.delete_belief(b["id"])
                        deleted += 1
                        print("    Deleted.")
                    print()
                print(f"  Deleted {deleted} beliefs.\n")
            else:
                for i, b in enumerate(beliefs[:10]):
                    print(f"  {i + 1}. [{b['confidence']:.1f}] {b['statement'][:100]}")

        elif cmd == "/reconsolidate":
            if not self.evolution:
                print("  Evolution engine not available.\n")
            else:
                # Get recent consolidated episodes with enough turns
                episodes = self.episodic.get_recent_episodes(limit=10)
                candidates = [
                    ep for ep in episodes
                    if ep.get("summarized") == 1  # summarized=1 means consolidated
                    and ep.get("turn_count", 0) >= self.evolution.min_turns
                    and ep.get("id") != self.current_episode_id
                ]
                if not candidates:
                    print("  No recent conversations to reconsolidate.\n")
                else:
                    print(f"  Found {len(candidates)} conversation(s):\n")
                    for i, ep in enumerate(candidates):
                        summary = (ep.get("summary") or "No summary")[:70]
                        tc = ep.get("turn_count", 0)
                        print(f"  [{i + 1}] {summary}  ({tc} turns)")
                    print(f"\n  [a]ll | Enter number | [c]ancel")
                    try:
                        choice = input("  > ").strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        choice = "c"
                    if choice == "a":
                        for ep in candidates:
                            self.episodic.mark_episode_unconsolidated(ep["id"])
                        print(f"  Marked {len(candidates)} for reconsolidation.")
                        for ep in candidates:
                            self.evolution.consolidate(ep["id"], interactive=True)
                    elif choice == "c":
                        print("  Cancelled.\n")
                    else:
                        try:
                            idx = int(choice) - 1
                            if 0 <= idx < len(candidates):
                                ep = candidates[idx]
                                self.episodic.mark_episode_unconsolidated(ep["id"])
                                self.evolution.consolidate(ep["id"], interactive=True)
                            else:
                                print(f"  Invalid number.\n")
                        except ValueError:
                            print(f"  Invalid input.\n")

        elif cmd == "/freetime":
            exp_cfg = self.config.get("exploration", self.config.get("freetime", {}))
            if not exp_cfg.get("enabled", True) or not self.exploration_engine:
                print("  Exploration engine not available.\n")
            else:
                max_exp = exp_cfg.get("max_explorations_per_session", 3)
                print(f"  Exploration window — exploring up to {max_exp} topics...\n", flush=True)
                self._run_exploration_window()

        elif cmd == "/explorations":
            if not self.explorations:
                print("  Exploration memory not available.\n")
            else:
                subcmd = parts[1].strip().lower() if len(parts) > 1 else ""
                recent = self.explorations.get_accepted(limit=10)

                if not recent and not subcmd:
                    print("  No explorations yet.\n")
                elif subcmd == "":
                    for i, exp in enumerate(recent):
                        trigger = exp["trigger_text"][:70]
                        search_label = "searched" if exp["search_used"] else "internal"
                        print(f"  [{i + 1}] ({exp['trigger_type']}, {search_label}) {trigger}")
                    print(f"\n  /explorations N — view details\n")
                else:
                    try:
                        idx = int(subcmd) - 1
                        if 0 <= idx < len(recent):
                            exp = recent[idx]
                            search_label = "searched" if exp["search_used"] else "internal"
                            print(f"\n  Trigger ({exp['trigger_type']}): {exp['trigger_text']}")
                            print(f"  Confidence: {exp.get('internal_confidence', 0):.2f} ({search_label})")
                            if exp.get("queries"):
                                print(f"  Queries: {', '.join(exp['queries'])}")
                            print(f"  Findings:\n  {exp.get('findings', 'None')}")
                            if exp.get("reflection"):
                                print(f"\n  Reflection:\n  {exp['reflection']}")
                            if exp.get("new_questions"):
                                print(f"\n  Questions remaining:")
                                for q in exp["new_questions"]:
                                    print(f"    - {q}")
                            print()
                        else:
                            print(f"  Invalid number. Use /explorations to see list.\n")
                    except ValueError:
                        print(f"  Usage: /explorations [N]\n")

        else:
            print(f"  Unknown command: {cmd}. Type /help for available commands.\n")

        return True

    def web_sleep(self):
        """Web-mode sleep — consolidate without tearing down connections.

        Runs full consolidation + state logging, marks the episode complete,
        but keeps DB connections and inference alive so the server can still
        serve the approval queue and dashboard.
        """
        if self._sleeping:
            return
        self._sleeping = True

        logger.info("Web sleep initiated")

        if self.current_episode_id and self.turn_count > 0:
            logger.info(
                f"Episode {self.current_episode_id} web-sleeping: "
                f"{self.turn_count} turns"
            )
            if self.evolution:
                try:
                    self.evolution.consolidate(
                        self.current_episode_id, interactive=False,
                        queue_mode=True,
                    )
                except Exception as e:
                    logger.warning(f"Web sleep consolidation failed: {e}")
                    try:
                        self.episodic.mark_episode_consolidated(
                            self.current_episode_id
                        )
                    except Exception:
                        pass
        elif self.turn_count == 0 and self.evolution:
            # DMS mode — no conversation turns, but beliefs exist from document
            # ingestion. Run dreams directly to build graph edges.
            logger.info("Web sleep (DMS mode): no turns, running dreams only")
            try:
                from datetime import datetime, timezone
                dream_start = datetime.now(timezone.utc).isoformat()
                self.evolution._run_dreams(queue_mode=True)
                self.evolution._dedup_edges(since_timestamp=dream_start)
            except Exception as e:
                logger.warning(f"DMS dream-only sleep failed: {e}")

        if self.state_monitor:
            self.state_monitor.log_state(
                "sleep", episode_id=self.current_episode_id
            )

        logger.info("Web sleep complete")

        # Reset for next cycle — start fresh episode, clear sleeping flag
        self.current_episode_id = self.episodic.create_episode()
        self.turn_count = 0
        self._sleeping = False

    def sleep(self):
        """Go to sleep — consolidate the day's experience, then rest."""
        if self._sleeping:
            return
        self._sleeping = True

        self._print("\n  Going to sleep...")

        if self.current_episode_id and self.turn_count > 0:
            logger.info(
                f"Episode {self.current_episode_id} sleeping: {self.turn_count} turns"
            )
            # Sleep consolidation — process the day's experience
            if self.evolution:
                try:
                    if self._web_mode:
                        self.evolution.consolidate(
                            self.current_episode_id, interactive=False,
                            queue_mode=True,
                        )
                    else:
                        self.evolution.consolidate(
                            self.current_episode_id, interactive=True,
                        )
                except Exception as e:
                    logger.warning(f"Sleep consolidation failed: {e}")
                    # Mark consolidated anyway so missed sleep recovery doesn't re-process
                    try:
                        self.episodic.mark_episode_consolidated(
                            self.current_episode_id
                        )
                    except Exception:
                        pass

        # Log sleep state (before closing memory systems)
        if self.state_monitor:
            self.state_monitor.log_state("sleep", episode_id=self.current_episode_id)
            self.state_monitor.close()

        if self.episodic:
            self.episodic.close()
        if self.semantic:
            self.semantic.close()
        if self.reflective:
            self.reflective.close()
        if self.curiosity:
            self.curiosity.close()
        if self.explorations:
            self.explorations.close()
        if self.inference:
            self.inference.unload()

        self._print("  Goodnight.\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Persistence CLI")
    parser.add_argument("--web", action="store_true", help="Launch web interface instead of CLI")
    args = parser.parse_args()

    if args.web:
        # Launch web server via uvicorn
        import uvicorn
        from interface.web_server import app
        web_cfg = load_config().get("web", {})
        uvicorn.run(app, host=web_cfg.get("host", "0.0.0.0"), port=web_cfg.get("port", 8899))
        return

    signal.signal(signal.SIGINT, lambda s, f: None)  # let KeyboardInterrupt propagate naturally
    cli = PersistenceCLI()

    # Universal sleep triggers — ensure consolidation happens on any exit
    def _sleep_on_signal(signum, frame):
        logger.info(f"Received signal {signum}, initiating sleep")
        cli.sleep()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _sleep_on_signal)
    atexit.register(cli.sleep)

    cli.run()


if __name__ == "__main__":
    main()
