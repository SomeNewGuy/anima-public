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

"""Tool execution framework for Persistence.

Implements ReAct-style tool use: the model outputs structured tool calls,
we parse them, execute the tool, inject results, and let the model respond.
"""

import re
import logging
import time
from ddgs import DDGS

logger = logging.getLogger(__name__)

# Pattern to detect tool calls in model output
TOOL_PATTERN = re.compile(
    r'\[TOOL:\s*(search|calculate|think|explore)\]\s*\[QUERY:\s*"([^"]+)"\]',
    re.IGNORECASE,
)


def detect_tool_call(text):
    """Parse model output for a tool call.

    Returns (tool_name, query) if found, (None, None) otherwise.
    """
    match = TOOL_PATTERN.search(text)
    if match:
        return match.group(1).lower(), match.group(2)
    return None, None


def execute_tool(tool_name, query):
    """Execute a tool and return formatted results."""
    if tool_name == "search":
        return execute_search(query)
    elif tool_name == "calculate":
        return execute_calculate(query)
    elif tool_name == "think":
        # Think is just a reasoning step — no external action
        return f'[THINKING STEP: "{query}" — continue reasoning.]'
    elif tool_name == "explore":
        # Explore is handled by cli.py — return a marker for the caller to intercept
        return f'[EXPLORE REQUEST: "{query}"]'
    else:
        return f"[ERROR: Unknown tool '{tool_name}']"


def execute_search(query, max_results=3):
    """Run a web search via DuckDuckGo and return formatted results."""
    logger.info(f"Executing search: {query}")
    t0 = time.time()

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        elapsed = time.time() - t0
        logger.info(f"Search completed in {elapsed:.1f}s, {len(results)} results")

        if not results:
            return (
                f'[SEARCH RESULTS for: "{query}"]\n'
                f'No results found.\n'
                f'[END RESULTS]\n'
                f'INSTRUCTION: The search returned no results. State that you could not '
                f'verify this and explain what you do or do not know from your own knowledge.'
            )

        lines = [f'[SEARCH RESULTS for: "{query}"]']
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            url = r.get("href", "No URL")
            snippet = r.get("body", "No snippet")
            lines.append(f"Source {i}: [{title}] ({url})")
            lines.append(f"  {snippet}")
            lines.append("")

        lines.append("[END RESULTS]")
        lines.append(
            "INSTRUCTION: Evaluate these results critically. Do they agree with each "
            "other? Do they contradict anything you know? "
            "State what you found in your own words. "
            "Do not blindly trust search results."
        )

        return "\n".join(lines)

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return (
            f'[SEARCH ERROR: Could not complete search for "{query}". '
            f'Error: {str(e)}]\n'
            f'INSTRUCTION: The search failed. Respond based on what you know, '
            f'and clearly state that you were unable to verify.'
        )


def execute_calculate(expression):
    """Evaluate a simple math expression safely."""
    logger.info(f"Calculating: {expression}")

    # Whitelist: only allow numbers, basic operators, parentheses, decimal points
    if not re.match(r'^[\d\s\+\-\*/\(\)\.\,]+$', expression):
        return f'[CALCULATION ERROR: Expression contains invalid characters. Only basic arithmetic is supported.]'

    try:
        # Remove commas (thousand separators)
        clean = expression.replace(",", "")
        result = eval(clean)  # safe because we whitelist characters above
        return f'[CALCULATION RESULT: {expression} = {result}]'
    except Exception as e:
        return f'[CALCULATION ERROR: Could not evaluate "{expression}". Error: {str(e)}]'


# Few-shot examples for the system prompt
TOOL_USE_EXAMPLES = """
TOOLS:
You can search the web, calculate math, or explore a topic. To use a tool, your ENTIRE response must be ONLY the tool call line — nothing else before or after it.

Format: [TOOL: search] [QUERY: "your search terms"]
Format: [TOOL: calculate] [QUERY: "math expression"]
Format: [TOOL: explore] [QUERY: "topic you want to investigate"]
Format: [TOOL: explore] [QUERY: "session"]

WHEN TO USE:
- Uncertain about a specific real-world fact → search.
- Someone claims a specific verifiable fact you can't confirm → search to check.
- Math beyond basic arithmetic → calculate.
- You identified a gap in your knowledge during conversation → explore with a specific topic.
- A topic emerged that you want to investigate further → explore with a specific topic.
- You have a hypothesis you want to test against external information → explore with a specific topic.
- You want to run a full exploration window (multiple topics) → explore with "session".
- Opinion or discussion → just respond, no tool.
- You already know the answer confidently → just respond, no tool.
- Greetings, small talk, emotional messages → just respond, NEVER use a tool.

NEVER SEARCH FOR:
- Yourself — your name, architecture, creator, capabilities, or how you work.
- Abstract concepts — "what is critical thinking", "what is consciousness". Just answer.
- Things the user just told you — engage with what they said, don't look it up.
- Philosophical or subjective questions — reason about them, don't search.
- Your own internal systems — described in your prompt, you already know them.
- Greetings or casual conversation — "hi", "how are you", "good morning" are not research queries.

IMPORTANT: When using a tool, output ONLY the tool line. Do not add explanation before or after it. Most messages do NOT need a tool — just respond naturally.
""".strip()
