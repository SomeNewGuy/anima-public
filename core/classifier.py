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

"""Rule-based query classifier for retrieval weight profile selection.

Classifies incoming queries into: factual, reasoning, emotional, meta.
Drives both retrieval weight profiles and response budget estimation.

Rule-based for Phase 0. Log every classification for future training data.
"""

import re
import logging

logger = logging.getLogger(__name__)

# Patterns for each category — order matters, first match wins
PATTERNS = {
    "meta": [
        r"\b(your|you)\b.*\b(belie|think|remember|memor|correct|bias|pattern)\b",
        r"\bhave (you|we)\b.*\b(discuss|talk|mention)\b",
        r"\bwhat do you (know|think|believe|remember)\b",
        r"\bhow (confident|certain|sure)\b",
        r"\bprevious(ly)?\b.*\b(said|thought|believed)\b",
    ],
    "emotional": [
        r"\b(feel|feeling|afraid|worried|concern|hope|trust|angry|frustrat)\b",
        r"\b(matter|care|important) to (me|us)\b",
        r"\bi('m| am)\b.*\b(struggling|confused|overwhelmed|excited)\b",
    ],
    "reasoning": [
        r"^(why|how)\b",
        r"\b(consistent|contradict|logic|argument|reason|because|therefore)\b",
        r"\b(compar|contrast|differ|similar|relat)\b.*\b(between|to|with)\b",
        r"\b(should|would|could)\b.*\b(if|when|because)\b",
        r"\b(implicat|consequen|tradeoff|trade-off)\b",
    ],
    "factual": [
        r"^(what|who|when|where|which|how many|how much)\b",
        r"\b(did we|what did|tell me about)\b",
        r"\b(define|explain|describe|list)\b",
    ],
}

# Compiled patterns
_COMPILED = {
    category: [re.compile(p, re.IGNORECASE) for p in patterns]
    for category, patterns in PATTERNS.items()
}


def classify_query(text):
    """Classify a query into a retrieval profile category.

    Returns one of: 'factual', 'reasoning', 'emotional', 'meta'
    Defaults to 'factual' if ambiguous (safest retrieval profile).
    """
    text = text.strip()

    for category, patterns in _COMPILED.items():
        for pattern in patterns:
            if pattern.search(text):
                logger.info(f"Query classified as '{category}': {text[:80]}")
                return category

    # Default to factual — over-retrieves on entity/semantic which is safe
    logger.info(f"Query classified as 'factual' (default): {text[:80]}")
    return "factual"


def get_weight_profile(category, config):
    """Get retrieval weights for a query category from config."""
    weights = config.get("retrieval", {}).get("weights", {}).get(category)
    if weights is None:
        # Fallback to factual
        weights = config["retrieval"]["weights"]["factual"]
    return weights


def estimate_response_budget(category, context_window):
    """Estimate what fraction of context window to reserve for response.

    Reasoning queries need more response room. Factual queries need less.
    Returns fraction (0.0 to 1.0) to reserve for response generation.
    """
    budgets = {
        "factual": 0.35,
        "reasoning": 0.50,
        "emotional": 0.40,
        "meta": 0.45,
    }
    return max(budgets.get(category, 0.35), 0.30)  # enforce 30% floor
