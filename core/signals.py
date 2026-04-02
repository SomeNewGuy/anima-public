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

"""Shared confusion signal detection.

Two categories of signals detected in assistant responses:
- HEDGING: soft uncertainty ("I think", "I'm not sure", "it's possible")
- DEFERRAL: hard knowledge gaps ("I don't know", "I couldn't find", "I was unable to")

Used by:
- evolution.py quality gate
- cli.py real-time gap detection
"""

# Hard gaps — the model explicitly doesn't know
DEFERRAL_SIGNALS = [
    "i don't know",
    "i couldn't find",
    "i was unable to",
    "i cannot find",
    "i'm unable to",
    "no results found",
    "search error",
]

# Soft uncertainty — the model is hedging
HEDGING_SIGNALS = [
    "i'm not sure",
    "i think",
    "it's possible",
    "i believe",
    "i'm ready to respond",
    "please provide",
    "could you clarify",
    "i apologize for",
]

ALL_SIGNALS = DEFERRAL_SIGNALS + HEDGING_SIGNALS


def detect_confusion(text):
    """Check text for confusion signals.

    Returns:
        (is_confused, signal_type, matched_signal)
        signal_type is "deferral" or "hedging", None if not confused.
    """
    lower = text.lower()

    for signal in DEFERRAL_SIGNALS:
        if signal in lower:
            return (True, "deferral", signal)

    for signal in HEDGING_SIGNALS:
        if signal in lower:
            return (True, "hedging", signal)

    return (False, None, None)
