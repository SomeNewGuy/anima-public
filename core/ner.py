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

"""Named entity extraction using spaCy.

Dedicated lightweight NER — don't trust the base model for this.
"""

import spacy


class EntityExtractor:
    def __init__(self):
        self.nlp = None

    def load(self):
        self.nlp = spacy.load("en_core_web_sm")
        return self

    def extract(self, text):
        """Extract named entities from text.

        Returns list of dicts with text, label, and start/end positions.
        """
        if self.nlp is None:
            raise RuntimeError("spaCy model not loaded. Call load() first.")

        doc = self.nlp(text)
        entities = []
        seen = set()
        for ent in doc.ents:
            key = (ent.text.lower(), ent.label_)
            if key not in seen:
                seen.add(key)
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                })
        return entities

    def extract_names(self, text):
        """Extract just entity text strings, deduplicated and lowercased."""
        return list({e["text"].lower() for e in self.extract(text)})
