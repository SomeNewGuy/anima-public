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

"""Sentence-transformers wrapper for embedding generation."""

import logging
import os

from sentence_transformers import SentenceTransformer
import toml


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "settings.toml")
    return toml.load(os.path.normpath(config_path))


class EmbeddingEngine:
    def __init__(self, config=None):
        self.config = config or load_config()
        self.model = None

    def load(self):
        model_name = self.config["model"]["embedding_model"]
        # Suppress noisy warnings from transformers loading report and HF hub
        logging.getLogger("transformers.utils.loading_report").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        # Load from local cache only — never contact HuggingFace
        self.model = SentenceTransformer(model_name, local_files_only=True)
        return self

    def embed(self, text):
        if self.model is None:
            raise RuntimeError("Embedding model not loaded. Call load() first.")
        return self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)

    def embed_batch(self, texts):
        if self.model is None:
            raise RuntimeError("Embedding model not loaded. Call load() first.")
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def unload(self):
        self.model = None
