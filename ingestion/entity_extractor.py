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

"""Document entity extractor — structured metadata extraction via LLM.

Extracts people, organizations, codes/references, dates, topics, identifiers,
and document type classification from documents. Entities are document metadata,
not beliefs — they don't enter the graph or go through triage.

Supports single and batch extraction, same batching model as belief extractor.
"""

import json
import logging
import re

logger = logging.getLogger("ingestion.entity_extractor")

ENTITY_EXTRACTION_PROMPT = """Analyze this document and extract structured metadata.

Return a JSON object with these fields (use empty arrays if none found):

{{
  "people": ["full names of people mentioned"],
  "organizations": ["companies, teams, projects mentioned"],
  "codes_references": ["version numbers, codes, item references, standards, legal codes"],
  "dates": ["dates mentioned, normalized to YYYY-MM-DD where possible"],
  "topics": ["key subject areas, 2-5 words each"],
  "identifiers": ["commit hashes, SHA values, ticket IDs, file paths"],
  "document_type": "one of: {document_types}"
}}

Rules:
- People: use full names when available, otherwise the name as given
- Topics: broad enough to match across documents, not document-specific phrases
- Codes/references: include version numbers, legal codes, internal reference IDs
- Document type: pick the single best match from the list
- Be precise — only extract what's explicitly in the document

Document filename: {filename}
Document type: {file_type}

Document content:
{content}"""

BATCH_ENTITY_PROMPT = """Analyze the following {doc_count} documents and extract structured metadata from each.

For EACH document, return a JSON object tagged with its DOCID:

ENTITIES DOCID:<docid>
{{
  "people": ["full names"],
  "organizations": ["companies, teams, projects"],
  "codes_references": ["codes, versions, standards"],
  "dates": ["YYYY-MM-DD where possible"],
  "topics": ["key subject areas"],
  "identifiers": ["hashes, IDs, paths"],
  "document_type": "one of: {document_types}"
}}

Rules:
- People: use full names when available
- Topics: broad enough to match across documents
- Document type: pick the single best match from the list
- Be precise — only extract what's explicitly in the document

{documents}"""

BATCH_DOC_TEMPLATE = """--- DOCID: {docid} | Filename: {filename} | Type: {file_type} ---
{content}
"""

DEFAULT_DOCUMENT_TYPES = [
    "decision", "event", "telemetry", "analysis",
    "narrative", "code", "config", "research", "reference",
]

ENTITY_FIELDS = [
    "people", "organizations", "codes_references",
    "dates", "topics", "identifiers",
]

# Map JSON field names to entity_type values in the DB
FIELD_TO_TYPE = {
    "people": "person",
    "organizations": "org",
    "codes_references": "code",
    "dates": "date",
    "topics": "topic",
    "identifiers": "identifier",
}


def _no_think(text):
    """Append /no_think to suppress thinking mode for structured output."""
    return text + " /no_think"


class EntityExtractor:
    """Extracts structured entity metadata from documents via LLM."""

    def __init__(self, inference_engine, config):
        self.inference = inference_engine
        self.config = config
        extraction_cfg = config.get("extraction", {})
        self.extraction_max_tokens = extraction_cfg.get(
            "extraction_max_tokens",
            config.get("reflection", {}).get("analysis_max_tokens", 1024),
        )
        self.document_types = extraction_cfg.get(
            "document_type_categories", DEFAULT_DOCUMENT_TYPES
        )
        ctx_window = config.get("model", {}).get("context_window", 16384)
        self.context_budget_chars = int(ctx_window * 3 * 0.70)

    def extract(self, content, filename, file_type, sha256):
        """Extract entities from a single document.

        Returns list of (entity_type, entity_value) tuples.
        """
        if not content or not content.strip():
            return []

        if len(content) > self.context_budget_chars:
            content = content[:self.context_budget_chars]

        prompt_text = ENTITY_EXTRACTION_PROMPT.format(
            content=content,
            filename=filename,
            file_type=file_type,
            document_types=" | ".join(self.document_types),
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a document analyst. Extract structured metadata "
                    "from documents. Return valid JSON only."
                ),
            },
            {"role": "user", "content": _no_think(prompt_text)},
        ]

        try:
            response = self.inference.generate_with_messages(
                messages,
                max_tokens=self.extraction_max_tokens,
                temperature=0.1, task="entities",
                timeout=180,
            )
            entities = self._parse_single_response(response, sha256)
            logger.info(f"Extracted {len(entities)} entities from {filename}")
            return entities

        except Exception as e:
            logger.warning(f"Entity extraction failed for {filename}: {e}")
            return []

    def extract_batch(self, documents):
        """Extract entities from multiple documents in a single LLM call.

        Args:
            documents: List of dicts with: content, filename, file_type, sha256, docid

        Returns:
            Dict mapping sha256 → list of (entity_type, entity_value) tuples.
        """
        if not documents:
            return {}

        doc_sections = []
        docid_map = {}
        for doc in documents:
            docid = doc["docid"]
            docid_map[docid] = doc["sha256"]
            doc_sections.append(BATCH_DOC_TEMPLATE.format(
                docid=docid,
                filename=doc["filename"],
                file_type=doc["file_type"],
                content=doc["content"],
            ))

        prompt_text = BATCH_ENTITY_PROMPT.format(
            doc_count=len(documents),
            document_types=" | ".join(self.document_types),
            documents="\n".join(doc_sections),
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a document analyst. Extract structured metadata "
                    "from documents. Return valid JSON for each document."
                ),
            },
            {"role": "user", "content": _no_think(prompt_text)},
        ]

        batch_max_tokens = self.extraction_max_tokens * len(documents)

        try:
            response = self.inference.generate_with_messages(
                messages,
                max_tokens=batch_max_tokens,
                temperature=0.1, task="entities",
                timeout=180 + (60 * len(documents)),
            )
            return self._parse_batch_response(response, docid_map)

        except Exception as e:
            logger.warning(f"Batch entity extraction failed ({len(documents)} docs): {e}")
            return {}

    def _parse_single_response(self, response, sha256):
        """Parse JSON entity response for a single document.

        Returns list of (entity_type, entity_value) tuples.
        """
        data = self._extract_json(response)
        if not data:
            logger.warning(f"Entity parse failure for {sha256[:12]}: no JSON found")
            return []

        return self._json_to_entities(data)

    def _parse_batch_response(self, response, docid_map):
        """Parse batch entity response with DOCID tags.

        Returns dict mapping sha256 → list of (entity_type, entity_value) tuples.
        """
        results = {}

        # Split on ENTITIES DOCID: markers
        sections = re.split(r'ENTITIES\s+DOCID:\s*(\S+)', response)

        # sections[0] is preamble (before first marker), then alternating docid, content
        for i in range(1, len(sections) - 1, 2):
            docid = sections[i].strip()
            content = sections[i + 1]
            sha = docid_map.get(docid)
            if not sha:
                logger.warning(f"Unknown DOCID in entity batch: {docid}")
                continue

            data = self._extract_json(content)
            if data:
                results[sha] = self._json_to_entities(data)
            else:
                logger.warning(f"Entity parse failure for DOCID {docid}")
                results[sha] = []

        return results

    def _extract_json(self, text):
        """Extract first JSON object from text, tolerant of surrounding prose."""
        # Try direct parse first
        text = text.strip()
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            pass

        # Find JSON block in markdown fence
        m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except (json.JSONDecodeError, ValueError):
                pass

        # Find first { ... } block
        m = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except (json.JSONDecodeError, ValueError):
                pass

        return None

    def _json_to_entities(self, data):
        """Convert parsed JSON dict to list of (entity_type, entity_value) tuples."""
        entities = []

        for field, entity_type in FIELD_TO_TYPE.items():
            values = data.get(field, [])
            if isinstance(values, list):
                for v in values:
                    v = str(v).strip()
                    if v:
                        entities.append((entity_type, v))

        # Document type is special — single value, not a list
        doc_type = data.get("document_type", "")
        if isinstance(doc_type, str) and doc_type.strip():
            entities.append(("doc_type", doc_type.strip().lower()))

        return entities
