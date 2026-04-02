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

"""Document extractor — per-document belief extraction via LLM.

Different from conversation extraction: extracts factual claims, design decisions,
and domain knowledge from documents rather than summarizing conversation learnings.

Supports batching: multiple small documents can be packed into a single LLM call
with per-document provenance preserved via DOCID tags in the output format.

Produces belief dicts in the same format as _parse_beliefs_response() so they
flow through the existing governance gate identically.
"""

import logging
import os
import re

logger = logging.getLogger("ingestion.extractor")

# Single-document prompt (for large docs that fill context solo)
DOCUMENT_EXTRACTION_PROMPT = """Read this document and extract the important facts, claims, decisions, and knowledge it contains.

Look for:
- Factual claims and assertions
- Design decisions and their rationale
- Technical specifications or configurations
- Principles, rules, or guidelines stated
- Relationships between concepts

For each one, write a line in this exact format:
BELIEF: <statement> | CONFIDENCE: <low/medium/high> | SOURCE: document | EVIDENCE: <what in the document supports this>

Extract up to {max_beliefs} beliefs. Focus on substantive claims, not formatting or metadata.
If the document contains no extractable knowledge, respond with: NONE

Document filename: {filename}
Document type: {file_type}

Document content:
{content}"""

# Batch prompt — multiple documents, provenance-tagged output
BATCH_EXTRACTION_PROMPT = """Read the following {doc_count} documents and extract the important facts, claims, decisions, and knowledge each contains.

Look for:
- Factual claims and assertions
- Design decisions and their rationale
- Technical specifications or configurations
- Principles, rules, or guidelines stated
- Relationships between concepts

CRITICAL: Tag each belief with the DOCID of the document it came from.
Use this exact format for each belief:
BELIEF: <statement> | CONFIDENCE: <low/medium/high> | DOCID: <docid> | EVIDENCE: <what in the document supports this>

Extract up to {max_beliefs} beliefs total across all documents.
If a document contains no extractable knowledge, skip it.
If no documents contain extractable knowledge, respond with: NONE

{documents}"""

# Template for each document within a batch
BATCH_DOC_TEMPLATE = """--- DOCID: {docid} | Filename: {filename} | Type: {file_type} ---
{content}
"""


# System prompt — default for all modes. Plugins can override via get_prompts().
DEFAULT_SYSTEM_PROMPT = "You are a document analyst. Extract the key facts, claims, and knowledge from documents. Be precise and factual."


def _no_think(text):
    """Append /no_think to suppress Qwen3 thinking mode for structured output."""
    return text + " /no_think"


def _strip_prefix(line):
    """Strip numbered list prefixes (1. 2. etc) and bullet markers."""
    return re.sub(r"^[\d]+[\.\)]\s*", "", line).lstrip("- •*")


class DocumentExtractor:
    """Extracts beliefs from documents using the LLM. Supports single and batch extraction."""

    def __init__(self, inference_engine, config):
        self.inference = inference_engine
        self.config = config
        self._config_ref = config  # for PDF extractor access
        extraction_cfg = config.get("extraction", {})
        self.max_beliefs = extraction_cfg.get("max_beliefs_per_document", 10)
        self.extraction_max_tokens = extraction_cfg.get(
            "extraction_max_tokens",
            config.get("reflection", {}).get("analysis_max_tokens", 1024),
        )
        # Context budget for batching — reserve space for prompt template + response
        ctx_window = config.get("model", {}).get("context_window", 16384)
        # ~3 chars per token estimate, reserve 30% for prompt overhead + response
        self.context_budget_chars = int(ctx_window * 3 * 0.70)

    def extract(self, content, filename, file_type, sha256):
        """Extract beliefs from a single document.

        Returns list of belief dicts tagged with document provenance.
        """
        if not content or not content.strip():
            logger.info(f"Empty document: {filename}")
            return []

        # Truncate if needed
        if len(content) > self.context_budget_chars:
            content = content[:self.context_budget_chars]
            logger.info(f"Truncated {filename} to {self.context_budget_chars} chars")

        # Select prompt — plugin-provided first, then core default
        plugin_prompts = getattr(self, '_plugin_prompts', None)
        if plugin_prompts and plugin_prompts.get("extraction_prompt"):
            prompt_template = plugin_prompts["extraction_prompt"]
        else:
            prompt_template = DOCUMENT_EXTRACTION_PROMPT

        prompt_text = prompt_template.format(
            content=content,
            filename=filename,
            file_type=file_type,
            max_beliefs=self.max_beliefs,
        )

        if plugin_prompts and plugin_prompts.get("system_prompt"):
            system_prompt = plugin_prompts["system_prompt"]
        else:
            system_prompt = DEFAULT_SYSTEM_PROMPT

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": _no_think(prompt_text),
            },
        ]

        try:
            response = self.inference.generate_with_messages(
                messages,
                max_tokens=self.extraction_max_tokens,
                temperature=0.1, task="extraction",
                timeout=240,
            )
            logger.info(f"Extraction response for {filename}:\n{response}")
            beliefs = self._parse_single_response(response)

            # Tag provenance
            for b in beliefs:
                b["source"] = "document"
                b["extraction_context"] = "document"
                b["_document_sha"] = sha256
                b["_document_filename"] = filename

            # Adversarial verification: ask model to quote supporting text
            # Catches hallucinated beliefs that sound plausible but aren't in the doc
            # Skip under system pressure — verification doubles LLM load per document
            _do_verify = True
            if hasattr(self, 'inference') and hasattr(self.inference, '_system_pressure'):
                if self.inference._system_pressure() > 0.6:
                    _do_verify = False
                    logger.info(f"Skipping verification for {filename} — system under pressure")
            if beliefs and content and _do_verify:
                beliefs = self._verify_extraction(beliefs, content, filename)

            logger.info(f"Extracted {len(beliefs)} beliefs from {filename}")
            return beliefs

        except Exception as e:
            logger.warning(f"Extraction failed for {filename}: {e}")
            return []

    def extract_with_task(self, content, filename, file_type, sha256, task="extraction"):
        """Extract beliefs using a specific task routing (for escalation to large-tier).

        Same as extract() but overrides the task parameter for model selection.
        """
        if not content or not content.strip():
            return []

        if len(content) > self.context_budget_chars:
            content = content[:self.context_budget_chars]

        prompt_text = DOCUMENT_EXTRACTION_PROMPT.format(
            content=content,
            filename=filename,
            file_type=file_type,
            max_beliefs=self.max_beliefs,
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a document analyst. Extract the key facts, claims, "
                    "and knowledge from documents. Be precise and factual."
                ),
            },
            {"role": "user", "content": _no_think(prompt_text)},
        ]

        try:
            response = self.inference.generate_with_messages(
                messages,
                max_tokens=self.extraction_max_tokens,
                temperature=0.1, task=task,
                timeout=240,
            )
            logger.info(f"Escalated extraction ({task}) for {filename}")
            beliefs = self._parse_single_response(response)

            for b in beliefs:
                b["source"] = "document"
                b["extraction_context"] = "document"
                b["_document_sha"] = sha256
                b["_document_filename"] = filename

            _do_verify_esc = True
            if hasattr(self, 'inference') and hasattr(self.inference, '_system_pressure'):
                if self.inference._system_pressure() > 0.6:
                    _do_verify_esc = False
            if beliefs and content and _do_verify_esc:
                beliefs = self._verify_extraction(beliefs, content, filename)

            logger.info(f"Escalated: {len(beliefs)} beliefs from {filename}")
            return beliefs
        except Exception as e:
            logger.warning(f"Escalated extraction failed for {filename}: {e}")
            return []

    def extract_batch(self, documents):
        """Extract beliefs from multiple documents in a single LLM call.

        Args:
            documents: List of dicts with keys: content, filename, file_type, sha256, docid

        Returns:
            List of belief dicts, each tagged with _document_sha and _document_filename
            from the correct source document via DOCID matching.
        """
        if not documents:
            return []

        # Build document sections
        doc_sections = []
        docid_map = {}  # docid -> {sha256, filename}
        for doc in documents:
            docid = doc["docid"]
            docid_map[docid] = {
                "sha256": doc["sha256"],
                "filename": doc["filename"],
            }
            doc_sections.append(BATCH_DOC_TEMPLATE.format(
                docid=docid,
                filename=doc["filename"],
                file_type=doc["file_type"],
                content=doc["content"],
            ))

        max_beliefs_total = self.max_beliefs * len(documents)
        prompt_text = BATCH_EXTRACTION_PROMPT.format(
            doc_count=len(documents),
            max_beliefs=max_beliefs_total,
            documents="\n".join(doc_sections),
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a document analyst. Extract the key facts, claims, "
                    "and knowledge from documents. Be precise and factual. "
                    "Always include the DOCID tag so each belief traces to its source."
                ),
            },
            {
                "role": "user",
                "content": _no_think(prompt_text),
            },
        ]

        # Scale response tokens for batch size
        batch_max_tokens = self.extraction_max_tokens * len(documents)

        try:
            response = self.inference.generate_with_messages(
                messages,
                max_tokens=batch_max_tokens,
                temperature=0.1, task="extraction",
                timeout=240 + (60 * len(documents)),
            )
            logger.info(f"Batch extraction response ({len(documents)} docs):\n{response}")
            beliefs = self._parse_batch_response(response, docid_map)
            logger.info(
                f"Batch extracted {len(beliefs)} beliefs from {len(documents)} documents"
            )
            return beliefs

        except Exception as e:
            logger.warning(f"Batch extraction failed ({len(documents)} docs): {e}")
            return []

    def build_batches(self, doc_list):
        """Group documents into batches that fit within context budget.

        Args:
            doc_list: List of dicts with keys: sha256, filename, file_path, file_type, size_bytes
                      Plus 'content' (loaded text) added by caller.

        Returns:
            List of batches. Each batch is a list of doc dicts with 'docid' added.
            Single-doc batches for oversized files; multi-doc batches for small files.
        """
        batches = []
        current_batch = []
        current_chars = 0
        docid_counter = 0

        for doc in doc_list:
            content = doc.get("content", "")
            doc_chars = len(content)

            if doc_chars == 0:
                continue

            docid_counter += 1
            doc["docid"] = f"D{docid_counter}"

            # Oversized doc — solo batch, truncated by extract()
            if doc_chars > self.context_budget_chars:
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_chars = 0
                batches.append([doc])
                continue

            # Would this doc overflow the current batch?
            if current_chars + doc_chars > self.context_budget_chars:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [doc]
                current_chars = doc_chars
            else:
                current_batch.append(doc)
                current_chars += doc_chars

        if current_batch:
            batches.append(current_batch)

        return batches

    def _parse_single_response(self, response):
        """Parse BELIEF lines from single-document extraction."""
        if not response:
            return []
        if response.strip().upper() == "NONE":
            return []

        beliefs = []
        for line in response.strip().split("\n"):
            line = _strip_prefix(line.strip())
            if not line.upper().startswith("BELIEF:"):
                continue

            remainder = line.split(":", 1)[1]
            segments = remainder.split("|")

            parts = {}
            if segments:
                parts["statement"] = segments[0].strip()

            for seg in segments[1:]:
                seg = seg.strip()
                seg_upper = seg.upper()
                if seg_upper.startswith("CONFIDENCE:"):
                    parts["confidence"] = seg.split(":", 1)[1].strip().lower()
                elif seg_upper.startswith("SOURCE:"):
                    parts["source"] = seg.split(":", 1)[1].strip().lower()
                elif seg_upper.startswith("EVIDENCE:"):
                    parts["evidence"] = seg.split(":", 1)[1].strip()

            if parts.get("statement"):
                # Filter non-beliefs: LLM admitting it has nothing
                stmt_lower = parts["statement"].lower()
                _NON_BELIEF_PATTERNS = [
                    "does not explicitly mention",
                    "no direct connection",
                    "the document does not discuss",
                    "the document does not contain",
                    "no explicit mention",
                    "not mentioned in the document",
                    "there is no information",
                    "cannot be determined from",
                    "no evidence in the document",
                    "the text does not address",
                ]
                if any(p in stmt_lower for p in _NON_BELIEF_PATTERNS):
                    logger.info(f"Non-belief filtered: {parts['statement'][:60]}")
                    continue
                beliefs.append(parts)

        if not beliefs and response.strip().upper() != "NONE":
            logger.warning(
                f"Single extraction parse failure — 0 beliefs. "
                f"First 200 chars: {response[:200]}"
            )

        return beliefs[:self.max_beliefs]

    def _parse_batch_response(self, response, docid_map):
        """Parse BELIEF lines from batch extraction with DOCID provenance.

        Each line must include DOCID: <id> to trace back to source document.
        Beliefs without a valid DOCID are tagged as 'unknown' source.
        """
        if response.strip().upper() == "NONE":
            return []

        beliefs = []
        for line in response.strip().split("\n"):
            line = _strip_prefix(line.strip())
            if not line.upper().startswith("BELIEF:"):
                continue

            remainder = line.split(":", 1)[1]
            segments = remainder.split("|")

            parts = {}
            docid = None
            if segments:
                parts["statement"] = segments[0].strip()

            for seg in segments[1:]:
                seg = seg.strip()
                seg_upper = seg.upper()
                if seg_upper.startswith("CONFIDENCE:"):
                    parts["confidence"] = seg.split(":", 1)[1].strip().lower()
                elif seg_upper.startswith("DOCID:"):
                    docid = seg.split(":", 1)[1].strip()
                elif seg_upper.startswith("SOURCE:"):
                    parts["source"] = seg.split(":", 1)[1].strip().lower()
                elif seg_upper.startswith("EVIDENCE:"):
                    parts["evidence"] = seg.split(":", 1)[1].strip()

            if not parts.get("statement"):
                continue

            # Filter non-beliefs: LLM admitting it has nothing
            stmt_lower = parts["statement"].lower()
            _NON_BELIEF_PATTERNS = [
                "does not explicitly mention",
                "no direct connection",
                "the document does not discuss",
                "the document does not contain",
                "no explicit mention",
                "not mentioned in the document",
                "there is no information",
                "cannot be determined from",
                "no evidence in the document",
                "the text does not address",
            ]
            if any(p in stmt_lower for p in _NON_BELIEF_PATTERNS):
                logger.info(f"Non-belief filtered: {parts['statement'][:60]}")
                continue

            # Resolve DOCID to provenance
            parts["source"] = "document"
            parts["extraction_context"] = "document"

            if docid and docid in docid_map:
                parts["_document_sha"] = docid_map[docid]["sha256"]
                parts["_document_filename"] = docid_map[docid]["filename"]
            else:
                # DOCID missing or unrecognized — log but still keep the belief
                logger.warning(
                    f"Batch belief missing/invalid DOCID '{docid}': "
                    f"{parts['statement'][:60]}"
                )
                parts["_document_sha"] = None
                parts["_document_filename"] = "unknown"

            beliefs.append(parts)

        if not beliefs and response.strip().upper() != "NONE":
            logger.warning(
                f"Batch extraction parse failure — 0 beliefs. "
                f"First 200 chars: {response[:200]}"
            )

        return beliefs

    def extract_targeted(self, section, filename, sha256, gap_question,
                         gap_context="", existing_beliefs=None):
        """Targeted extraction — re-read a document section with a curiosity gap as framing.

        Fundamentally different from blind extraction: the system knows what it's
        looking for, has existing context, and extracts beliefs that address the gap.

        Args:
            section: Document section text (from corpus matcher).
            filename: Source document filename.
            sha256: Document SHA for provenance.
            gap_question: The curiosity gap question being investigated.
            gap_context: Additional context about why the gap exists.
            existing_beliefs: List of related belief statements (for dedup).

        Returns list of belief dicts tagged with provenance.
        """
        if not section or not section.strip():
            return []

        existing_text = ""
        if existing_beliefs:
            belief_lines = [f"- {b}" for b in existing_beliefs[:5]]
            existing_text = (
                "\n\nEXISTING KNOWLEDGE AROUND THIS GAP:\n"
                + "\n".join(belief_lines)
                + "\n\nDo not repeat what is already known."
            )

        prompt_text = (
            f"You are re-reading a document section to answer a specific question "
            f"that emerged from the knowledge graph.\n\n"
            f"QUESTION: {gap_question}\n"
        )
        if gap_context:
            prompt_text += f"QUESTION CONTEXT: {gap_context}\n"
        prompt_text += existing_text
        prompt_text += (
            f"\n\nDOCUMENT SECTION (from {filename}):\n{section}\n\n"
            f"Extract beliefs that ADDRESS this question. Focus on what this section "
            f"reveals about the question, not general facts.\n"
            f"Format: BELIEF: <statement> | CONFIDENCE: <low/medium/high> | "
            f"SOURCE: document | EVIDENCE: <what in the document supports this>\n"
            f"If the section doesn't address the question, respond with: NONE"
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a document analyst performing targeted re-reading. "
                    "Extract beliefs that specifically address the given question. "
                    "Be precise — only extract what the document section actually says."
                ),
            },
            {"role": "user", "content": _no_think(prompt_text)},
        ]

        try:
            response = self.inference.generate_with_messages(
                messages,
                max_tokens=self.extraction_max_tokens,
                temperature=0.1, task="extraction",
                timeout=180,
            )
            beliefs = self._parse_single_response(response)

            # Tag provenance — targeted extraction
            for b in beliefs:
                b["source"] = "document"
                b["extraction_context"] = "targeted"
                b["_document_sha"] = sha256
                b["_document_filename"] = filename
                b["_source_gap"] = gap_question

            logger.info(
                f"Targeted extraction from {filename}: "
                f"{len(beliefs)} beliefs for gap '{gap_question[:60]}'"
            )
            return beliefs

        except Exception as e:
            logger.warning(f"Targeted extraction failed for {filename}: {e}")
            return []

    def _verify_extraction(self, beliefs, source_content, filename):
        """Adversarial verification pass on extracted beliefs.

        For each belief, asks: "Quote the exact text from the document that
        supports this claim." If the model can't quote supporting text,
        the belief is likely hallucinated.

        Returns filtered list of verified beliefs.
        """
        if not beliefs or not self.inference:
            return beliefs

        # Build verification prompt — batch all beliefs in one call
        belief_list = "\n".join(
            f"{i+1}. {b['statement'][:150]}"
            for i, b in enumerate(beliefs[:10])  # cap at 10 to fit context
        )

        # Use a truncated version of source to fit context
        max_source = min(len(source_content), self.context_budget_chars // 2)
        truncated_source = source_content[:max_source]

        prompt = (
            f"For each claim below, quote the EXACT phrase or sentence from the "
            f"document that supports it. If no supporting text exists in the "
            f"document, write UNSUPPORTED.\n\n"
            f"CLAIMS:\n{belief_list}\n\n"
            f"DOCUMENT:\n{truncated_source}\n\n"
            f"For each numbered claim, respond with:\n"
            f"<number>. SUPPORTED: \"<exact quote>\" OR UNSUPPORTED\n/no_think"
        )

        try:
            response = self.inference.generate_with_messages(
                messages=[
                    {"role": "system", "content": "You verify claims against source text. Be strict — only mark SUPPORTED if the document actually contains the claim."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024, task="extraction",
                temperature=0.0,
            )

            import re
            response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)

            # Parse verification results
            unsupported = set()
            for line in response.strip().split("\n"):
                line = line.strip()
                match = re.match(r"(\d+)\.\s*UNSUPPORTED", line)
                if match:
                    unsupported.add(int(match.group(1)))

            if unsupported:
                verified = []
                for i, b in enumerate(beliefs):
                    if (i + 1) in unsupported:
                        logger.info(
                            f"Verification rejected ({filename}): {b['statement'][:60]}"
                        )
                    else:
                        verified.append(b)
                logger.info(
                    f"Verification: {len(verified)}/{len(beliefs)} beliefs survived "
                    f"({len(unsupported)} unsupported) from {filename}"
                )
                return verified

        except Exception as e:
            logger.warning(f"Verification failed for {filename}: {e}")

        return beliefs  # if verification fails, keep all (don't block pipeline)

    def read_document(self, filepath):
        """Read a document file and return its text content, or None on failure.

        Handles text files directly; PDFs via pdf_extractor (structured sections).
        """
        if filepath.lower().endswith(".pdf"):
            return self._read_pdf(filepath)

        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Failed to read {filepath}: {e}")
            return None

    def _read_pdf(self, filepath):
        """Extract text from a PDF using the research PDF extractor."""
        try:
            from .pdf_extractor import PDFExtractor
            pdf = PDFExtractor(
                inference_engine=self.inference,
                config=getattr(self, "_config_ref", {}),
            )
            sections = pdf.extract_sections(filepath)
            if not sections:
                return None
            # Combine sections into a single text with headers
            parts = []
            for name, text in sections.items():
                parts.append(f"## {name.replace('_', ' ').title()}\n\n{text}")
            return "\n\n".join(parts)
        except Exception as e:
            logger.warning(f"PDF extraction failed for {filepath}: {e}")
            return None
