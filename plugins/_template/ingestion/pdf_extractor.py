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

"""Research Product — PDF Section Extractor.

Two-pass approach (Chronicle recommendation — every regex approach in this project has failed):
1. Regex on common section headers (Introduction, Methods, Results, Discussion, etc.)
2. LLM fallback for non-standard papers

Rule: Missing a section = acceptable. Hallucinating a section = not.

Extracts: abstract + methods + conclusions + discussion.
Full text available as user-flagged fallback for unconventional papers.
"""

import logging
import os
import re

logger = logging.getLogger("ingestion.pdf_extractor")

# Common section header patterns (case-insensitive)
SECTION_PATTERNS = [
    (r"(?i)^#+\s*abstract\b|^abstract\s*$|^ABSTRACT\b", "abstract"),
    (r"(?i)^#+\s*introduction\b|^introduction\s*$|^INTRODUCTION\b|^1[\.\s]+introduction", "introduction"),
    (r"(?i)^#+\s*(?:materials?\s+and\s+)?methods?\b|^(?:materials?\s+and\s+)?methods?\s*$|^METHODS?\b|^2[\.\s]+methods", "methods"),
    (r"(?i)^#+\s*results?\b|^results?\s*$|^RESULTS?\b|^3[\.\s]+results|^results?\s+and\s+discussion", "results"),
    (r"(?i)^#+\s*discussion\b|^discussion\s*$|^DISCUSSION\b|^4[\.\s]+discussion", "discussion"),
    (r"(?i)^#+\s*conclusions?\b|^conclusions?\s*$|^CONCLUSIONS?\b|^5[\.\s]+conclusion|^concluding\s+remarks", "conclusion"),
    (r"(?i)^#+\s*(?:findings?|analysis)\b|^(?:findings?|analysis)\s*$", "results"),
    (r"(?i)^#+\s*(?:summary)\b|^summary\s*$", "conclusion"),
    (r"(?i)^#+\s*references?\b|^references?\s*$|^REFERENCES?\b|^bibliography", "references"),
    (r"(?i)^#+\s*(?:supplementary|appendix|acknowledgements?)\b", "supplementary"),
]

# Sections we want to extract (skip references and supplementary)
TARGET_SECTIONS = {"abstract", "introduction", "methods", "results", "discussion", "conclusion"}


class PDFExtractor:
    """Extract structured sections from PDF files for belief extraction."""

    def __init__(self, inference_engine=None, config=None):
        self.inference = inference_engine
        self.config = config or {}
        research_cfg = config.get("research", {}) if config else {}
        self.section_detection = research_cfg.get("pdf_section_detection", "two_pass")

    def extract_text(self, pdf_path):
        """Extract raw text from a PDF file.

        Tries pypdf first, falls back to pdfplumber.
        Returns full text string or None on failure.
        """
        text = self._try_pypdf(pdf_path)
        if text and len(text.strip()) > 100:
            return text

        text = self._try_pdfplumber(pdf_path)
        if text and len(text.strip()) > 100:
            return text

        logger.warning(f"Could not extract text from {pdf_path}")
        return None

    def extract_sections(self, pdf_path, full_text=False):
        """Extract structured sections from a PDF.

        Args:
            pdf_path: Path to the PDF file
            full_text: If True, return all text (user-flagged override)

        Returns:
            dict: {section_name: text_content} for target sections
            Falls back to full text if section detection fails
        """
        text = self.extract_text(pdf_path)
        if not text:
            return None

        if full_text:
            return {"full_text": text}

        # Pass 1: Regex section detection
        sections = self._regex_section_detect(text)

        # Check if regex found enough
        found = set(sections.keys()) & TARGET_SECTIONS
        if len(found) >= 2:
            logger.info(
                f"PDF regex found {len(found)} sections in {os.path.basename(pdf_path)}: "
                f"{', '.join(found)}"
            )
            return {k: v for k, v in sections.items() if k in TARGET_SECTIONS}

        # Pass 2: LLM fallback
        if self.inference and self.section_detection == "two_pass":
            logger.info(
                f"Regex found only {len(found)} sections, trying LLM fallback "
                f"for {os.path.basename(pdf_path)}"
            )
            llm_sections = self._llm_section_detect(text)
            if llm_sections:
                found_llm = set(llm_sections.keys()) & TARGET_SECTIONS
                if len(found_llm) > len(found):
                    return {k: v for k, v in llm_sections.items() if k in TARGET_SECTIONS}

        # Fallback: if we have any sections, use them; otherwise return chunked text
        if sections:
            return {k: v for k, v in sections.items() if k in TARGET_SECTIONS}

        # Last resort: split into rough chunks for extraction
        logger.info(f"No sections detected, chunking {os.path.basename(pdf_path)}")
        return self._chunk_text(text)

    def extract_for_ingestion(self, pdf_path, max_chars=None):
        """Extract text suitable for the belief extraction pipeline.

        Returns a list of (section_name, text) tuples, each sized for batch extraction.
        """
        if max_chars is None:
            max_chars = self.config.get("extraction", {}).get("max_file_size_bytes", 100000)

        sections = self.extract_sections(pdf_path)
        if not sections:
            return []

        result = []
        for section_name, text in sections.items():
            if len(text) > max_chars:
                # Split large sections into chunks
                chunks = self._split_section(text, max_chars)
                for i, chunk in enumerate(chunks):
                    result.append((f"{section_name}_part{i+1}", chunk))
            else:
                result.append((section_name, text))

        return result

    def _regex_section_detect(self, text):
        """Detect sections using regex patterns on common headers."""
        lines = text.split("\n")
        sections = {}
        current_section = None
        current_lines = []

        for line in lines:
            matched = False
            for pattern, section_name in SECTION_PATTERNS:
                if re.match(pattern, line.strip()):
                    # Save previous section
                    if current_section and current_lines:
                        content = "\n".join(current_lines).strip()
                        if content:
                            sections[current_section] = content
                    current_section = section_name
                    current_lines = []
                    matched = True
                    break

            if not matched and current_section:
                current_lines.append(line)

        # Save last section
        if current_section and current_lines:
            content = "\n".join(current_lines).strip()
            if content:
                sections[current_section] = content

        return sections

    def _llm_section_detect(self, text, max_chars=8000):
        """Use LLM to identify sections in non-standard papers."""
        # Truncate to avoid token overflow
        truncated = text[:max_chars]

        prompt = """Identify the sections in this research paper text. For each section found, output its name and the line numbers where it starts and ends.

Use ONLY these section names: abstract, introduction, methods, results, discussion, conclusion

Respond in JSON format:
[{"section": "abstract", "start_line": 1, "end_line": 15}, ...]

If a section doesn't exist, omit it. Do NOT hallucinate sections that aren't there.

Paper text (first portion):
---
""" + truncated + """
---
/no_think"""

        try:
            import json
            content = self.inference.generate_with_messages(
                messages=[
                    {"role": "system", "content": "You identify sections in research papers. Respond only in JSON."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=512, task="extraction",
            )
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

            # Parse JSON
            section_ranges = None
            try:
                section_ranges = json.loads(content)
            except json.JSONDecodeError:
                match = re.search(r"\[.*\]", content, re.DOTALL)
                if match:
                    try:
                        section_ranges = json.loads(match.group())
                    except json.JSONDecodeError:
                        pass

            if not section_ranges:
                return None

            # Extract text by line ranges
            lines = text.split("\n")
            sections = {}
            for sr in section_ranges:
                name = sr.get("section", "").lower()
                start = sr.get("start_line", 1) - 1
                end = sr.get("end_line", start + 1)
                if name in TARGET_SECTIONS and 0 <= start < len(lines):
                    end = min(end, len(lines))
                    sections[name] = "\n".join(lines[start:end]).strip()

            return sections if sections else None

        except Exception as e:
            logger.warning(f"LLM section detection failed: {e}")
            return None

    def _chunk_text(self, text, chunk_size=4000):
        """Split text into rough chunks when no sections detected."""
        chunks = {}
        paragraphs = text.split("\n\n")
        current = []
        current_len = 0
        chunk_idx = 1

        for para in paragraphs:
            if current_len + len(para) > chunk_size and current:
                chunks[f"chunk_{chunk_idx}"] = "\n\n".join(current)
                chunk_idx += 1
                current = []
                current_len = 0
            current.append(para)
            current_len += len(para)

        if current:
            chunks[f"chunk_{chunk_idx}"] = "\n\n".join(current)

        return chunks

    def _split_section(self, text, max_chars):
        """Split a large section into smaller pieces at paragraph boundaries."""
        paragraphs = text.split("\n\n")
        chunks = []
        current = []
        current_len = 0

        for para in paragraphs:
            if current_len + len(para) > max_chars and current:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            current.append(para)
            current_len += len(para)

        if current:
            chunks.append("\n\n".join(current))

        return chunks

    def _try_pypdf(self, pdf_path):
        """Extract text using pypdf."""
        try:
            from pypdf import PdfReader
            reader = PdfReader(pdf_path)
            pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            return "\n\n".join(pages) if pages else None
        except ImportError:
            logger.debug("pypdf not installed, skipping")
            return None
        except Exception as e:
            logger.debug(f"pypdf failed on {pdf_path}: {e}")
            return None

    def _try_pdfplumber(self, pdf_path):
        """Extract text using pdfplumber."""
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                pages = []
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        pages.append(text)
            return "\n\n".join(pages) if pages else None
        except ImportError:
            logger.debug("pdfplumber not installed, skipping")
            return None
        except Exception as e:
            logger.debug(f"pdfplumber failed on {pdf_path}: {e}")
            return None
