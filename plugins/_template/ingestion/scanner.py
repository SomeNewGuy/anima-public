"""Filesystem scanner — walks datafiles directory, hashes files, manages document ledger.

Populates the document_ledger table in persistence.db with SHA256 hashes,
file metadata, and status lifecycle (discovered → indexed → skipped).
"""

import hashlib
import logging
import os
import sqlite3
from datetime import datetime, timezone
from fnmatch import fnmatch

logger = logging.getLogger("ingestion.scanner")

DEFAULT_INCLUDE_TYPES = [
    ".md", ".py", ".toml", ".txt", ".json", ".yaml", ".yml",
    ".sh", ".bat", ".csv",
]
DEFAULT_EXCLUDE_TYPES = [
    ".png", ".jpg", ".gif", ".pdf", ".db", ".sqlite",
    ".bin", ".exe", ".zip",
]
DEFAULT_EXCLUDE_PATTERNS = [".git*"]
DEFAULT_MAX_FILE_SIZE = 100_000  # bytes


def _sha256_file(filepath):
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_binary(filepath, sample_size=8192):
    """Heuristic: file is binary if it contains null bytes in the first chunk."""
    try:
        with open(filepath, "rb") as f:
            chunk = f.read(sample_size)
        return b"\x00" in chunk
    except (OSError, IOError):
        return True


class DocumentScanner:
    """Scans a directory tree, hashes files, and populates the document ledger."""

    def __init__(self, db_conn, config):
        self.db = db_conn
        extraction_cfg = config.get("extraction", {})
        # Support both singular and plural config keys
        dirs = extraction_cfg.get("datafiles_dirs", [])
        if not dirs:
            single = extraction_cfg.get("datafiles_dir", "")
            dirs = [single] if single else []
        self.datafiles_dirs = dirs
        # Keep singular for backward compat
        self.datafiles_dir = dirs[0] if dirs else ""
        self.include_types = extraction_cfg.get("include_types", DEFAULT_INCLUDE_TYPES)
        self.exclude_types = extraction_cfg.get("exclude_types", DEFAULT_EXCLUDE_TYPES)
        self.exclude_patterns = extraction_cfg.get("exclude_patterns", DEFAULT_EXCLUDE_PATTERNS)
        self.max_file_size = extraction_cfg.get("max_file_size_bytes", DEFAULT_MAX_FILE_SIZE)

    def scan(self):
        """Walk all datafiles directories, hash each file, update ledger.

        Returns dict with counts: discovered, indexed, skipped, unchanged, changed.
        """
        if not self.datafiles_dirs:
            logger.error("No datafiles directories configured")
            return {"error": "no datafiles_dirs configured"}

        stats = {
            "discovered": 0, "indexed": 0, "skipped": 0,
            "unchanged": 0, "changed": 0, "errors": 0,
        }
        now = datetime.now(timezone.utc).isoformat()

        for root_dir in self.datafiles_dirs:
            if not os.path.isdir(root_dir):
                logger.warning(f"datafiles_dir not a directory, skipping: {root_dir!r}")
                continue

            for dirpath, dirnames, filenames in os.walk(root_dir):
                dirnames[:] = [d for d in dirnames if d != ".git"]
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    rel_path = os.path.relpath(filepath, root_dir)

                    try:
                        result = self._process_file(filepath, rel_path, filename, now)
                        stats[result] += 1
                    except Exception as e:
                        logger.warning(f"Error processing {rel_path}: {e}")
                        stats["errors"] += 1

        self.db.commit()
        logger.info(
            f"Scan complete: {stats['discovered']} new, {stats['indexed']} indexed, "
            f"{stats['skipped']} skipped, {stats['unchanged']} unchanged, "
            f"{stats['changed']} changed, {stats['errors']} errors"
        )
        return stats

    def _process_file(self, filepath, rel_path, filename, now):
        """Process a single file. Returns status string for stats counting."""
        ext = os.path.splitext(filename)[1].lower()
        size = os.path.getsize(filepath)
        mtime = datetime.fromtimestamp(
            os.path.getmtime(filepath), tz=timezone.utc
        ).isoformat()

        # Check if file already in ledger by path — most recent entry wins
        existing = self.db.execute(
            "SELECT sha256, status, last_modified FROM document_ledger "
            "WHERE file_path = ? AND source_changed = 0 "
            "ORDER BY indexed_at DESC LIMIT 1",
            (rel_path,),
        ).fetchone()

        if existing:
            # File already known — check for changes
            if existing["last_modified"] == mtime and existing["status"] != "error":
                return "unchanged"
            # mtime changed — re-hash to confirm
            sha = _sha256_file(filepath)
            if sha == existing["sha256"]:
                # Same content, just mtime drift — update mtime
                self.db.execute(
                    "UPDATE document_ledger SET last_modified = ? WHERE sha256 = ?",
                    (mtime, sha),
                )
                return "unchanged"
            # Content actually changed — flag old entry, don't delete
            # Old beliefs stay in graph with source_changed marker.
            # Operator decides deprecation.
            self.db.execute(
                "UPDATE document_ledger SET source_changed = 1 WHERE sha256 = ?",
                (existing["sha256"],),
            )
            logger.info(f"Document changed: {rel_path} (old SHA {existing['sha256'][:12]})")
            result_tag = "changed"
        else:
            result_tag = "discovered"

        # Hash the file
        sha = _sha256_file(filepath)

        # Check if this exact content already exists (dedup by SHA)
        dup = self.db.execute(
            "SELECT sha256 FROM document_ledger WHERE sha256 = ?", (sha,)
        ).fetchone()
        if dup:
            return "unchanged"

        # Determine skip reason
        skip_reason = self._should_skip(filepath, filename, ext, size)

        status = "skipped" if skip_reason else "indexed"
        self.db.execute(
            """INSERT OR REPLACE INTO document_ledger
               (sha256, filename, file_path, file_type, size_bytes,
                status, skip_reason, indexed_at, last_modified)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (sha, filename, rel_path, ext, size, status, skip_reason, now, mtime),
        )

        if skip_reason:
            logger.debug(f"Skipped {rel_path}: {skip_reason}")
            return "skipped"

        if result_tag == "changed":
            return "changed"
        return "indexed" if result_tag == "discovered" else result_tag

    def _should_skip(self, filepath, filename, ext, size):
        """Return skip reason string, or None if file should be indexed."""
        # Exclude patterns (glob)
        for pattern in self.exclude_patterns:
            if fnmatch(filename, pattern) or fnmatch(filepath, pattern):
                return f"exclude_pattern:{pattern}"

        # Exclude by type
        if ext in self.exclude_types:
            return f"exclude_type:{ext}"

        # Include-list check (if include list is set, file must match)
        if self.include_types and ext not in self.include_types:
            return f"not_in_include_types:{ext}"

        # Size check
        if size > self.max_file_size:
            return f"too_large:{size}>{self.max_file_size}"

        # Binary check
        if _is_binary(filepath):
            return "binary_content"

        return None

    def get_indexed_documents(self, limit=None):
        """Return documents with status='indexed', FIFO order (oldest first)."""
        query = (
            "SELECT sha256, filename, file_path, file_type, size_bytes "
            "FROM document_ledger WHERE status = 'indexed' OR (status = 'ingested' AND belief_count = 0) "
            "ORDER BY indexed_at ASC"
        )
        if limit:
            query += f" LIMIT {int(limit)}"
        return [dict(row) for row in self.db.execute(query).fetchall()]

    def get_ledger_summary(self):
        """Return status counts from the ledger."""
        rows = self.db.execute(
            "SELECT status, COUNT(*) as cnt FROM document_ledger GROUP BY status"
        ).fetchall()
        return {row["status"]: row["cnt"] for row in rows}

    def get_belief_count(self, sha256):
        """Get belief count for a document."""
        try:
            row = self.db.execute(
                "SELECT belief_count FROM document_ledger WHERE sha256 = ?",
                (sha256,),
            ).fetchone()
            return row["belief_count"] if row else 0
        except Exception:
            return 0

    def mark_ingested(self, sha256, belief_count):
        """Mark a document as ingested with belief count."""
        now = datetime.now(timezone.utc).isoformat()
        self.db.execute(
            "UPDATE document_ledger SET status = 'ingested', ingested_at = ?, belief_count = ? "
            "WHERE sha256 = ?",
            (now, belief_count, sha256),
        )
        self.db.commit()

    def mark_error(self, sha256, reason):
        """Mark a document as errored."""
        self.db.execute(
            "UPDATE document_ledger SET status = 'error', skip_reason = ? WHERE sha256 = ?",
            (reason, sha256),
        )
        self.db.commit()
