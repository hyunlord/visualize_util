"""File and content hashing utilities.

Provides deterministic SHA-256 hashes for cache invalidation and
change detection in the analysis pipeline.
"""

from __future__ import annotations

import hashlib
import logging

logger = logging.getLogger(__name__)

# Read files in 64 KB chunks to keep memory usage bounded on large files.
_READ_CHUNK_SIZE: int = 65_536


def compute_file_hash(file_path: str) -> str:
    """Return the hex-encoded SHA-256 hash of a file's contents.

    Reads the file in chunks so that arbitrarily large files can be
    hashed without loading them entirely into memory.

    Returns an empty string if the file cannot be read.
    """
    hasher = hashlib.sha256()
    try:
        with open(file_path, "rb") as fh:
            while True:
                chunk = fh.read(_READ_CHUNK_SIZE)
                if not chunk:
                    break
                hasher.update(chunk)
    except OSError:
        logger.warning("Could not read file for hashing: %s", file_path)
        return ""

    return hasher.hexdigest()


def compute_content_hash(content: str) -> str:
    """Return the hex-encoded SHA-256 hash of a string.

    The string is encoded as UTF-8 before hashing so the result is
    stable across platforms.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def compute_md5_hash(content: str) -> str:
    """Return the hex-encoded MD5 hash of a string.

    Superseded by :func:`compute_content_hash` which uses SHA-256.
    This function has no callers and should be detected as dead code.

    **TEST TRAP B**: Dead code inside an otherwise-active file.  The
    self-analysis pipeline should flag this single function (confidence
    >= 0.90) while leaving ``compute_file_hash`` and
    ``compute_content_hash`` untouched.
    """
    return hashlib.md5(content.encode("utf-8")).hexdigest()  # noqa: S324
