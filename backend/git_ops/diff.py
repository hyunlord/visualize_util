"""Git diff and commit history operations.

Provides structured dataclass-based results for comparing commits, listing
changed files, and retrieving commit history.  Functions are synchronous and
designed to be called from an async context via ``asyncio.to_thread`` when
needed, or used directly in synchronous code paths.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from git import Repo
from git.diff import Diff
from git.exc import BadName, InvalidGitRepositoryError, NoSuchPathError

logger = logging.getLogger(__name__)

# Map GitPython change_type codes to human-readable status strings.
_CHANGE_TYPE_MAP: dict[str, str] = {
    "A": "added",
    "D": "deleted",
    "M": "modified",
    "R": "renamed",
    "C": "copied",
    "T": "type_changed",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ChangedFile:
    """A file that changed between two git commits.

    Attributes:
        path: Relative path of the file (current location for renames).
        change_type: One of ``"added"``, ``"modified"``, ``"deleted"``,
            or ``"renamed"``.
        old_path: Previous path for renames, ``None`` otherwise.
    """

    path: str
    change_type: str  # "added" | "modified" | "deleted" | "renamed"
    old_path: str | None = None


@dataclass(slots=True)
class CommitInfo:
    """Metadata for a single git commit.

    Attributes:
        sha: Full 40-character hexadecimal commit SHA.
        message: Commit message with leading/trailing whitespace stripped.
        author: Author string in ``"Name <email>"`` format.
        timestamp: Timezone-aware UTC datetime of the commit.
    """

    sha: str
    message: str
    author: str
    timestamp: datetime


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_changed_files_since(
    repo_path: str,
    since_sha: str,
) -> list[ChangedFile]:
    """Return all files changed between *since_sha* and the current HEAD.

    Parameters:
        repo_path: Absolute path to the git repository working tree.
        since_sha: The older commit SHA (or any ``rev-parse`` expression)
            to compare from.

    Returns:
        A list of ``ChangedFile`` instances sorted by path.

    Raises:
        ValueError: When *repo_path* is not a valid git repository or
            *since_sha* cannot be resolved.
    """
    repo = _open_repo(repo_path)

    try:
        old_commit = repo.commit(since_sha)
    except BadName as exc:
        raise ValueError(
            f"Cannot resolve commit SHA '{since_sha}' in {repo_path}"
        ) from exc

    head_commit = repo.head.commit
    diffs = old_commit.diff(head_commit)

    changes = [_diff_to_changed_file(d) for d in diffs]
    changes.sort(key=lambda c: c.path)

    logger.info(
        "Changed files since %s: %d files",
        since_sha[:8],
        len(changes),
    )
    return changes


def get_commit_log(
    repo_path: str,
    max_count: int = 20,
) -> list[CommitInfo]:
    """Return recent commits for the current branch.

    Parameters:
        repo_path: Absolute path to the git repository working tree.
        max_count: Maximum number of commits to return (most recent first).

    Returns:
        A list of ``CommitInfo`` instances, newest first.

    Raises:
        ValueError: When *repo_path* is not a valid git repository or
            *max_count* is less than 1.
    """
    if max_count < 1:
        raise ValueError("max_count must be at least 1")

    repo = _open_repo(repo_path)
    commits: list[CommitInfo] = []

    for commit in repo.iter_commits(max_count=max_count):
        commit_dt = datetime.fromtimestamp(
            commit.committed_date, tz=timezone.utc,
        )
        commits.append(
            CommitInfo(
                sha=commit.hexsha,
                message=commit.message.strip(),
                author=f"{commit.author.name} <{commit.author.email}>",
                timestamp=commit_dt,
            )
        )

    logger.info(
        "Retrieved %d commits from %s",
        len(commits),
        repo_path,
    )
    return commits


def get_file_diff(
    repo_path: str,
    file_path: str,
    old_sha: str,
    new_sha: str,
) -> str:
    """Return a unified diff for a single file between two commits.

    Parameters:
        repo_path: Absolute path to the git repository working tree.
        file_path: Relative path of the file within the repository.
        old_sha: The older commit SHA (or any ``rev-parse`` expression).
        new_sha: The newer commit SHA (or any ``rev-parse`` expression).

    Returns:
        A unified diff string.  Returns an empty string if the file was
        not changed between the two commits or does not exist in either.

    Raises:
        ValueError: When *repo_path* is not a valid git repository or
            the specified SHAs cannot be resolved.
    """
    repo = _open_repo(repo_path)

    try:
        old_commit = repo.commit(old_sha)
    except BadName as exc:
        raise ValueError(
            f"Cannot resolve old commit SHA '{old_sha}' in {repo_path}"
        ) from exc

    try:
        new_commit = repo.commit(new_sha)
    except BadName as exc:
        raise ValueError(
            f"Cannot resolve new commit SHA '{new_sha}' in {repo_path}"
        ) from exc

    # Compute the diff restricted to the specific file path.
    diffs = old_commit.diff(new_commit, paths=[file_path], create_patch=True)

    if not diffs:
        return ""

    # Concatenate all diff hunks (typically there is exactly one for a
    # single file, but renames can produce two entries).
    parts: list[str] = []
    for diff_item in diffs:
        diff_text = diff_item.diff
        if isinstance(diff_text, bytes):
            diff_text = diff_text.decode("utf-8", errors="replace")
        if diff_text:
            parts.append(diff_text)

    return "\n".join(parts)


def get_changed_files_between(
    repo_path: str,
    old_sha: str,
    new_sha: str,
) -> list[ChangedFile]:
    """Return all files changed between two arbitrary commits.

    Unlike :func:`get_changed_files_since`, this function does not
    implicitly compare against HEAD -- the caller specifies both endpoints.

    Parameters:
        repo_path: Absolute path to the git repository working tree.
        old_sha: The older commit SHA.
        new_sha: The newer commit SHA.

    Returns:
        A list of ``ChangedFile`` instances sorted by path.

    Raises:
        ValueError: When *repo_path* is not a valid git repository or
            the specified SHAs cannot be resolved.
    """
    repo = _open_repo(repo_path)

    try:
        old_commit = repo.commit(old_sha)
    except BadName as exc:
        raise ValueError(
            f"Cannot resolve old commit SHA '{old_sha}' in {repo_path}"
        ) from exc

    try:
        new_commit = repo.commit(new_sha)
    except BadName as exc:
        raise ValueError(
            f"Cannot resolve new commit SHA '{new_sha}' in {repo_path}"
        ) from exc

    diffs = old_commit.diff(new_commit)
    changes = [_diff_to_changed_file(d) for d in diffs]
    changes.sort(key=lambda c: c.path)

    logger.info(
        "Changed files %s..%s: %d files",
        old_sha[:8],
        new_sha[:8],
        len(changes),
    )
    return changes


def get_current_sha(repo_path: str) -> str:
    """Return the full hexadecimal SHA of the current HEAD commit.

    Parameters:
        repo_path: Absolute path to the git repository.

    Returns:
        The 40-character hex SHA string.

    Raises:
        ValueError: When *repo_path* is not a valid git repository.
    """
    repo = _open_repo(repo_path)
    return repo.head.commit.hexsha


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _open_repo(repo_path: str) -> Repo:
    """Open and return a ``git.Repo`` instance, raising on invalid paths."""
    resolved = str(Path(repo_path).resolve())
    try:
        return Repo(resolved)
    except (InvalidGitRepositoryError, NoSuchPathError) as exc:
        raise ValueError(
            f"'{resolved}' is not a valid git repository"
        ) from exc


def _diff_to_changed_file(diff_item: Diff) -> ChangedFile:
    """Convert a GitPython ``Diff`` object to a ``ChangedFile``."""
    change_code = diff_item.change_type or "M"
    change_type = _CHANGE_TYPE_MAP.get(change_code, "modified")

    # Normalise change types that don't have their own category.
    if change_type in ("copied", "type_changed"):
        change_type = "modified"

    # Determine the current file path.
    if diff_item.b_path:
        path = diff_item.b_path
    elif diff_item.a_path:
        path = diff_item.a_path
    else:
        path = ""

    # Track previous path for renames.
    old_path: str | None = None
    if change_code in ("R", "C") and diff_item.a_path:
        old_path = diff_item.a_path

    return ChangedFile(path=path, change_type=change_type, old_path=old_path)
