"""Incremental diff analysis for detecting repository changes between commits.

Provides structured diff information by comparing two git commits, identifying
added, modified, deleted, and renamed files.  This enables the analysis engine
to perform incremental re-analysis on only the changed portions of a codebase
rather than re-scanning everything from scratch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from git import Repo
from git.diff import Diff
from git.exc import BadName, InvalidGitRepositoryError, NoSuchPathError

logger = logging.getLogger(__name__)

# Map GitPython change_type codes to semantic status strings.
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
class FileChange:
    """A single file change between two commits.

    Attributes:
        path: Relative file path (current location for renames).
        change_type: One of ``"added"``, ``"modified"``, ``"deleted"``,
            ``"renamed"``, ``"copied"``, or ``"type_changed"``.
        old_path: Previous path for renames and copies, ``None`` otherwise.
    """

    path: str
    change_type: str
    old_path: str | None = None


@dataclass(slots=True)
class DiffResult:
    """Aggregated result of a file-level diff between two commits.

    Provides categorised lists for each change type as well as a flat
    ``all_changes`` accessor for iteration convenience.

    Attributes:
        added_files: Files that exist only in the new commit.
        modified_files: Files changed between the two commits.
        deleted_files: Files that were removed in the new commit.
        renamed_files: Files that were moved or renamed.
        old_sha: The older commit SHA used for comparison.
        new_sha: The newer commit SHA used for comparison.
    """

    added_files: list[FileChange] = field(default_factory=list)
    modified_files: list[FileChange] = field(default_factory=list)
    deleted_files: list[FileChange] = field(default_factory=list)
    renamed_files: list[FileChange] = field(default_factory=list)
    old_sha: str = ""
    new_sha: str = ""

    @property
    def all_changes(self) -> list[FileChange]:
        """Return every change across all categories in a single flat list."""
        return (
            self.added_files
            + self.modified_files
            + self.deleted_files
            + self.renamed_files
        )

    @property
    def total_changed(self) -> int:
        """Total number of changed files across all categories."""
        return (
            len(self.added_files)
            + len(self.modified_files)
            + len(self.deleted_files)
            + len(self.renamed_files)
        )

    @property
    def has_changes(self) -> bool:
        """Return ``True`` if any file was changed."""
        return self.total_changed > 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_diff(
    repo_path: str,
    old_sha: str,
    new_sha: str,
) -> DiffResult:
    """Compute a file-level diff between two commits in a repository.

    Parameters:
        repo_path: Absolute path to the git repository working tree.
        old_sha: The older commit SHA (or any ``rev-parse`` expression).
        new_sha: The newer commit SHA (or any ``rev-parse`` expression).

    Returns:
        A ``DiffResult`` with categorised file change lists.

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
    result = DiffResult(old_sha=old_sha, new_sha=new_sha)

    for diff_item in diffs:
        file_change = _diff_to_file_change(diff_item)
        _categorise_change(result, file_change)

    logger.info(
        "Diff %s..%s: +%d ~%d -%d R%d",
        old_sha[:8],
        new_sha[:8],
        len(result.added_files),
        len(result.modified_files),
        len(result.deleted_files),
        len(result.renamed_files),
    )
    return result


def get_changed_files(
    repo_path: str,
    since_sha: str,
) -> list[str]:
    """Return relative paths of all files changed since a given commit.

    Compares *since_sha* against the current ``HEAD`` of the repository.
    This is useful for determining which files need incremental re-analysis.

    Parameters:
        repo_path: Absolute path to the git repository working tree.
        since_sha: The commit SHA to diff from (compared against HEAD).

    Returns:
        A deduplicated, sorted list of relative file paths.

    Raises:
        ValueError: When *repo_path* is not a valid git repository or
            the specified SHA cannot be resolved.
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

    paths: set[str] = set()
    for diff_item in diffs:
        # Collect both source and destination paths to cover renames.
        if diff_item.a_path:
            paths.add(diff_item.a_path)
        if diff_item.b_path:
            paths.add(diff_item.b_path)

    sorted_paths = sorted(paths)
    logger.info(
        "Files changed since %s: %d paths",
        since_sha[:8],
        len(sorted_paths),
    )
    return sorted_paths


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


def _diff_to_file_change(diff_item: Diff) -> FileChange:
    """Convert a GitPython ``Diff`` object to a ``FileChange``."""
    change_code = diff_item.change_type or "M"
    change_type = _CHANGE_TYPE_MAP.get(change_code, "modified")

    # Determine the current (destination) path.
    if diff_item.b_path:
        path = diff_item.b_path
    elif diff_item.a_path:
        path = diff_item.a_path
    else:
        path = ""

    # For renames/copies, track the previous path.
    old_path: str | None = None
    if change_code in ("R", "C") and diff_item.a_path:
        old_path = diff_item.a_path

    return FileChange(path=path, change_type=change_type, old_path=old_path)


def _categorise_change(result: DiffResult, change: FileChange) -> None:
    """Append a ``FileChange`` to the appropriate list in *result*."""
    if change.change_type == "added":
        result.added_files.append(change)
    elif change.change_type == "deleted":
        result.deleted_files.append(change)
    elif change.change_type == "renamed":
        result.renamed_files.append(change)
    else:
        # "modified", "copied", "type_changed" all go into modified.
        result.modified_files.append(change)
