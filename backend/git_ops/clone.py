"""Git clone and local repository management operations.

Provides both synchronous and async-friendly functions for cloning repositories,
fetching updates, and querying repository metadata.  All blocking git operations
are wrapped in ``asyncio.to_thread`` for the async variants so the event loop is
never stalled.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

import git
from git import InvalidGitRepositoryError, NoSuchPathError, Repo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def clone_repo(
    url: str,
    dest_dir: str,
    branch: str = "main",
) -> str:
    """Clone a remote git repository to a local directory.

    If *dest_dir* already contains a repository whose ``origin`` remote
    matches *url*, a fetch-and-checkout is performed instead of a full
    re-clone.  This makes the operation idempotent.

    Parameters:
        url: Remote repository URL (HTTPS or SSH).
        dest_dir: Local directory to clone into.  Parent directories are
            created automatically if they do not exist.
        branch: Branch to check out after cloning.

    Returns:
        Absolute path to the local repository working tree.

    Raises:
        ValueError: When *url* or *dest_dir* is empty.
        git.GitCommandError: When the underlying git operation fails.
    """
    if not url:
        raise ValueError("url must be a non-empty string")
    if not dest_dir:
        raise ValueError("dest_dir must be a non-empty path")

    target = Path(dest_dir)

    # If the target already exists and is the same repo, just fetch.
    if target.exists() and _is_same_remote(target, url):
        logger.info(
            "Repository already exists at %s; fetching branch '%s'",
            target,
            branch,
        )
        _fetch_and_checkout(str(target), branch)
        return str(target.resolve())

    logger.info("Cloning %s (branch=%s) into %s", url, branch, target)
    target.mkdir(parents=True, exist_ok=True)

    Repo.clone_from(
        url,
        str(target),
        branch=branch,
        single_branch=False,
    )

    logger.info("Clone complete: %s", target.resolve())
    return str(target.resolve())


async def clone_repo_async(
    url: str,
    dest_dir: str,
    branch: str = "main",
) -> str:
    """Async wrapper around :func:`clone_repo`.

    Runs the blocking clone/fetch operation in a thread pool so the
    event loop is never blocked.
    """
    return await asyncio.to_thread(clone_repo, url, dest_dir, branch)


def validate_local_repo(path: str) -> bool:
    """Check whether *path* is a valid git repository.

    Parameters:
        path: Absolute or relative path to test.

    Returns:
        ``True`` if *path* is a valid git repository, ``False`` otherwise.
    """
    try:
        Repo(path)
        return True
    except (InvalidGitRepositoryError, NoSuchPathError):
        return False


def get_current_sha(repo_path: str) -> str:
    """Return the full 40-character hexadecimal SHA of HEAD.

    Parameters:
        repo_path: Absolute path to the git repository.

    Returns:
        The full hex SHA string of the current HEAD commit.

    Raises:
        ValueError: When *repo_path* is not a valid git repository.
    """
    repo = _open_repo(repo_path)
    return repo.head.commit.hexsha


def get_repo_branch(repo_path: str) -> str:
    """Return the name of the currently checked-out branch.

    If the repository is in detached HEAD state, returns ``"HEAD"``.

    Parameters:
        repo_path: Absolute path to the git repository.

    Returns:
        The active branch name, or ``"HEAD"`` if detached.

    Raises:
        ValueError: When *repo_path* is not a valid git repository.
    """
    repo = _open_repo(repo_path)
    try:
        return repo.active_branch.name
    except TypeError:
        # Detached HEAD state.
        return "HEAD"


def fetch_latest(repo_path: str, branch: str = "main") -> str:
    """Fetch the latest changes from origin and return the new HEAD SHA.

    Performs ``git fetch origin`` followed by a checkout and fast-forward
    merge of the specified branch.

    Parameters:
        repo_path: Absolute path to the git repository.
        branch: The branch to fetch and check out.

    Returns:
        The full hex SHA of HEAD after the fetch/merge.

    Raises:
        ValueError: When *repo_path* is not a valid git repository.
        git.GitCommandError: When the fetch or merge fails.
    """
    _fetch_and_checkout(repo_path, branch)
    repo = _open_repo(repo_path)
    new_sha = repo.head.commit.hexsha
    logger.info(
        "Fetched latest for branch '%s': HEAD is now %s",
        branch,
        new_sha[:8],
    )
    return new_sha


async def fetch_latest_async(repo_path: str, branch: str = "main") -> str:
    """Async wrapper around :func:`fetch_latest`."""
    return await asyncio.to_thread(fetch_latest, repo_path, branch)


def get_repo_info(repo_path: str) -> dict:
    """Return metadata about the repository at *repo_path*.

    Returns:
        A dict with keys ``branch``, ``commit_sha``, ``commit_message``,
        and ``commit_date`` (ISO-8601 UTC string).

    Raises:
        ValueError: When *repo_path* is not a valid git repository.
    """
    repo = _open_repo(repo_path)
    head_commit = repo.head.commit

    try:
        branch_name = repo.active_branch.name
    except TypeError:
        branch_name = "HEAD"

    commit_dt = datetime.fromtimestamp(
        head_commit.committed_date, tz=timezone.utc,
    )

    return {
        "branch": branch_name,
        "commit_sha": head_commit.hexsha,
        "commit_message": head_commit.message.strip(),
        "commit_date": commit_dt.isoformat(),
    }


async def get_repo_info_async(repo_path: str) -> dict:
    """Async wrapper around :func:`get_repo_info`."""
    return await asyncio.to_thread(get_repo_info, repo_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _open_repo(repo_path: str) -> Repo:
    """Open a ``Repo`` instance, raising ``ValueError`` on invalid paths."""
    resolved = str(Path(repo_path).resolve())
    try:
        return Repo(resolved)
    except (InvalidGitRepositoryError, NoSuchPathError) as exc:
        raise ValueError(
            f"'{resolved}' is not a valid git repository"
        ) from exc


def _fetch_and_checkout(repo_path: str, branch: str) -> None:
    """Fetch all remotes and check out *branch* with fast-forward merge.

    If the branch does not exist locally but exists on ``origin``, a new
    local tracking branch is created.
    """
    repo = _open_repo(repo_path)

    if not repo.remotes:
        raise git.GitCommandError(
            "No remotes configured",
            status=128,
        )

    origin = repo.remotes.origin
    origin.fetch()

    if branch in repo.heads:
        repo.heads[branch].checkout()
    else:
        # Attempt to create a local tracking branch from the remote.
        remote_ref = f"origin/{branch}"
        remote_ref_names = [str(ref) for ref in origin.refs]
        if remote_ref in remote_ref_names:
            repo.create_head(branch, remote_ref).checkout()
        else:
            raise git.GitCommandError(
                f"Branch '{branch}' not found on remote 'origin'",
                status=128,
            )

    # Fast-forward to latest remote state.
    origin.pull(branch)
    logger.info(
        "Fetched and checked out '%s' at %s",
        branch,
        repo.head.commit.hexsha[:8],
    )


def _is_same_remote(target: Path, url: str) -> bool:
    """Check whether *target* is a git repo whose origin matches *url*."""
    try:
        repo = Repo(str(target))
    except (InvalidGitRepositoryError, NoSuchPathError):
        return False

    if not repo.remotes:
        return False

    try:
        origin_urls = list(repo.remotes.origin.urls)
    except (AttributeError, ValueError):
        return False

    normalized_url = _normalize_git_url(url)
    return any(_normalize_git_url(u) == normalized_url for u in origin_urls)


def _normalize_git_url(url: str) -> str:
    """Normalize a git remote URL for comparison.

    Strips trailing ``.git`` suffix and trailing slashes so that
    ``https://github.com/user/repo`` and ``https://github.com/user/repo.git``
    compare equal.
    """
    url = url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    return url.lower()
