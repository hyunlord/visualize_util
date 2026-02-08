"""Repository management API endpoints.

Provides CRUD operations for tracked Git repositories:
- Register a remote repository (clone it) or a local directory.
- List, inspect, and delete tracked repositories.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import get_settings
from backend.database import get_db
from backend.git_ops.clone import clone_repo_async, validate_local_repo
from backend.git_ops.diff import get_current_sha
from backend.models.db_models import Repository
from backend.models.schemas import RepoCreateRequest, RepoResponse

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# POST /repos  --  create / register a repository
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=RepoResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register or clone a Git repository",
)
async def create_repo(
    body: RepoCreateRequest,
    db: AsyncSession = Depends(get_db),
) -> Repository:
    """Register a repository for analysis.

    If *url* is provided the repository is cloned into the configured
    ``REPOS_DIR``.  If only *local_path* is supplied the directory is
    validated as a Git repository and linked directly.
    """
    settings = get_settings()

    repo = Repository(
        url=body.url,
        branch=body.branch,
    )

    if body.url:
        # Clone the repository into REPOS_DIR / <repo_id>
        target_dir = str(settings.REPOS_DIR / repo.id)
        try:
            local_path = await clone_repo_async(
                url=body.url,
                dest_dir=target_dir,
                branch=body.branch,
            )
        except Exception as exc:
            logger.error("Failed to clone %s: %s", body.url, exc)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to clone repository: {exc}",
            ) from exc

        repo.local_path = local_path

    elif body.local_path:
        # Validate that the path exists and is a git repository
        path = Path(body.local_path).resolve()
        if not path.is_dir():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Directory does not exist: {body.local_path}",
            )
        if not validate_local_repo(str(path)):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Not a valid Git repository: {body.local_path}",
            )
        repo.local_path = str(path)

    # Capture the current commit SHA
    try:
        repo.last_commit_sha = get_current_sha(repo.local_path)
    except Exception:
        logger.warning(
            "Could not read HEAD SHA for %s", repo.local_path, exc_info=True,
        )

    db.add(repo)
    await db.flush()
    await db.refresh(repo)
    return repo


# ---------------------------------------------------------------------------
# GET /repos  --  list all repositories
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=list[RepoResponse],
    summary="List all tracked repositories",
)
async def list_repos(
    db: AsyncSession = Depends(get_db),
) -> list[Repository]:
    """Return every tracked repository ordered by creation date descending."""
    result = await db.execute(
        select(Repository).order_by(Repository.created_at.desc())
    )
    return list(result.scalars().all())


# ---------------------------------------------------------------------------
# GET /repos/{repo_id}  --  get a single repository
# ---------------------------------------------------------------------------


@router.get(
    "/{repo_id}",
    response_model=RepoResponse,
    summary="Get repository details",
)
async def get_repo(
    repo_id: str,
    db: AsyncSession = Depends(get_db),
) -> Repository:
    """Return details for a single tracked repository."""
    repo = await db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository not found: {repo_id}",
        )
    return repo


# ---------------------------------------------------------------------------
# DELETE /repos/{repo_id}  --  delete a repository and all associated data
# ---------------------------------------------------------------------------


@router.delete(
    "/{repo_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a repository and all associated data",
)
async def delete_repo(
    repo_id: str,
    db: AsyncSession = Depends(get_db),
) -> None:
    """Delete a tracked repository and cascade-remove all snapshots, nodes,
    edges, and features.

    If the repository was cloned (has a *url*), the local clone directory
    is also removed from disk.
    """
    repo = await db.get(Repository, repo_id)
    if repo is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository not found: {repo_id}",
        )

    settings = get_settings()
    cloned_path = settings.REPOS_DIR / repo_id

    await db.delete(repo)
    await db.flush()

    # Remove the cloned directory from disk if it was created by us
    if repo.url and cloned_path.is_dir():
        try:
            shutil.rmtree(cloned_path)
            logger.info("Removed cloned repo directory: %s", cloned_path)
        except OSError:
            logger.warning(
                "Failed to remove cloned directory: %s",
                cloned_path,
                exc_info=True,
            )
