"""GitHub-related utility functions shared across views, indexing, and QA pipeline."""

from __future__ import annotations
import os
import base64
import requests
import re
from urllib.parse import urlparse
from typing import Optional, List, Dict

def _get_github_token() -> Optional[str]:
    """Read GITHUB_TOKEN from environment at call time (not import time).

    Reading at call time ensures that tokens set after module import (e.g. by
    Django's dotenv loading or test harnesses) are picked up correctly.
    """
    return os.environ.get("GITHUB_TOKEN")


def parse_github_url(repo_url: str):
    """Return normalized (owner, repo) from a GitHub URL in varied forms.

    Normalization handled:
      - Leading/trailing whitespace.
      - Optional scheme (assumes https if missing).
      - http:// or https:// treated identically.
      - Trailing slashes removed.
      - Optional .git suffix removed.
      - Accepts extra path segments (e.g., /tree/main) – only first two used.
      - Supports scp-like form: git@github.com:owner/repo(.git)

    Raises ValueError if owner/repo cannot be extracted.
    """
    if not repo_url or not isinstance(repo_url, str):  # Basic type guard
        raise ValueError("Invalid repository URL: expected non-empty string")

    raw = repo_url.strip()

    # Handle scp-like syntax: git@github.com:owner/repo.git
    if raw.startswith("git@"):
        # Split at first ':' after host
        try:
            path_part = raw.split(":", 1)[1]
        except IndexError as e:  # noqa: BLE001
            raise ValueError(f"Malformed GitHub scp-style URL: {repo_url}") from e
        path = path_part
    else:
        # Prepend scheme if missing (treat bare github.com/... as https)
        if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", raw):
            raw = "https://" + raw.lstrip("/")
        parsed = urlparse(raw)
        # If user passed something like github.com/owner/repo (no scheme), urlparse would put it in path – handled above.
        path = parsed.path or ""

    # Strip query/fragment if any leaked in earlier
    path = path.split("?")[0].split("#")[0]
    # Normalize suffixes and separators
    path = path.removesuffix(".git").rstrip("/").lstrip("/")
    segments = [seg for seg in path.split("/") if seg]
    if len(segments) < 2:
        raise ValueError(f"Could not parse owner/repo from URL: {repo_url}")
    owner, repo = segments[0].lower(), segments[1].lower()
    return owner, repo


def fetch_repo_tree(repo_url: str):
    """Fetch the repository file tree (blob entries) using GitHub API.

    Returns a list of blob dictionaries or None on error.
    """
    token = _get_github_token()
    if not token:
        print("Error: GITHUB_TOKEN environment variable not set.")
        return None
    try:
        owner, repo = parse_github_url(repo_url)
        headers = {"Authorization": f"token {token}"}
        repo_api = f"https://api.github.com/repos/{owner}/{repo}"
        repo_info = requests.get(repo_api, headers=headers, timeout=15).json()
        default_branch = repo_info.get("default_branch", "main")
        ref_url = f"{repo_api}/git/refs/heads/{default_branch}"
        ref = requests.get(ref_url, headers=headers, timeout=15).json()
        sha = ref.get("object", {}).get("sha")
        if not sha:
            return None
        tree_url = f"{repo_api}/git/trees/{sha}?recursive=1"
        tree = requests.get(tree_url, headers=headers, timeout=30).json().get("tree", [])
        blobs = [item for item in tree if item.get("type") == "blob"]
        print(f"Fetched {len(blobs)} blob paths from GitHub API.")
        return blobs
    except Exception as e:  # noqa: BLE001
        print(f"Error fetching repo tree: {e}")
        return None


def get_file_content(repo_url: str, relative_file_path: str) -> Optional[str]:
    """Retrieve file content from GitHub (UTF-8 decoded, ignoring errors)."""
    token = _get_github_token()
    if not token:
        print("Error: GITHUB_TOKEN environment variable is not set.")
        return None
    try:
        owner, repo = parse_github_url(repo_url)
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{relative_file_path.lstrip('/')}"
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        }
        resp = requests.get(api_url, headers=headers, timeout=10)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        if data.get("type") != "file" or "content" not in data:
            return None
        raw_bytes = base64.b64decode(data["content"])
        return raw_bytes.decode("utf-8", errors="ignore")
    except Exception as e:  # noqa: BLE001
        print(f"Warning: GitHub API error fetching {relative_file_path}: {e}")
        return None


__all__ = ["parse_github_url", "fetch_repo_tree", "get_file_content"]


def fetch_json(url: str, headers: Dict[str, str]):
    """Thin helper around requests.get().json() to keep _get_latest_commit concise."""
    import requests

    return requests.get(url, headers=headers, timeout=15).json()


def _get_latest_commit(repo_url: str) -> str | None:
    """Resolve the current HEAD commit SHA for the repo's default branch.

    Used to tie a generated summary to a specific repository state for potential
    staleness detection. Returns None if token missing or any API failure occurs.
    """
    token = _get_github_token()
    if not token:
        return None  # Anonymous requests would be rate limited / less reliable
    try:
        owner, repo = parse_github_url(repo_url)
        headers = {"Authorization": f"token {token}"}
        repo_api = f"https://api.github.com/repos/{owner}/{repo}"
        repo_info = fetch_json(repo_api, headers)
        default_branch = repo_info.get("default_branch", "main")
        ref_url = f"{repo_api}/git/refs/heads/{default_branch}"
        ref = fetch_json(ref_url, headers)
        return ref.get("object", {}).get("sha")
    except Exception:  # noqa: BLE001 – treat failure as non-fatal
        return None
