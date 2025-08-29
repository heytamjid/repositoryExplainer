"""GitHub-related utility functions shared across views, indexing, and QA pipeline."""

from __future__ import annotations
import os
import base64
import requests
from urllib.parse import urlparse
from typing import Optional, List, Dict

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")


def parse_github_url(repo_url: str):
    """Return (owner, repo) tuple from a GitHub repository URL."""
    path = urlparse(repo_url).path
    path = path.removesuffix(".git")
    owner, repo = path.lstrip("/").split("/")[:2]
    return owner, repo


def fetch_repo_tree(repo_url: str):
    """Fetch the repository file tree (blob entries) using GitHub API.

    Returns a list of blob dictionaries or None on error.
    """
    if not GITHUB_TOKEN:
        print("Error: GITHUB_TOKEN environment variable not set.")
        return None
    try:
        owner, repo = parse_github_url(repo_url)
        headers = {"Authorization": f"token {GITHUB_TOKEN}"}
        repo_api = f"https://api.github.com/repos/{owner}/{repo}"
        repo_info = requests.get(repo_api, headers=headers).json()
        default_branch = repo_info.get("default_branch", "main")
        ref_url = f"{repo_api}/git/refs/heads/{default_branch}"
        ref = requests.get(ref_url, headers=headers).json()
        sha = ref.get("object", {}).get("sha")
        if not sha:
            return None
        tree_url = f"{repo_api}/git/trees/{sha}?recursive=1"
        tree = requests.get(tree_url, headers=headers).json().get("tree", [])
        blobs = [item for item in tree if item.get("type") == "blob"]
        print(f"Fetched {len(blobs)} blob paths from GitHub API.")
        return blobs
    except Exception as e:  # noqa: BLE001
        print(f"Error fetching repo tree: {e}")
        return None


def get_file_content(repo_url: str, relative_file_path: str) -> Optional[str]:
    """Retrieve file content from GitHub (UTF-8 decoded, ignoring errors)."""
    if not GITHUB_TOKEN:
        print("Error: GITHUB_TOKEN environment variable is not set.")
        return None
    try:
        owner, repo = parse_github_url(repo_url)
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{relative_file_path.lstrip('/')}"
        headers = {
            "Authorization": f"token {GITHUB_TOKEN}",
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
