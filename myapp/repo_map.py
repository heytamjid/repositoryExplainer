"""Repository map & lookup utilities extracted from `embedder`.

Provides lightweight access for agentic lookups without importing the heavy
indexing logic.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import os
import json
import time
from base64 import b64decode
from urllib.parse import urlparse

from . import embedder as _embedder

REPO_MAP_SUFFIX = getattr(_embedder, "REPO_MAP_SUFFIX", "_repo_map.json")


def _safe_repo_basename(repo_url: str) -> str:
    return (
        repo_url.replace("://", "_")
        .replace("/", "_")
        .replace(":", "_")
        .replace(".git", "")
    )


def _repo_map_path(repo_url: str, persist_dir: str = "chroma_persist") -> Path:
    return Path(persist_dir) / f"{_safe_repo_basename(repo_url)}{REPO_MAP_SUFFIX}"


def get_repo_map(repo_url: str, persist_dir: str = "chroma_persist") -> Optional[Dict]:
    path = _repo_map_path(repo_url, persist_dir)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def get_repo_map_summary(
    repo_url: str,
    persist_dir: str = "chroma_persist",
    max_files: int = 150,
    max_units_per_file: int = 40,
    max_total_chars: int = 12000,
) -> str:
    rm = get_repo_map(repo_url, persist_dir)
    if not rm or not rm.get("files"):
        return "<no repository map available>"
    lines: List[str] = [
        f"REPO MAP (commit {rm.get('commit','?')}) - files={rm.get('file_count')} created_at={int(rm.get('generated_at',0))}",
        "Format: file_path (unit_count) -> unit_name[start-end]",
    ]
    total_chars = 0
    for f in rm["files"][:max_files]:
        unit_list = f.get("units", [])
        header = f"{f['path']} ({len(unit_list)})"
        lines.append(header)
        total_chars += len(header) + 1
        if total_chars > max_total_chars:
            break
        for u in unit_list[:max_units_per_file]:
            uline = f"  - {u['type']} {u['name']}[{u['start_line']}-{u['end_line']}]"
            lines.append(uline)
            total_chars += len(uline) + 1
            if total_chars > max_total_chars:
                break
        if total_chars > max_total_chars:
            break
    if total_chars > max_total_chars:
        lines.append("<truncated>")
    return "\n".join(lines)


def lookup_repo_items(
    repo_url: str,
    requests_list: List[Dict],
    persist_dir: str = "chroma_persist",
) -> List[Dict]:
    rm = get_repo_map(repo_url, persist_dir)
    if not rm:
        return []
    commit = rm.get("commit")
    file_index = {f["path"]: f for f in rm.get("files", [])}
    outputs: List[Dict] = []
    for req in requests_list[:10]:  # safety limit
        rtype = req.get("type")
        path = req.get("path")
        if not path or path not in file_index:
            continue
        file_entry = file_index[path]
        file_content = _fetch_file_content_github(repo_url, path, commit)
        if not file_content:
            continue
        if rtype == "file":
            outputs.append(
                {
                    "type": "file",
                    "path": path,
                    "content": file_content,
                    "commit": commit,
                }
            )
        elif rtype == "function":
            fname = req.get("name")
            if not fname:
                continue
            unit = next(
                (u for u in file_entry.get("units", []) if u.get("name") == fname),
                None,
            )
            if not unit:
                continue
            try:
                lines = file_content.splitlines()
                start = max(1, unit["start_line"])
                end = min(len(lines), unit["end_line"])
                snippet = "\n".join(lines[start - 1 : end])
            except Exception:  # noqa: BLE001
                snippet = ""
            outputs.append(
                {
                    "type": "function",
                    "path": path,
                    "name": fname,
                    "start_line": unit.get("start_line"),
                    "end_line": unit.get("end_line"),
                    "content": snippet,
                    "commit": commit,
                }
            )
    return outputs


def build_repo_map(
    repo_url: str, persist_dir: str = "chroma_persist"
) -> Optional[Dict]:
    """Construct a repository map on-demand.

    This duplicates the lightweight map produced during indexing (paths + units) so
    the agent can still function if indexing created the embeddings but the map file
    was somehow removed, or if legacy data exists. We do NOT clone here; instead we
    require that the repository has already been indexed (so we can recover commit and
    rely on the embedding manifest jsonl for structure). If no manifest is available,
    we fall back to returning None.
    """
    # Try to locate an embedding chunks manifest to reconstruct minimal map.
    try:
        # Derive safe repo filename similar to embedder logic
        safe_repo = repo_url.replace("://", "_").replace("/", "_").replace(":", "_")
        manifest_guess = Path(persist_dir) / f"{safe_repo}_embedding_chunks.jsonl"
        if not manifest_guess.exists():
            return None
        files: Dict[str, Dict] = {}
        commit = None
        with manifest_guess.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                path = rec.get("file_path")
                unit_name = rec.get("unit_name")
                if not path or not unit_name:
                    continue
                # Skip summary pseudo file
                if path == "repository-summary.md":
                    continue
                fentry = files.setdefault(
                    path,
                    {
                        "path": path,
                        "size": None,
                        "language": None,
                        "units": [],
                    },
                )
                # Heuristic: treat any non file_chunk_* as a unit (function/class/etc.)
                if not unit_name.startswith("file_chunk_") and not any(
                    u.get("name") == unit_name for u in fentry["units"]
                ):
                    start_line = rec.get("start_line") or 1
                    end_line = rec.get("end_line") or start_line
                    fentry["units"].append(
                        {
                            "name": unit_name,
                            "type": rec.get("unit_type") or "unit",
                            "start_line": start_line,
                            "end_line": end_line,
                        }
                    )
        file_list = list(files.values())
        repo_map = {
            "repo_url": repo_url,
            "commit": commit,  # unknown in reconstruction (not stored in manifest recs per chunk)
            "generated_at": time.time(),
            "file_count": len(file_list),
            "files": file_list,
        }
        # Persist
        path = _repo_map_path(repo_url, persist_dir)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                json.dump(repo_map, f, indent=2)
        except Exception:  # noqa: BLE001
            pass
        return repo_map
    except Exception:  # noqa: BLE001
        return None


def _parse_github_url(repo_url: str) -> Tuple[str, str]:
    from urllib.parse import urlparse

    path = urlparse(repo_url).path.rstrip("/")
    path = path.replace(".git", "")
    parts = [p for p in path.split("/") if p]
    if len(parts) < 2:
        raise ValueError("Invalid GitHub repo URL")
    return parts[0], parts[1]


def _fetch_file_content_github(
    repo_url: str, rel_path: str, commit: Optional[str] = None
) -> Optional[str]:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return None
    try:
        owner, repo = _parse_github_url(repo_url)
        base = f"https://api.github.com/repos/{owner}/{repo}/contents/{rel_path}"
        if commit:
            base += f"?ref={commit}"
        import requests

        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        }
        r = requests.get(base, headers=headers, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        if data.get("type") != "file" or "content" not in data:
            return None
        return b64decode(data["content"]).decode("utf-8", errors="ignore")
    except Exception:  # noqa: BLE001
        return None


__all__ = [
    "get_repo_map",
    "get_repo_map_summary",
    "lookup_repo_items",
    "build_repo_map",
]
