import os
import shutil
import hashlib
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Literal
import threading
import subprocess
import time

import pathspec
import chromadb
import ast

# --- Configuration ---
# ## MODIFIED ##: Corrected the model name to the official one.
EMBEDDING_MODEL = "models/embedding-001"
EMBEDDING_TASK_TYPE = Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]
EMBEDDING_BATCH_SIZE = 100
EMBEDDING_DIMENSIONS = 768

# --- Module-level State Management ---
_indexing_state: Dict[str, str] = {}
_indexing_start_times: Dict[str, float] = {}
_index_lock = threading.Lock()


def clear_indexing_state(repo_url: str = None):
    """Clear indexing state for a specific repo or all repos if none specified."""
    with _index_lock:
        if repo_url:
            _indexing_state.pop(repo_url, None)
            _indexing_start_times.pop(repo_url, None)
            print(f"Cleared indexing state for {repo_url}")
        else:
            _indexing_state.clear()
            _indexing_start_times.clear()
            print("Cleared all indexing states")


def _read_gitignore(repo_path: Path) -> Optional[pathspec.PathSpec]:
    gi = repo_path / ".gitignore"
    if not gi.exists():
        return None
    with gi.open("r", encoding="utf-8", errors="ignore") as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]
    return pathspec.PathSpec.from_lines("gitwildmatch", lines)


def _is_ignored(path: Path, repo_root: Path, spec: Optional[pathspec.PathSpec]) -> bool:
    if spec is None:
        return False
    rel_path = str(path.relative_to(repo_root)).replace("\\", "/")
    return spec.match_file(rel_path)


def _clone_repo(repo_url: str, dst: Path) -> Optional[str]:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.check_call(["git", "clone", "--depth", "1", repo_url, str(dst)])
        proc = subprocess.check_output(["git", "-C", str(dst), "rev-parse", "HEAD"])
        return proc.decode().strip()
    except Exception as e:
        print(f"Error cloning repo: {e}")
        return None


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _extract_python_units(path: Path) -> List[Dict]:
    """Extracts functions and classes from a Python file."""
    try:
        src = path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(src)
    except Exception:
        return []

    units = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno
            end = getattr(node, "end_lineno", start)
            code = "\n".join(src.splitlines()[start - 1 : end])
            units.append(
                {
                    "name": node.name,
                    "type": "class" if isinstance(node, ast.ClassDef) else "function",
                    "start_line": start,
                    "end_line": end,
                    "code": code,
                }
            )
    return units


def _fallback_file_unit(path: Path) -> Dict:
    """Creates a single 'unit' for an entire file."""
    try:
        src = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        src = ""
    return {
        "name": path.name,
        "type": "file",
        "start_line": 1,
        "end_line": src.count("\n") + 1,
        "code": src,
    }


def _chunk_text(text: str, max_chars: int = 2000, overlap: int = 200) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += max_chars - overlap
    return chunks


def get_embeddings(
    texts: List[str], task_type: EMBEDDING_TASK_TYPE
) -> List[List[float]]:
    """
    Gets embeddings for a list of texts using the Google Generative AI API.
    """
    import requests

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")

    all_embeddings = []

    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch_texts = texts[i : i + EMBEDDING_BATCH_SIZE]

        print(
            f"Processing embedding batch {i // EMBEDDING_BATCH_SIZE + 1}/{(len(texts) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE} ({len(batch_texts)} items)"
        )

        requests_payload = [
            {
                "model": EMBEDDING_MODEL,
                "content": {"parts": [{"text": text}]},
                "taskType": task_type,
            }
            for text in batch_texts
        ]

        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBEDDING_MODEL.split('/')[1]}:batchEmbedContents?key={api_key}"
            headers = {"Content-Type": "application/json"}
            data = {"requests": requests_payload}

            response = requests.post(url, json=data, headers=headers, timeout=60)
            response.raise_for_status()
            result = response.json()

            batch_embeddings = [item["values"] for item in result["embeddings"]]
            all_embeddings.extend(batch_embeddings)

            # ## MODIFIED ##: Increased sleep time to avoid hitting API rate limits.
            # A 4-second delay respects a 15 requests-per-minute limit.
            print("Waiting for 4 seconds to respect API rate limits...")
            time.sleep(4)

        except Exception as e:
            # ## MODIFIED ##: Removed zero-vector fallback.
            # Now, if an error occurs, we print it and re-raise the exception
            # to stop the indexing process immediately.
            print(f"Fatal error embedding batch: {e}. Stopping indexing.")
            raise e

    return all_embeddings


def index_repository(
    repo_url: str,
    persist_dir: str = "chroma_persist",
    collection_name: str = "repo_functions",
) -> Dict:
    """Clones a repo, extracts code units, embeds them, and persists to ChromaDB."""
    with _index_lock:
        if _indexing_state.get(repo_url) == "running":
            start_time = _indexing_start_times.get(repo_url)
            if start_time and time.time() - start_time > 600:
                print(f"Indexing for {repo_url} timed out, resetting state")
                _indexing_state.pop(repo_url, None)
                _indexing_start_times.pop(repo_url, None)
            else:
                return {"status": "already_running"}
        _indexing_state[repo_url] = "running"
        _indexing_start_times[repo_url] = time.time()

    tmpdir = Path(tempfile.mkdtemp(prefix="repo_index_"))
    try:
        commit = _clone_repo(repo_url, tmpdir)
        if not commit:
            return {"status": "clone_failed"}

        spec = _read_gitignore(tmpdir)
        client = chromadb.PersistentClient(path=persist_dir)
        collection = client.get_or_create_collection(collection_name)

        texts_to_embed, metadata_for_text = [], []

        print("Walking repository files...")
        for root, _, files in os.walk(tmpdir):
            rootp = Path(root)
            if ".git" in rootp.parts:
                continue
            for fn in files:
                fp = rootp / fn
                if _is_ignored(fp, tmpdir, spec):
                    continue

                units = (
                    _extract_python_units(fp)
                    if fp.suffix == ".py"
                    else [_fallback_file_unit(fp)]
                )
                if not units:
                    units = [_fallback_file_unit(fp)]

                for u in units:
                    chunks = _chunk_text(u["code"])
                    for idx, ch in enumerate(chunks):
                        doc = f"Unit: {u['name']}\nType: {u['type']}\nLines: {u['start_line']}-{u['end_line']}\n```\n{ch}\n```"
                        texts_to_embed.append(doc)
                        metadata_for_text.append(
                            {
                                "repo_url": repo_url,
                                "commit": commit,
                                "file_path": str(fp.relative_to(tmpdir)).replace(
                                    "\\", "/"
                                ),
                                "unit_name": u["name"],
                                "unit_type": u["type"],
                                "start_line": u["start_line"],
                                "end_line": u["end_line"],
                                "chunk_index": idx,
                            }
                        )

        if not texts_to_embed:
            return {"status": "ok", "commit": commit, "indexed_chunks": 0}

        print(f"Starting to embed {len(texts_to_embed)} text chunks...")
        embeddings = get_embeddings(texts_to_embed, "RETRIEVAL_DOCUMENT")

        ids = [
            f"{m['repo_url']}:{m['file_path']}:{m['unit_name']}:{m['start_line']}:{m['chunk_index']}"
            for m in metadata_for_text
        ]

        collection.add(
            ids=ids,
            documents=texts_to_embed,
            metadatas=metadata_for_text,
            embeddings=embeddings,
        )

        return {"status": "ok", "commit": commit, "indexed_chunks": len(ids)}

    except Exception as e:
        print(f"An unexpected error occurred during indexing: {e}")
        # Mark the state as 'failed' so it can be retried without clearing
        with _index_lock:
            _indexing_state[repo_url] = "failed"
        return {"status": "error", "message": str(e)}

    finally:
        with _index_lock:
            # Only mark as 'done' if it hasn't failed
            if _indexing_state.get(repo_url) == "running":
                _indexing_state[repo_url] = "done"
            _indexing_start_times.pop(repo_url, None)
        shutil.rmtree(tmpdir, ignore_errors=True)


def query_repository(
    question: str,
    repo_url: str,
    persist_dir: str = "chroma_persist",
    collection_name: str = "repo_functions",
    top_k: int = 5,
) -> List[Dict]:
    """Queries the vector store for relevant code chunks."""
    try:
        client = chromadb.PersistentClient(path=persist_dir)
        collection = client.get_collection(collection_name)
    except Exception as e:
        print(f"Could not get collection '{collection_name}': {e}")
        return []

    try:
        query_embedding = get_embeddings([question], "RETRIEVAL_QUERY")[0]
    except Exception as e:
        print(f"Failed to embed query: {e}")
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"repo_url": repo_url},
    )

    if not results or not results.get("documents"):
        return []

    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    return [{"document": doc, "metadata": meta} for doc, meta in zip(docs, metadatas)]
