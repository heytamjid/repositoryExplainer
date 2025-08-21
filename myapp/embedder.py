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
# Embedding Configuration
EMBEDDING_MODE = os.environ.get("EMBEDDING_MODE", "local")  # "local" or "remote"
LOCAL_EMBEDDING_MODEL = (
    "nomic-ai/nomic-embed-text-v1.5"  # Nomic model for local embedding
)
REMOTE_EMBEDDING_MODEL = "models/embedding-001"  # Google Gemini model
EMBEDDING_TASK_TYPE = Literal["RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY"]
EMBEDDING_BATCH_SIZE = 100
EMBEDDING_DIMENSIONS = 768  # This may vary based on the model
LOCAL_EMBEDDING_DEVICE = os.environ.get(
    "LOCAL_EMBEDDING_DEVICE", "cpu"
)  # "cpu" or "cuda"

# --- Module-level State Management ---
STATUS_FILE_NAME = "indexing_status.json"
_index_lock = threading.Lock()

# Global variables for local embedding model (lazy loading)
_local_model = None
_local_tokenizer = None


def _get_status_path(persist_dir: str) -> Path:
    """Gets the path to the persistent status file."""
    return Path(persist_dir) / STATUS_FILE_NAME


def _read_status(status_path: Path) -> Dict:
    """Reads the indexing status from the persistent file."""
    if not status_path.exists():
        return {}
    try:
        with status_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def _write_status(status_path: Path, data: Dict):
    """Writes the indexing status to the persistent file."""
    with _index_lock:
        status_path.parent.mkdir(exist_ok=True)
        with status_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def clear_indexing_state(repo_url: str = None, persist_dir: str = "chroma_persist"):
    """Clear indexing state for a specific repo or all repos if none specified."""
    status_path = _get_status_path(persist_dir)
    statuses = _read_status(status_path)
    if repo_url:
        if repo_url in statuses:
            del statuses[repo_url]
            print(f"Cleared indexing state for {repo_url}")
    else:
        statuses.clear()
        print("Cleared all indexing states")
    _write_status(status_path, statuses)


def get_indexing_status(repo_url: str, persist_dir: str = "chroma_persist") -> Dict:
    """Gets the indexing status for a specific repository."""
    status_path = _get_status_path(persist_dir)
    statuses = _read_status(status_path)
    return statuses.get(repo_url, {"status": "not_indexed"})


def _load_local_embedding_model():
    """Lazy loading of the local embedding model."""
    global _local_model, _local_tokenizer

    if _local_model is not None:
        return _local_model, _local_tokenizer

    try:
        from sentence_transformers import SentenceTransformer

        print(f"Loading local embedding model: {LOCAL_EMBEDDING_MODEL}")
        _local_model = SentenceTransformer(
            LOCAL_EMBEDDING_MODEL,
            device=LOCAL_EMBEDDING_DEVICE,
            trust_remote_code=True,
        )
        print(f"Local embedding model loaded successfully on {LOCAL_EMBEDDING_DEVICE}")
        return _local_model, None
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for local embeddings. "
            "Install it with: pip install sentence-transformers"
        )
    except Exception as e:
        raise Exception(f"Failed to load local embedding model: {e}")


def get_embeddings_local(texts: List[str]) -> List[List[float]]:
    """
    Gets embeddings for a list of texts using the local Nomic model.
    """
    model, _ = _load_local_embedding_model()

    print(f"Processing {len(texts)} texts with local embedding model...")

    # Process in batches for memory efficiency
    all_embeddings = []
    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch_texts = texts[i : i + EMBEDDING_BATCH_SIZE]
        print(
            f"Processing local embedding batch {i // EMBEDDING_BATCH_SIZE + 1}/{(len(texts) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE} ({len(batch_texts)} items)"
        )

        try:
            # Encode the batch
            batch_embeddings = model.encode(
                batch_texts,
                convert_to_tensor=False,
                show_progress_bar=False,
                normalize_embeddings=True,  # Normalize for better similarity search
            )
            all_embeddings.extend(batch_embeddings.tolist())

            # Small delay to prevent overheating
            if len(batch_texts) == EMBEDDING_BATCH_SIZE:
                time.sleep(0.1)

        except Exception as e:
            print(f"Fatal error embedding batch locally: {e}. Stopping indexing.")
            raise e

    return all_embeddings


def get_embeddings_remote(
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
            f"Processing remote embedding batch {i // EMBEDDING_BATCH_SIZE + 1}/{(len(texts) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE} ({len(batch_texts)} items)"
        )

        requests_payload = [
            {
                "model": REMOTE_EMBEDDING_MODEL,
                "content": {"parts": [{"text": text}]},
                "taskType": task_type,
            }
            for text in batch_texts
        ]

        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{REMOTE_EMBEDDING_MODEL.split('/')[1]}:batchEmbedContents?key={api_key}"
            headers = {"Content-Type": "application/json"}
            data = {"requests": requests_payload}

            response = requests.post(url, json=data, headers=headers, timeout=60)
            response.raise_for_status()
            result = response.json()

            batch_embeddings = [item["values"] for item in result["embeddings"]]
            all_embeddings.extend(batch_embeddings)

            # Increased sleep time to avoid hitting API rate limits.
            # A 4-second delay respects a 15 requests-per-minute limit.
            print("Waiting for 4 seconds to respect API rate limits...")
            time.sleep(4)

        except Exception as e:
            print(f"Fatal error embedding batch: {e}. Stopping indexing.")
            raise e

    return all_embeddings


def get_embeddings(
    texts: List[str],
    task_type: EMBEDDING_TASK_TYPE = "RETRIEVAL_DOCUMENT",
    mode: Optional[str] = None,
) -> List[List[float]]:
    """
    Gets embeddings for a list of texts using either local or remote model.

    Args:
        texts: List of texts to embed
        task_type: Task type for remote embeddings (ignored for local)
        mode: Override the global EMBEDDING_MODE ("local" or "remote")
    """
    embedding_mode = mode or EMBEDDING_MODE

    print(f"Using {embedding_mode} embeddings for {len(texts)} texts")

    if embedding_mode == "local":
        return get_embeddings_local(texts)
    elif embedding_mode == "remote":
        return get_embeddings_remote(texts, task_type)
    else:
        raise ValueError(
            f"Invalid embedding mode: {embedding_mode}. Use 'local' or 'remote'"
        )


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


def _write_chunks_manifest(
    persist_dir: str,
    repo_url: str,
    texts: List[str],
    metadatas: List[Dict],
    filename: Optional[str] = None,
) -> Path:
    """
    Write a newline-delimited JSON manifest listing all chunks that will be embedded.
    Each line contains metadata plus the chunk text and a content hash for inspection.
    Returns the Path to the created manifest file.
    """
    dest = Path(persist_dir)
    dest.mkdir(parents=True, exist_ok=True)

    safe_repo = repo_url.replace("://", "_").replace("/", "_").replace(":", "_")
    manifest_name = filename or f"{safe_repo}_embedding_chunks.jsonl"
    manifest_path = dest / manifest_name

    with manifest_path.open("w", encoding="utf-8") as fh:
        for text, meta in zip(texts, metadatas):
            rec = {
                "file_path": meta.get("file_path"),
                "unit_name": meta.get("unit_name"),
                "unit_type": meta.get("unit_type"),
                "start_line": meta.get("start_line"),
                "end_line": meta.get("end_line"),
                "chunk_index": meta.get("chunk_index"),
                "char_len": len(text) if text is not None else 0,
                "hash": _hash_text(text or ""),
                "text": text,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return manifest_path


def index_repository(
    repo_url: str,
    persist_dir: str = "chroma_persist",
    collection_name: str = "repo_functions",
    embedding_mode: Optional[str] = None,
) -> Dict:
    """
    Clones a repo, extracts code units, embeds them, and persists to ChromaDB.

    Args:
        repo_url: URL of the repository to index
        persist_dir: Directory to persist ChromaDB
        collection_name: Name of the ChromaDB collection
        embedding_mode: Override global embedding mode ("local" or "remote")
    """
    # Use the specified mode or fall back to global setting
    mode = embedding_mode or EMBEDDING_MODE
    print(f"Indexing repository with {mode} embeddings")

    status_path = _get_status_path(persist_dir)
    statuses = _read_status(status_path)
    repo_status = statuses.get(repo_url, {})

    if repo_status.get("status") == "done":
        print(f"Repository {repo_url} already indexed. Skipping.")
        return {"status": "already_indexed", "commit": repo_status.get("commit")}

    if repo_status.get("status") == "running":
        start_time = repo_status.get("start_time")
        if start_time and time.time() - start_time > 600:  # 10-minute timeout
            print(f"Indexing for {repo_url} timed out. Resetting.")
        else:
            return {"status": "already_running"}

    # Set status to running
    statuses[repo_url] = {
        "status": "running",
        "start_time": time.time(),
        "embedding_mode": mode,
    }
    _write_status(status_path, statuses)

    # Clean up previous failed attempts for this repo
    try:
        client = chromadb.PersistentClient(path=persist_dir)
        collection = client.get_or_create_collection(collection_name)
        collection.delete(where={"repo_url": repo_url})
        print(f"Cleared previous entries for {repo_url} before re-indexing.")
    except Exception as e:
        print(f"Could not clear old entries for {repo_url}: {e}")

    tmpdir = Path(tempfile.mkdtemp(prefix="repo_index_"))
    try:
        commit = _clone_repo(repo_url, tmpdir)
        if not commit:
            raise Exception("Failed to clone repository.")

        spec = _read_gitignore(tmpdir)

        texts_to_embed, metadata_for_text = [], []

        # --- Embed Summary ---
        from myapp.views import get_summary  # Local import to avoid circular dependency

        summary_data = get_summary(repo_url)
        if summary_data and summary_data.get("summary"):
            print("Embedding repository summary...")
            summary = summary_data["summary"]
            for section_id, content in summary.items():
                doc = f"Summary of Section: {section_id}\n\n{content}"
                texts_to_embed.append(doc)
                metadata_for_text.append(
                    {
                        "repo_url": repo_url,
                        "commit": summary_data.get("commit"),
                        "file_path": "repository-summary.md",
                        "unit_name": f"Summary: {section_id}",
                        "unit_type": "summary",
                        "start_line": 1,
                        "end_line": 1,
                        "chunk_index": 0,
                        "embedding_mode": mode,
                    }
                )

        # Create a temporary file for logging what's being embedded
        log_file_path = tmpdir / "embedding_log.txt"
        with log_file_path.open("w", encoding="utf-8") as log_f:
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
                                    "embedding_mode": mode,
                                }
                            )
                            log_f.write(f"--- Document ---\n{doc}\n\n")

        if not texts_to_embed:
            statuses[repo_url] = {
                "status": "done",
                "commit": commit,
                "indexed_chunks": 0,
                "embedding_mode": mode,
            }
            _write_status(status_path, statuses)
            return {"status": "ok", "commit": commit, "indexed_chunks": 0}

        print(
            f"Starting to embed {len(texts_to_embed)} text chunks using {mode} embeddings..."
        )

        # Write a manifest of all chunks that will be embedded for inspection/debugging
        try:
            manifest_path = _write_chunks_manifest(
                persist_dir, repo_url, texts_to_embed, metadata_for_text
            )
            print(f"Wrote embedding manifest to: {manifest_path}")
        except Exception as e:
            print(f"Warning: failed to write chunks manifest: {e}")

        embeddings = get_embeddings(texts_to_embed, "RETRIEVAL_DOCUMENT", mode)

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

        print(f"Embedding log saved to: {log_file_path}")

        statuses[repo_url] = {
            "status": "done",
            "commit": commit,
            "indexed_chunks": len(ids),
            "embedding_mode": mode,
        }
        _write_status(status_path, statuses)

        return {"status": "ok", "commit": commit, "indexed_chunks": len(ids)}

    except Exception as e:
        print(f"An unexpected error occurred during indexing: {e}")
        statuses[repo_url] = {"status": "failed", "error": str(e)}
        _write_status(status_path, statuses)
        return {"status": "error", "message": str(e)}

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def query_repository(
    question: str,
    repo_url: str,
    persist_dir: str = "chroma_persist",
    collection_name: str = "repo_functions",
    top_k: int = 25,
    embedding_mode: Optional[str] = None,
) -> List[Dict]:
    """
    Queries the vector store for relevant code chunks.

    Args:
        question: The question to search for
        repo_url: URL of the repository to query
        persist_dir: Directory where ChromaDB is persisted
        collection_name: Name of the ChromaDB collection
        top_k: Number of results to return
        embedding_mode: Override global embedding mode ("local" or "remote")
    """
    # Use the specified mode or fall back to global setting
    mode = embedding_mode or EMBEDDING_MODE

    try:
        client = chromadb.PersistentClient(path=persist_dir)
        collection = client.get_collection(collection_name)
    except Exception as e:
        print(f"Could not get collection '{collection_name}': {e}")
        return []

    try:
        query_embedding = get_embeddings([question], "RETRIEVAL_QUERY", mode)[0]
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


def get_embedding_info() -> Dict:
    """
    Returns information about the current embedding configuration.
    """
    return {
        "mode": EMBEDDING_MODE,
        "local_model": LOCAL_EMBEDDING_MODEL,
        "remote_model": REMOTE_EMBEDDING_MODEL,
        "device": LOCAL_EMBEDDING_DEVICE,
        "batch_size": EMBEDDING_BATCH_SIZE,
        "dimensions": EMBEDDING_DIMENSIONS,
    }
