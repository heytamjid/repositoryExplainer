import os
import shutil
import hashlib
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Literal, Tuple
import threading
import subprocess
import time

import pathspec
import chromadb
import ast

# --- Configuration ---
# Embedding Configuration (unit-level and file-level)
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

# File-level dual indexing configuration
FILE_LEVEL_MAX_CHARS = int(os.environ.get("REPO_EXPLAINER_FILE_MAX_CHARS", 8000))
FILE_LEVEL_WINDOW_TARGET = int(
    os.environ.get("REPO_EXPLAINER_FILE_WINDOW_CHARS", 4000)
)  # Target size for windowed file chunks
FILE_LEVEL_OVERLAP_FUNCTIONS = int(
    os.environ.get("REPO_EXPLAINER_FILE_OVERLAP_FUNCS", 1)
)  # Number of functions to overlap between large file windows
MAX_TOTAL_GROUPED_CONTEXT_CHARS = int(
    os.environ.get("REPO_EXPLAINER_MAX_TOTAL_GROUPED", 60000)
)

# Retrieval tuning
HIGH_LEVEL_TOP_K_FILES = int(os.environ.get("REPO_EXPLAINER_TOP_K_FILES", 8))
HIGH_LEVEL_UNIT_PER_FILE = int(os.environ.get("REPO_EXPLAINER_UNITS_PER_FILE", 6))

# Classifier model (separate from embedding model so we don't interfere with primary embedding pipeline)
CLASSIFIER_MODEL_NAME = os.environ.get(
    "REPO_EXPLAINER_CLASSIFIER_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# --- Module-level State Management ---
STATUS_FILE_NAME = "indexing_status.json"
_index_lock = threading.Lock()

# --- File exclusion / denylist configuration ---
# Extensions, filenames and directory names we never want to index
DENY_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".ico",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".mp4",
    ".mp3",
    ".wav",
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".exe",
    ".dll",
    ".so",
    ".pyc",
    ".class",
    ".jar",
    ".pdf",
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
}

# Specific file names to always skip
DENY_FILENAMES = {"db.sqlite3", "thumbs.db"}

# Directory names that are commonly noisy
DENY_DIR_NAMES = {
    "node_modules",
    "venv",
    ".venv",
    "env",
    "build",
    "dist",
    "target",
    "out",
    "public",
    "static",
    "media",
    "__pycache__",
    "coverage",
    "vendor",
    "bower_components",
    ".next",
    ".nuxt",
    "django_bundles",
    "django_bundle",
}

# Max file size (bytes) to consider for indexing (default 200 KB)
MAX_FILE_SIZE_BYTES = int(os.environ.get("REPO_EXPLAINER_MAX_FILE_SIZE", 200 * 1024))


# Small helper to heuristically detect binary files by sampling the beginning bytes
def _is_probably_binary(path: Path, sample_size: int = 4096) -> bool:
    try:
        with path.open("rb") as fh:
            chunk = fh.read(sample_size)
            if not chunk:
                return False
            # Null byte is a strong signal of binary
            if b"\x00" in chunk:
                return True
            # Heuristic: proportion of non-text bytes
            text_chars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)))
            nontext = sum(1 for b in chunk if b not in text_chars)
            return (nontext / max(1, len(chunk))) > 0.30
    except Exception:
        # If we cannot read the file, be conservative and skip it
        return True


def _should_skip_file(
    path: Path, repo_root: Path, gitignore_spec: Optional[pathspec.PathSpec]
) -> bool:
    """
    Return True if the file should be skipped from indexing.
    Checks: .gitignore, denylist by extension/filename/dir, file size, and a binary heuristic.
    """
    try:
        if not path.exists() or path.is_dir():
            return True

        rel = str(path.relative_to(repo_root)).replace("\\", "/")

        # Respect .gitignore
        if gitignore_spec and gitignore_spec.match_file(rel):
            return True

        name_lower = path.name.lower()
        if name_lower in DENY_FILENAMES:
            return True

        # Skip files inside noisy dirs (any path component matches)
        for p in path.parts:
            if p.lower() in DENY_DIR_NAMES:
                return True

        # Skip by extension
        ext = path.suffix.lower()
        if ext in DENY_EXTENSIONS:
            return True

        # Skip very large files
        try:
            size = path.stat().st_size
            if size > MAX_FILE_SIZE_BYTES:
                return True
        except Exception:
            return True

        # Heuristic: skip binary-like files
        if _is_probably_binary(path):
            return True

        return False
    except Exception:
        return True


# Global variables for local embedding model (lazy loading)
_local_model = None
_local_tokenizer = None
_classifier_model = None


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


def _load_classifier_model():
    """Lazy load the light-weight sentence-transformers classifier model."""
    global _classifier_model
    if _classifier_model is not None:
        return _classifier_model
    try:
        from sentence_transformers import SentenceTransformer

        print(f"Loading classifier model: {CLASSIFIER_MODEL_NAME}")
        _classifier_model = SentenceTransformer(
            CLASSIFIER_MODEL_NAME, device="cpu", trust_remote_code=True
        )
        return _classifier_model
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for query classification. Install it with: pip install sentence-transformers"
        )
    except Exception as e:
        raise Exception(f"Failed to load classifier model: {e}")


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


def _group_units_into_file_chunks(
    path: Path, units: List[Dict]
) -> List[Tuple[str, int, int, int]]:
    """
    Create file-level chunks (dual indexing) without breaking function/class boundaries.
    Returns list of tuples: (chunk_text, chunk_index, start_line, end_line)
    """
    try:
        full_src = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    if not units:
        # Fall back to simple chunking of entire file text
        chunks = []
        texts = _chunk_text(full_src, max_chars=FILE_LEVEL_WINDOW_TARGET, overlap=200)
        for i, t in enumerate(texts):
            chunks.append((t, i, 1, len(full_src.splitlines())))
        return chunks

    # Single chunk if file small enough
    if len(full_src) <= FILE_LEVEL_MAX_CHARS:
        return [(full_src, 0, 1, len(full_src.splitlines()))]

    # Otherwise group sequential units into chunks up to FILE_LEVEL_WINDOW_TARGET chars
    chunks: List[Tuple[str, int, int, int]] = []
    current_parts: List[str] = []
    current_start = units[0]["start_line"]
    current_chars = 0
    chunk_index = 0

    def flush(end_line: int):
        nonlocal current_parts, current_chars, current_start, chunk_index
        if not current_parts:
            return
        text = "\n\n".join(current_parts)
        chunks.append((text, chunk_index, current_start, end_line))
        chunk_index += 1
        current_parts = []
        current_chars = 0
        current_start = end_line + 1

    for i, u in enumerate(units):
        block = u["code"].rstrip()
        sep = f"\n\n# ---- {u['type'].upper()} {u['name']} (lines {u['start_line']}-{u['end_line']}) ----\n"
        block_text = sep + block
        if current_chars + len(block_text) > FILE_LEVEL_WINDOW_TARGET and current_parts:
            # Overlap last few functions: add tail from previous before flush (handled implicitly via last_parts)
            flush(units[i - 1]["end_line"])
            # Start new chunk, optionally overlap previous N functions
            if FILE_LEVEL_OVERLAP_FUNCTIONS > 0 and i > 0:
                overlap_units = units[max(0, i - FILE_LEVEL_OVERLAP_FUNCTIONS) : i]
                for ou in overlap_units:
                    osep = f"\n\n# ---- {ou['type'].upper()} {ou['name']} (lines {ou['start_line']}-{ou['end_line']}) [overlap] ----\n"
                    current_parts.append(osep + ou["code"].rstrip())
                    current_chars += len(osep) + len(ou["code"].rstrip())
                    current_start = min(current_start, ou["start_line"])
        current_parts.append(block_text)
        current_chars += len(block_text)
    if current_parts:
        flush(units[-1]["end_line"])
    return chunks


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
        try:
            from myapp.views import (
                get_summary,
            )  # Local import to avoid circular dependency

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
                            "granularity": "summary",
                            "start_line": 1,
                            "end_line": 1,
                            "chunk_index": 0,
                            "embedding_mode": mode,
                        }
                    )
        except Exception as e:
            print(f"Skipping summary embedding due to error: {e}")

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
                    # Combined skip logic: respect .gitignore, denylist, binary/size heuristics
                    if _should_skip_file(fp, tmpdir, spec):
                        log_f.write(f"SKIPPED: {fp.relative_to(tmpdir)}\n")
                        continue

                    is_python = fp.suffix == ".py"
                    units = _extract_python_units(fp) if is_python else []
                    if not units:
                        # Non-Python or no units, treat whole file as single unit for unit-level fallback
                        fu = _fallback_file_unit(fp)
                        units = [fu]
                    rel_path_str = str(fp.relative_to(tmpdir)).replace("\\", "/")

                    # --- File-level chunks (granularity=file) ---
                    file_chunks = _group_units_into_file_chunks(
                        fp, units if is_python else []
                    )
                    for fch_text, fch_idx, fch_start, fch_end in file_chunks:
                        doc = (
                            f"File Chunk: {rel_path_str}\nChunk Index: {fch_idx}\nLines: {fch_start}-{fch_end}\n"  # header
                            f"```\n{fch_text}\n```"
                        )
                        texts_to_embed.append(doc)
                        metadata_for_text.append(
                            {
                                "repo_url": repo_url,
                                "commit": commit,
                                "file_path": rel_path_str,
                                "unit_name": f"file_chunk_{fch_idx}",
                                "unit_type": "file",
                                "granularity": "file",
                                "start_line": fch_start,
                                "end_line": fch_end,
                                "chunk_index": fch_idx,
                                "embedding_mode": mode,
                            }
                        )
                        log_f.write(f"--- FILE-LEVEL CHUNK ---\n{doc}\n\n")

                    # --- Unit-level chunks (granularity=unit) ---
                    for u in units:
                        unit_chunks = _chunk_text(u["code"])
                        for idx, ch in enumerate(unit_chunks):
                            doc = (
                                f"Unit: {u['name']}\nType: {u['type']}\nLines: {u['start_line']}-{u['end_line']}\n"  # header
                                f"```\n{ch}\n```"
                            )
                            texts_to_embed.append(doc)
                            metadata_for_text.append(
                                {
                                    "repo_url": repo_url,
                                    "commit": commit,
                                    "file_path": rel_path_str,
                                    "unit_name": u["name"],
                                    "unit_type": u["type"],
                                    "granularity": "unit",
                                    "start_line": u["start_line"],
                                    "end_line": u["end_line"],
                                    "chunk_index": idx,
                                    "embedding_mode": mode,
                                }
                            )
                            log_f.write(f"--- UNIT-LEVEL DOCUMENT ---\n{doc}\n\n")

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

        ids = []
        for m in metadata_for_text:
            base = f"{m['repo_url']}:{m['file_path']}:{m['unit_name']}:{m['start_line']}:{m['chunk_index']}"
            if m.get("granularity") == "file":
                base = "FILE::" + base
            elif m.get("granularity") == "unit":
                base = "UNIT::" + base
            else:
                base = "OTHER::" + base
            ids.append(base)

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
        "deny_extensions": sorted(list(DENY_EXTENSIONS)),
        "deny_filenames": sorted(list(DENY_FILENAMES)),
        "deny_dir_names": sorted(list(DENY_DIR_NAMES)),
        "max_file_size_bytes": MAX_FILE_SIZE_BYTES,
        "dual_indexing": {
            "file_level_max_chars": FILE_LEVEL_MAX_CHARS,
            "file_level_window_target": FILE_LEVEL_WINDOW_TARGET,
            "file_level_overlap_functions": FILE_LEVEL_OVERLAP_FUNCTIONS,
        },
        "retrieval": {
            "high_level_top_k_files": HIGH_LEVEL_TOP_K_FILES,
            "high_level_unit_per_file": HIGH_LEVEL_UNIT_PER_FILE,
        },
        "classifier_model": CLASSIFIER_MODEL_NAME,
    }


# =============================
# Advanced Retrieval Pipeline
# =============================

LOW_LEVEL_LABEL = "low_level"
HIGH_LEVEL_LABEL = "high_level"

_classification_lock = threading.Lock()


def classify_question(question: str) -> str:
    """Classify a user question as low-level or high-level using semantic similarity to prototype phrases.

    Falls back to heuristic keywords if model unavailable.
    """
    prototypes = {
        LOW_LEVEL_LABEL: [
            "how is implemented",
            "implementation detail",
            "function",
            "class",
            "parameter",
            "algorithm",
            "internal logic",
            "source code of",
            "definition of",
        ],
        HIGH_LEVEL_LABEL: [
            "architecture",
            "overall design",
            "data flow",
            "component interaction",
            "module overview",
            "high level",
            "system overview",
            "lifecycle",
            "process flow",
        ],
    }
    try:
        model = _load_classifier_model()
        with _classification_lock:
            q_emb = model.encode([question], normalize_embeddings=True)[0]
            scores = {}
            import numpy as np

            for label, phrases in prototypes.items():
                emb = model.encode(phrases, normalize_embeddings=True)
                cat_vec = emb.mean(axis=0)
                scores[label] = float(
                    np.dot(q_emb, cat_vec) / (np.linalg.norm(cat_vec) + 1e-8)
                )
            # Decide winner
            winner = max(scores.items(), key=lambda kv: kv[1])[0]
            # Light threshold: if difference small and question contains architecture keywords => high-level
            diff = abs(scores[HIGH_LEVEL_LABEL] - scores[LOW_LEVEL_LABEL])
            if diff < 0.03 and any(
                k in question.lower()
                for k in ["architecture", "overview", "data flow", "flow", "design"]
            ):
                winner = HIGH_LEVEL_LABEL
            print(
                f"[PIPELINE] Classification: '{question}' -> {winner} (scores: low={scores.get(LOW_LEVEL_LABEL):.4f}, high={scores.get(HIGH_LEVEL_LABEL):.4f})"
            )
            return winner
    except Exception as e:
        print(f"Classifier fallback due to error: {e}")
        ql = question.lower()
        if any(
            k in ql
            for k in [
                "architecture",
                "overview",
                "data flow",
                "flow",
                "design",
                "interaction",
                "components",
            ]
        ):
            return HIGH_LEVEL_LABEL
        return LOW_LEVEL_LABEL


def _query_chroma_raw(
    collection, query_embedding, where: Dict, top_k: int
) -> List[Dict]:
    """Internal helper to query Chroma with support for multi-key equality filters.

    Some Chroma versions require a single operator in `where`. When multiple
    equality constraints are needed we synthesize an $and list.
    """

    def _normalize_where(w: Dict) -> Dict:
        if not w:
            return {}
        if len(w) == 1:
            return w
        # Convert to $and list of simple equality objects
        return {"$and": [{k: v} for k, v in w.items()]}

    where_clause = _normalize_where(where)
    try:
        print(f"[PIPELINE] Querying Chroma where={where_clause} top_k={top_k}")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause,
        )
        if not results or not results.get("documents"):
            print("[PIPELINE] Query returned 0 documents")
            return []
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        return [
            {"document": d, "metadata": m}
            for d, m in zip(docs, metas)
            if m.get("granularity") != "summary"  # exclude summaries for now
        ]
    except Exception as e:
        print(f"Raw query error: {e}")
        return []


def query_repository_advanced(
    question: str,
    repo_url: str,
    persist_dir: str = "chroma_persist",
    collection_name: str = "repo_functions",
    embedding_mode: Optional[str] = None,
    top_k_units: int = 25,
) -> Dict:
    """Advanced retrieval implementing classification + dual-stage retrieval.

    Returns dict with keys: classification, documents (list like query_repository), debug(optional)
    """
    print("[PIPELINE] Starting advanced retrieval pipeline")
    classification = classify_question(question)
    mode = embedding_mode or EMBEDDING_MODE
    try:
        import chromadb  # local import

        client = chromadb.PersistentClient(path=persist_dir)
        collection = client.get_collection(collection_name)
    except Exception as e:
        print(f"Could not open collection for advanced query: {e}")
        return {"classification": classification, "documents": []}

    # Embed query once
    try:
        query_embedding = get_embeddings([question], "RETRIEVAL_QUERY", mode)[0]
    except Exception as e:
        print(f"Failed embedding query in advanced retrieval: {e}")
        return {"classification": classification, "documents": []}

    debug = {"classification": classification}

    if classification == LOW_LEVEL_LABEL:
        print("[PIPELINE] Path: LOW_LEVEL -> single-stage unit retrieval")
        unit_docs = _query_chroma_raw(
            collection,
            query_embedding,
            {"repo_url": repo_url, "granularity": "unit"},
            top_k_units,
        )
        print(f"[PIPELINE] Retrieved {len(unit_docs)} unit docs")
        grouped = _group_unit_docs(unit_docs)
        print(f"[PIPELINE] Grouped into {len(grouped)} grouped docs (low-level)")
        return {"classification": classification, "documents": grouped, "debug": debug}

    # High-level: 2 stage
    print("[PIPELINE] Path: HIGH_LEVEL -> stage1 file-level retrieval")
    file_docs = _query_chroma_raw(
        collection,
        query_embedding,
        {"repo_url": repo_url, "granularity": "file"},
        HIGH_LEVEL_TOP_K_FILES,
    )
    if not file_docs:
        print(
            "[PIPELINE] Stage1 returned 0 file docs; falling back to LOW_LEVEL strategy"
        )
        unit_docs = _query_chroma_raw(
            collection,
            query_embedding,
            {"repo_url": repo_url, "granularity": "unit"},
            top_k_units,
        )
        print(f"[PIPELINE] Fallback retrieved {len(unit_docs)} unit docs")
        grouped = _group_unit_docs(unit_docs)
        print(f"[PIPELINE] Grouped into {len(grouped)} fallback grouped docs")
        return {"classification": classification, "documents": grouped, "debug": debug}

    selected_files = [d["metadata"]["file_path"] for d in file_docs]
    debug["selected_files"] = selected_files
    print(f"[PIPELINE] Stage1 selected {len(selected_files)} files: {selected_files}")

    # Retrieve unit-level inside each selected file
    per_file_units: List[Dict] = []
    for fp in selected_files:
        # smaller per-file query to get most relevant units from that file
        units_for_file = _query_chroma_raw(
            collection,
            query_embedding,
            {"repo_url": repo_url, "granularity": "unit", "file_path": fp},
            HIGH_LEVEL_UNIT_PER_FILE,
        )
        per_file_units.extend(units_for_file)
        print(
            f"[PIPELINE] Retrieved {len(units_for_file)} unit docs for file {fp} (cumulative {len(per_file_units)})"
        )

    grouped = _group_unit_docs(per_file_units, file_docs)
    print(
        f"[PIPELINE] Grouped high-level results into {len(grouped)} grouped docs (total unit docs considered {len(per_file_units)})"
    )
    return {"classification": classification, "documents": grouped, "debug": debug}


def _group_unit_docs(
    unit_docs: List[Dict], file_docs: Optional[List[Dict]] = None
) -> List[Dict]:
    """Group multiple unit-level docs under their file path and merge content.

    If file_docs provided (from high-level stage1) we prepend the first file-level chunk for broader context.
    """
    # Map file_path -> list of units
    files: Dict[str, Dict] = {}
    # Index first file-level doc for each file
    file_level_map = {}
    if file_docs:
        for fd in file_docs:
            fp = fd["metadata"].get("file_path")
            if fp not in file_level_map:
                file_level_map[fp] = fd

    for doc in unit_docs:
        meta = doc["metadata"]
        fp = meta.get("file_path")
        if fp not in files:
            files[fp] = {"units": [], "file_meta": meta}
        files[fp]["units"].append(doc)

    merged_docs: List[Dict] = []
    total_chars = 0
    for fp, bundle in files.items():
        # Sort units by start_line
        units_sorted = sorted(
            bundle["units"], key=lambda d: d["metadata"].get("start_line", 0)
        )
        # Build content
        content_parts = []
        if fp in file_level_map:
            content_parts.append(
                f"# FILE CONTEXT: {fp}\n" + file_level_map[fp]["document"] + "\n\n"
            )
        content_parts.append(f"# SELECTED FUNCTIONS / CLASSES FROM {fp}\n")
        min_line = None
        max_line = None
        for ud in units_sorted:
            m = ud["metadata"]
            min_line = (
                m.get("start_line")
                if min_line is None
                else min(min_line, m.get("start_line"))
            )
            max_line = (
                m.get("end_line")
                if max_line is None
                else max(max_line, m.get("end_line"))
            )
            content_parts.append(ud["document"])
        merged_text = "\n\n".join(content_parts)
        if total_chars + len(merged_text) > MAX_TOTAL_GROUPED_CONTEXT_CHARS:
            break
        total_chars += len(merged_text)
        merged_docs.append(
            {
                "document": merged_text,
                "metadata": {
                    "file_path": fp,
                    "start_line": min_line or 1,
                    "end_line": max_line or 1,
                    "unit_type": "grouped",
                    "granularity": "group",
                },
            }
        )
    print(
        f"[PIPELINE] Grouping complete: {len(merged_docs)} grouped docs; truncated due to size limit? {'YES' if len(merged_docs) < len(files) else 'NO'}"
    )
    # Fallback: if grouping produced nothing due to size limit, return raw units
    if not merged_docs:
        return unit_docs
    return merged_docs
