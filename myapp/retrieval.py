"""Retrieval pipeline & classification separated from `embedder` to reduce file size.

Exports:
- classify_question
- query_repository_advanced
- _group_unit_docs (internal)

Backwards compatibility: functions are imported into `embedder` namespace.
"""

from __future__ import annotations
from typing import List, Dict, Optional
import threading
import time

from . import embedder as _embedder  # circular safe for runtime attribute access

LOW_LEVEL_LABEL = "low_level"
HIGH_LEVEL_LABEL = "high_level"
_classification_lock = threading.Lock()

# Re-export config values from embedder for cohesion
MAX_TOTAL_GROUPED_CONTEXT_CHARS = getattr(
    _embedder, "MAX_TOTAL_GROUPED_CONTEXT_CHARS", 60000
)
HIGH_LEVEL_TOP_K_FILES = getattr(_embedder, "HIGH_LEVEL_TOP_K_FILES", 8)
HIGH_LEVEL_UNIT_PER_FILE = getattr(_embedder, "HIGH_LEVEL_UNIT_PER_FILE", 6)
EMBEDDING_MODE = getattr(_embedder, "EMBEDDING_MODE", "remote")


def classify_question(question: str) -> str:
    """Classify a user question as low-level or high-level.

    Attempts semantic similarity using SentenceTransformer if available;
    falls back to keyword heuristics. Kept lightweight for concurrency safety.
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
        model = _embedder._load_classifier_model()  # type: ignore[attr-defined]
        with _classification_lock:
            q_emb = model.encode([question], normalize_embeddings=True)[0]
            scores = {}
            import numpy as np  # local import to avoid hard dep at module import

            for label, phrases in prototypes.items():
                emb = model.encode(phrases, normalize_embeddings=True)
                cat_vec = emb.mean(axis=0)
                scores[label] = float(
                    np.dot(q_emb, cat_vec) / (np.linalg.norm(cat_vec) + 1e-8)
                )
            winner = max(scores.items(), key=lambda kv: kv[1])[0]
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
    except Exception as e:  # noqa: BLE001
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
    """Internal helper to query Chroma with support for multi-key equality filters."""

    def _normalize_where(w: Dict) -> Dict:
        if not w:
            return {}
        if len(w) == 1:
            return w
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
            if m.get("granularity") != "summary"
        ]
    except Exception as e:  # noqa: BLE001
        print(f"Raw query error: {e}")
        return []


def _group_unit_docs(
    unit_docs: List[Dict], file_docs: Optional[List[Dict]] = None
) -> List[Dict]:
    """Group multiple unit-level docs under their file path and merge content.

    If file_docs provided (from high-level stage1) we prepend the first file-level
    chunk for broader context.
    """
    MAX_TOTAL_GROUPED_CONTEXT_CHARS = getattr(
        _embedder, "MAX_TOTAL_GROUPED_CONTEXT_CHARS", 60000
    )
    files: Dict[str, Dict] = {}
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
        units_sorted = sorted(
            bundle["units"], key=lambda d: d["metadata"].get("start_line", 0)
        )
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
    truncated = len(merged_docs) < len(files)
    print(
        f"[PIPELINE] Grouping complete: {len(merged_docs)} grouped docs (from {len(files)} files); total_chars={total_chars}; limit={MAX_TOTAL_GROUPED_CONTEXT_CHARS}; truncated={'YES' if truncated else 'NO'}"
    )
    if not merged_docs:
        return unit_docs
    return merged_docs


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
        import chromadb  # local import to avoid mandatory dependency at import time

        client = chromadb.PersistentClient(path=persist_dir)
        collection = client.get_collection(collection_name)
    except Exception as e:  # noqa: BLE001
        print(f"Could not open collection for advanced query: {e}")
        return {"classification": classification, "documents": []}

    # Embed query once
    try:
        query_embedding = _embedder.get_embeddings([question], "RETRIEVAL_QUERY", mode)[
            0
        ]
    except Exception as e:  # noqa: BLE001
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

    # High-level: stage1 file-level retrieval
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

    per_file_units: List[Dict] = []
    for fp in selected_files:
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


__all__ = [
    "LOW_LEVEL_LABEL",
    "HIGH_LEVEL_LABEL",
    "classify_question",
    "query_repository_advanced",
]
