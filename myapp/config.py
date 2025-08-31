"""Centralized configuration for the Repository Explainer application.

This module consolidates settings that are shared across different parts of the
application, such as API keys, LLM parameters, and persistence configurations.
By centralizing these values, we can ensure consistency and make it easier to
update the application's behavior from a single source of truth.
"""

import os
from typing import Any, Dict, List, Literal
import threading

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
except Exception:  # pragma: no cover - optional dependency errors handled later
    RecursiveCharacterTextSplitter = None
    Language = None

# --- External Service Credentials ---
# These keys are required for core functionality. They should be set as environment variables.
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

# --- LLM Configuration ---
# These parameters control the behavior of the language models used for summarization and QA.
LLM_MODEL_NAME = "gemini-2.5-flash"  # Unified model for all LLM tasks
LLM_TEMPERATURE_IDENTIFY = (
    0.1  # Low temperature for deterministic tasks like file classification
)
LLM_TEMPERATURE_GENERATE = (
    0.4  # Higher temperature for creative tasks like generating documentation
)
MAX_TOTAL_CONTEXT_CHARS = 100000  # Global character limit for LLM prompt context

# --- Embedding Configuration ---
# These settings control the embedding model and vector store.
EMBEDDING_MODE = os.environ.get("EMBEDDING_MODE", "remote")  # 'local' or 'remote'
EMBEDDING_MODEL_NAME_LOCAL = (
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

# --- Persistence & Cache Configuration ---
# Configuration for ChromaDB vector store and summary cache.
CHROMA_PERSIST_DIR = "chroma_persist"  # On-disk location for ChromaDB
CHROMA_COLLECTION_NAME = "repo_functions"  # Collection name for code units
SUMMARY_CACHE_FILE = "summaries.json"  # Flat file for caching repository summaries
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

# Generic (non-Python) code splitting configuration
CODE_SPLIT_CHUNK_SIZE = int(
    os.environ.get("REPO_EXPLAINER_CODE_CHUNK_SIZE", 1800)
)  # Target characters per chunk
CODE_SPLIT_CHUNK_OVERLAP = int(
    os.environ.get("REPO_EXPLAINER_CODE_CHUNK_OVERLAP", 200)
)  # Character overlap for continuity
MAX_UNIT_CHARS = int(
    os.environ.get("REPO_EXPLAINER_MAX_UNIT_CHARS", 6000)
)  # Hard ceiling safeguard for any single unit (trim/split if exceeded)

# Map file extensions to LangChain Language enum (best-effort; missing entries fallback to custom separators)
LANG_EXT_MAP = {
    ".js": "JS",
    ".ts": "TS",
    ".jsx": "JS",
    ".tsx": "TS",
    ".java": "JAVA",
    ".go": "GO",
    ".rs": "RUST",
    ".php": "PHP",
    ".rb": "RUBY",
    ".c": "C",
    ".h": "C",
    ".cpp": "CPP",
    ".cc": "CPP",
    ".hpp": "CPP",
    ".cs": "CSHARP",
    ".swift": "SWIFT",
    ".scala": "SCALA",
    ".kt": "KOTLIN",
    ".kts": "KOTLIN",
    ".html": "HTML",
    ".htm": "HTML",
    ".css": "CSS",
    ".md": "MARKDOWN",
    ".json": "JSON",
    ".xml": "XML",
    ".yml": "YAML",
    ".yaml": "YAML",
}

# --- Core Application Definitions ---
# Stable definitions used across the application for UI and prompts.
SECTION_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "id": "purpose_scope",
        "title": "Purpose & Scope",
        "description": "README, high-level documentation, and files that describe why the project exists and what it does.",
    },
    {
        "id": "system_architecture",
        "title": "System Architecture Overview",
        "description": "Configuration files, server entrypoints, routing, and high-level architecture information.",
    },
    {
        "id": "core_components",
        "title": "Core Components & Business Logic",
        "description": "Primary source code modules and packages implementing the core features and business logics.",
    },
    {
        "id": "data_model",
        "title": "Data Flow",
        "description": "Data flow across the system, database schemas and database interaction layers.",
    },
]
