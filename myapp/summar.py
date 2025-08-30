"""Repository summarization module.

Provides an end‑to‑end pipeline that:
    1. Defines high‑level documentation sections (``SECTION_DEFINITIONS``).
    2. Caches generated summaries keyed by repository URL + commit hash.
    3. Uses an LLM to classify repository files into those sections.
    4. Retrieves file contents (budgeting context size) and asks an LLM to
         draft structured Markdown documentation per section.
    5. Exposes orchestration helpers to either fetch a cached summary or
         regenerate a new one if forced or stale.

Design goals:
    * Keep view layer thin – all summarization logic lives here.
    * Be resilient to imperfect / non‑JSON LLM outputs with post‑processing.
    * Avoid excessive token usage via a hard character budget.
    * Provide deterministic section IDs so front‑end rendering is stable.

Public API Surface:
    * ``SECTION_DEFINITIONS`` – list of section metadata dicts.
    * ``get_summary(repo_url)`` – fetch cached summary payload (if any).
    * ``generate_or_get_summary(repo_url, force=False)`` – orchestrate run & cache.
    * ``get_important_files_by_category(repo_tree, sections=...)`` – LLM file selection.
    * ``generate_documentation(repo_url, files_by_category, sections=...)`` – LLM drafting.
"""

from __future__ import annotations

import os
import json
import time
import re
from typing import Dict, List, Any, Tuple

import markdown2
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .github_utils import (
    fetch_repo_tree,
    get_file_content,
    _get_latest_commit,
)
from .json_utils import parse_identified

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")  # Required for Gemini access
GITHUB_TOKEN = os.environ.get(
    "GITHUB_TOKEN"
)  # Enables higher GitHub rate limits + commit lookups

# LLM configuration knobs (kept local to summarization so they can diverge from other modules if needed)
LLM_MODEL_NAME = (
    "gemini-2.5-flash"  # Fast, cost‑aware model variant for summarization tasks
)
LLM_TEMPERATURE_IDENTIFY = 0.1  # Low temp for deterministic file categorization output
LLM_TEMPERATURE_GENERATE = (
    0.4  # Slightly creative but still grounded for documentation prose
)
MAX_TOTAL_CONTEXT_CHARS = (
    50000  # Hard cap on concatenated file content to limit prompt size & cost
)

SUMMARY_CACHE_FILE = (
    "summaries.json"  # Flat JSON store (simple persistence, no DB dependency)
)

SECTION_DEFINITIONS: List[Dict[str, Any]] = (
    [  # Stable section schema used by UI & LLM prompts
        {
            "id": "purpose_scope",  # High‑level intent / README perspective
            "title": "Purpose & Scope",
            "description": "README, high-level documentation, and files that describe why the project exists and what it does.",
        },
        {
            "id": "system_architecture",  # Entrypoints, config, routing, infra hints
            "title": "System Architecture Overview",
            "description": "Configuration files, server entrypoints, routing, and high-level architecture information.",
        },
        {
            "id": "core_components",  # Core business/domain logic modules
            "title": "Core Components & Business Logic",
            "description": "Primary source code modules and packages implementing the core features and business logics.",
        },
        {
            "id": "data_model",  # Data flow & persistence concerns
            "title": "Data Flow",
            "description": "Data flow across the system, database schemas and database interaction layers.",
        },
    ]
)


# ---------------- Cache Helpers -----------------
def _read_summaries() -> Dict[str, Any]:
    """Load the entire summaries cache file.

    Returns an empty dict if file is missing or corrupt to keep the system
    resilient (caller treats absence as cache miss).
    """
    if not os.path.exists(SUMMARY_CACHE_FILE):
        return {}
    try:
        with open(SUMMARY_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError):  # Silent failure -> recomputation path
        return {}


def _write_summary(repo_url: str, commit_hash: str, summary_data: Dict[str, Any]):
    """Persist (or overwrite) a repository's summary snapshot.

    Stores: last generation timestamp, commit hash for staleness checking, and the sectioned summary.
    Flat-file approach keeps infra minimal; race conditions are acceptable for this use‑case.
    """
    summaries = _read_summaries()
    summaries[repo_url] = {
        "timestamp": time.time(),  # Helps future TTL logic if desired
        "commit": commit_hash,  # Links summary to repository state
        "summary": summary_data,  # Section -> rendered HTML/Markdown fragment
    }
    with open(SUMMARY_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)


def get_summary(repo_url: str) -> Dict[str, Any] | None:
    """Return cached summary record (or None) for a repository URL.

    Caller interprets presence + commit hash for cache reuse decisions.
    """
    summaries = _read_summaries()
    return summaries.get(repo_url)


# --------------- LLM Categorization ---------------
def get_important_files_by_category(repo_tree, sections=SECTION_DEFINITIONS):
    """Use an LLM to map repo files to documentation sections.

    Strategy:
      * Provide the raw list of file paths + structured section metadata.
      * Instruct model to output strict JSON (id -> [paths]).
      * Attempt tolerant parsing: capture fenced blocks OR raw JSON objects.
      * If first pass fails, run a "fixer" prompt to coerce valid JSON.

    Returns dict: section_id -> list[file_path]. Missing / invalid output -> empty lists.
    """
    print("Identifying important files by category (structured JSON expected)...")
    if not repo_tree:  # Early exit: nothing to classify
        return {s["id"]: [] for s in sections}

    # Flatten tree into simple newline string for token efficiency
    file_list_str = "\n".join([item["path"] for item in repo_tree])

    # Lightweight schema handed to LLM (avoid verbose descriptions beyond necessity)
    sections_payload = [
        {"id": s["id"], "title": s["title"], "description": s.get("description", "")}
        for s in sections
    ]

    # Prompt engineered for deterministic JSON output
    prompt_template_files = """
You are an expert software architect. Given a repository file list, return a JSON object that maps section IDs to an array of the most relevant file paths for that section. "Most relevant" files are those that directly implement or strongly support the section’s described functionality or intention.
For example, views.py, models.py etc files are the most relevant ones that defines the core logics, data models etc in a Django web application repository. 
Important rules:
- Output MUST be valid JSON and only the JSON object (no surrounding explanation).
- Keys must be the section ids provided in the `sections` input.
- Values must be arrays of strings with relative file paths (or an empty array if none).
Input:
Sections: {sections}
Repository File List:
{file_list}
"""
    try:
        # Build chain: prompt -> model -> raw string
        llm_identify = ChatGoogleGenerativeAI(
            model=LLM_MODEL_NAME,
            temperature=LLM_TEMPERATURE_IDENTIFY,
            google_api_key=GOOGLE_API_KEY,
        )
        chain_identify = (
            ChatPromptTemplate.from_template(prompt_template_files)
            | llm_identify
            | StrOutputParser()
        )
        raw_response = chain_identify.invoke(
            {"file_list": file_list_str, "sections": json.dumps(sections_payload)}
        )
        identified = parse_identified(raw_response) or {}

        # If initial attempt failed, run a small corrective pass with the original output
        if not identified:
            try:
                fixer_prompt = ChatPromptTemplate.from_template(
                    """You attempted to output JSON but it was invalid or empty. Below is your previous response.\nReturn ONLY a valid JSON object following the schema: mapping of section id -> array of file path strings.\nPrevious response:\n{previous}\n"""
                )
                fixer_chain = fixer_prompt | llm_identify | StrOutputParser()
                fix_raw = fixer_chain.invoke({"previous": raw_response[:80000]})
                identified = parse_identified(fix_raw) or {}
            except Exception:  # noqa: BLE001
                pass  # Best effort – fall through with empty mapping

        # Normalize structure to ensure all sections exist even if LLM omitted them
        result = {s["id"]: [] for s in sections}
        if isinstance(identified, dict):
            for sid, val in identified.items():
                if sid in result and isinstance(val, list):
                    # Filter out non-string or blank paths defensively
                    result[sid] = [p for p in val if isinstance(p, str) and p.strip()]
        return result
    except Exception as e:  # noqa: BLE001
        print(f"Error during file identification: {e}")
        return {s["id"]: [] for s in sections}  # Safe fallback


def generate_documentation(
    repo_url: str, files_by_category, sections=SECTION_DEFINITIONS
):
    """Produce per‑section documentation using retrieved file contents.

    Steps:
      1. Flatten unique file paths chosen by classification stage.
      2. Retrieve each file's content (respecting context size budget).
      3. Assemble annotated snippets demarcated with start/end markers.
      4. Prompt LLM to draft Markdown with explicit H2 headings for each section.
      5. Slice resulting Markdown back into section‑scoped fragments.

    Returns mapping: section_id -> rendered HTML string (Markdown converted via ``markdown2``).
    """
    print("Generating documentation (using structured sections)...")
    all_files = set(f for files in files_by_category.values() for f in files)
    if not all_files:  # No selected files -> nothing to summarize
        return {s["id"]: "No relevant files were identified." for s in sections}

    # --------- Build bounded context window from file contents ---------
    context_str = ""
    current_total_chars = 0
    for file_path in all_files:
        if current_total_chars >= MAX_TOTAL_CONTEXT_CHARS:
            break  # Enforce global context character budget
        content = get_file_content(repo_url, file_path)
        if content:
            snippet = (
                f"\n\n--- Start: {file_path} ---\n{content}\n--- End: {file_path} ---\n"
            )
            if current_total_chars + len(snippet) <= MAX_TOTAL_CONTEXT_CHARS:
                context_str += snippet
                current_total_chars += len(snippet)

    if not context_str:  # Retrieval failures -> surface graceful message
        return {
            s["id"]: "Failed to retrieve content for the identified files."
            for s in sections
        }
    try:
        # Chain for generation: templated system + human context -> LLM -> text
        llm_generate = ChatGoogleGenerativeAI(
            model=LLM_MODEL_NAME,
            temperature=LLM_TEMPERATURE_GENERATE,
            google_api_key=GOOGLE_API_KEY,
        )
        system_prompt = """You are an expert technical writer. Your task is to create a high-level, structured, and clear documentation for a software repository based on the provided file contents.

        You will be given a list of sections, each with an 'id', 'title', and 'description'. Your output MUST be a single JSON object.
        The keys of the JSON object must be the 'id' from the sections provided.
        The values must be a string containing the generated documentation in Markdown format for that section.

        Follow these rules strictly:
        1.  **Generate documentation for ALL section IDs provided.**
        2.  **Base your analysis *only* on the provided file content.** Do not invent or assume features.
        3.  If the provided context is insufficient for a section, the value should be a short explanation like "Insufficient information to generate this section."

        **Section Definitions:**
        {sections}"""
        human_prompt = "Here is the repository context:\n\n{context}"  # Raw concatenated file snippets
        prompt_template = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", human_prompt)]
        )
        chain = prompt_template | llm_generate | StrOutputParser()
        raw_response = chain.invoke(
            {"context": context_str, "sections": json.dumps(sections)}
        )

        # Parse the JSON response from the LLM
        generated_docs = parse_identified(raw_response) or {}

        # Convert markdown content to HTML for each section
        generated_documentation: Dict[str, str] = {}
        for s in sections:
            section_id = s["id"]
            markdown_content = generated_docs.get(
                section_id, "Could not generate docs for this section."
            )
            generated_documentation[section_id] = markdown2.markdown(markdown_content)

        return generated_documentation
    except Exception as e:  # noqa: BLE001
        print(f"Error during documentation generation: {e}")
        return {s["id"]: "Error during generation." for s in sections}


# --------------- Orchestration ---------------


def generate_or_get_summary(
    repo_url: str, force: bool = False
) -> Tuple[Dict[str, str] | None, str | None, bool]:
    """High-level orchestration returning (documentation, commit_hash, is_cached).

    Flow:
      * If cache exists and not forced -> return cached summary immediately.
      * Else: fetch repo tree (abort if unavailable).
      * Resolve latest commit (optional – absence just skips cache write).
      * Classify important files per section (LLM) -> select file set.
      * Generate documentation (LLM) -> section -> HTML mapping.
      * Persist to cache (if commit available) for future reuse.
    """
    cached = get_summary(repo_url)
    if cached and not force:
        return cached.get("summary"), cached.get("commit"), True

    repo_tree = fetch_repo_tree(repo_url)
    if not repo_tree:  # Tree fetch failure -> signal no summary
        return None, None, False

    latest_commit = _get_latest_commit(repo_url)
    files_by_category = get_important_files_by_category(repo_tree, SECTION_DEFINITIONS)
    documentation = generate_documentation(
        repo_url, files_by_category, SECTION_DEFINITIONS
    )
    if latest_commit and documentation:
        _write_summary(repo_url, latest_commit, documentation)
    return documentation, latest_commit, False


__all__ = [
    "SECTION_DEFINITIONS",
    "get_summary",
    "get_important_files_by_category",
    "generate_documentation",
    "generate_or_get_summary",
]
