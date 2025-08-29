"""Repository summarization module.

Encapsulates:
 - Section definitions
 - Summary cache read/write
 - LLM-driven file categorization & documentation generation
 - Orchestrated summary generation with commit hash tracking

Public functions:
 - get_summary(repo_url)
 - generate_or_get_summary(repo_url, force=False)
 - SECTION_DEFINITIONS (exported)
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

from .github_utils import fetch_repo_tree, parse_github_url, get_file_content

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

LLM_MODEL_NAME = "gemini-2.5-flash"
LLM_TEMPERATURE_IDENTIFY = 0.1
LLM_TEMPERATURE_GENERATE = 0.4
MAX_TOTAL_CONTEXT_CHARS = 50000

SUMMARY_CACHE_FILE = "summaries.json"

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


# ---------------- Cache Helpers -----------------
def _read_summaries() -> Dict[str, Any]:
    if not os.path.exists(SUMMARY_CACHE_FILE):
        return {}
    try:
        with open(SUMMARY_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError):
        return {}


def _write_summary(repo_url: str, commit_hash: str, summary_data: Dict[str, Any]):
    summaries = _read_summaries()
    summaries[repo_url] = {
        "timestamp": time.time(),
        "commit": commit_hash,
        "summary": summary_data,
    }
    with open(SUMMARY_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)


def get_summary(repo_url: str) -> Dict[str, Any] | None:
    summaries = _read_summaries()
    return summaries.get(repo_url)


# --------------- LLM Categorization ---------------
def get_important_files_by_category(repo_tree, sections=SECTION_DEFINITIONS):
    print("Identifying important files by category (structured JSON expected)...")
    if not repo_tree:
        return {s["id"]: [] for s in sections}
    file_list_str = "\n".join([item["path"] for item in repo_tree])
    sections_payload = [
        {"id": s["id"], "title": s["title"], "description": s.get("description", "")}
        for s in sections
    ]
    prompt_template_files = """
You are an expert software architect. Given a repository file list, return a JSON object that maps section IDs to an array of the most relevant file paths for that section. "Most relevant" files are those that directly implement or strongly support the section’s described functionality or intention.
Important rules:
- Output MUST be valid JSON and only the JSON object (no surrounding explanation).
- Keys must be the section ids provided in the `sections` input.
- Values must be arrays of strings with relative file paths (or an empty array if none).
Input:
Sections: {sections}
Repository File List:
{file_list}
"""

    def _extract_json_candidates(txt: str):
        fenced = re.findall(r"```(?:json)?\n(.*?)```", txt, re.DOTALL | re.IGNORECASE)
        if fenced:
            for f in fenced:
                yield f.strip()
        stack = []
        start = None
        for i, ch in enumerate(txt):
            if ch == "{":
                if start is None:
                    start = i
                stack.append(ch)
            elif ch == "}" and stack:
                stack.pop()
                if not stack and start is not None:
                    candidate = txt[start : i + 1]
                    yield candidate.strip()
                    start = None

    def _clean_json(txt: str) -> str:
        txt = re.sub(r",\s*(\]|})", r"\1", txt)
        txt = re.sub(
            r"'([^']*)'(?=\s*:)",
            lambda m: '"' + m.group(1).replace('"', '\\"') + '"',
            txt,
        )
        txt = re.sub(
            r":\s*'([^']*)'",
            lambda m: ': "' + m.group(1).replace('"', '\\"') + '"',
            txt,
        )
        return txt.strip()

    def _parse_identified(raw: str):
        tried = set()
        for cand in _extract_json_candidates(raw):
            if cand in tried:
                continue
            tried.add(cand)
            for attempt in range(2):
                attempt_txt = cand if attempt == 0 else _clean_json(cand)
                try:
                    obj = json.loads(attempt_txt)
                    if isinstance(obj, dict):
                        return obj
                except Exception:  # noqa: BLE001
                    continue
        return None

    try:
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
        identified = _parse_identified(raw_response) or {}
        if not identified:
            try:
                fixer_prompt = ChatPromptTemplate.from_template(
                    """You attempted to output JSON but it was invalid or empty. Below is your previous response.\nReturn ONLY a valid JSON object following the schema: mapping of section id -> array of file path strings.\nPrevious response:\n{previous}\n"""
                )
                fixer_chain = fixer_prompt | llm_identify | StrOutputParser()
                fix_raw = fixer_chain.invoke({"previous": raw_response[:8000]})
                identified = _parse_identified(fix_raw) or {}
            except Exception:  # noqa: BLE001
                pass
        result = {s["id"]: [] for s in sections}
        if isinstance(identified, dict):
            for sid, val in identified.items():
                if sid in result and isinstance(val, list):
                    result[sid] = [p for p in val if isinstance(p, str) and p.strip()]
        return result
    except Exception as e:  # noqa: BLE001
        print(f"Error during file identification: {e}")
        return {s["id"]: [] for s in sections}


def generate_documentation(
    repo_url: str, files_by_category, sections=SECTION_DEFINITIONS
):
    print("Generating documentation (using structured sections)...")
    all_files = set(f for files in files_by_category.values() for f in files)
    if not all_files:
        return {s["id"]: "No relevant files were identified." for s in sections}

    context_str = ""
    current_total_chars = 0
    for file_path in all_files:
        if current_total_chars >= MAX_TOTAL_CONTEXT_CHARS:
            break
        content = get_file_content(repo_url, file_path)
        if content:
            snippet = (
                f"\n\n--- Start: {file_path} ---\n{content}\n--- End: {file_path} ---\n"
            )
            if current_total_chars + len(snippet) <= MAX_TOTAL_CONTEXT_CHARS:
                context_str += snippet
                current_total_chars += len(snippet)

    if not context_str:
        return {
            s["id"]: "Failed to retrieve content for the identified files."
            for s in sections
        }
    try:
        llm_generate = ChatGoogleGenerativeAI(
            model=LLM_MODEL_NAME,
            temperature=LLM_TEMPERATURE_GENERATE,
            google_api_key=GOOGLE_API_KEY,
        )
        sections_desc = "\n".join([f"- {s['title']}" for s in sections])
        system_prompt = """You are an expert technical writer. Your task is to create a high-level, structured, and clear documentation for a software repository based on the provided file contents.

Follow these rules strictly:
1.  **Generate documentation for ALL sections listed below.**
2.  **Use the exact section titles provided, formatted as Markdown H2 headings (e.g., `## Purpose & Scope`).**
3.  **Base your analysis *only* on the provided file content.** Do not invent or assume features.
4.  If the provided context is insufficient for a section, write "Insufficient information to generate this section."

**Sections to generate:**
{sections}"""
        human_prompt = "Here is the repository context:\n\n{context}"
        prompt_template = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", human_prompt)]
        )
        chain = prompt_template | llm_generate | StrOutputParser()
        all_docs_str = chain.invoke({"context": context_str, "sections": sections_desc})

        generated_documentation: Dict[str, str] = {}
        if "##" not in all_docs_str:
            return {s["id"]: markdown2.markdown(all_docs_str) for s in sections}
        for s in sections:
            title = s["title"]
            pattern = f"## {re.escape(title)}(.*?)(\n## |$)"
            match = re.search(pattern, all_docs_str, re.DOTALL | re.IGNORECASE)
            content = (
                match.group(1).strip()
                if match
                else "Could not generate docs for this section."
            )
            generated_documentation[s["id"]] = markdown2.markdown(content)
        return generated_documentation
    except Exception as e:  # noqa: BLE001
        print(f"Error during documentation generation: {e}")
        return {s["id"]: "Error during generation." for s in sections}


# --------------- Orchestration ---------------
def _get_latest_commit(repo_url: str) -> str | None:
    if not GITHUB_TOKEN:
        return None
    try:
        owner, repo = parse_github_url(repo_url)
        headers = {"Authorization": f"token {GITHUB_TOKEN}"}
        repo_api = f"https://api.github.com/repos/{owner}/{repo}"
        repo_info = fetch_json(repo_api, headers)
        default_branch = repo_info.get("default_branch", "main")
        ref_url = f"{repo_api}/git/refs/heads/{default_branch}"
        ref = fetch_json(ref_url, headers)
        return ref.get("object", {}).get("sha")
    except Exception:  # noqa: BLE001
        return None


def fetch_json(url: str, headers: Dict[str, str]):
    import requests

    return requests.get(url, headers=headers).json()


def generate_or_get_summary(
    repo_url: str, force: bool = False
) -> Tuple[Dict[str, str] | None, str | None, bool]:
    """Return (documentation, commit_hash, is_cached)."""
    cached = get_summary(repo_url)
    if cached and not force:
        return cached.get("summary"), cached.get("commit"), True

    repo_tree = fetch_repo_tree(repo_url)
    if not repo_tree:
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
