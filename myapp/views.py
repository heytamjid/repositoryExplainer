import os
import requests
import base64
import re
import json
import time
import threading
from urllib.parse import urlparse

from django.shortcuts import render
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import markdown2

from . import embedder

# --- Configuration & Constants ---
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

LLM_MODEL_NAME = "gemini-2.5-flash"
LLM_TEMPERATURE_IDENTIFY = 0.1
LLM_TEMPERATURE_GENERATE = 0.4
MAX_TOTAL_CONTEXT_CHARS = 50000

CHROMA_PERSIST_DIR = "chroma_persist"
CHROMA_COLLECTION_NAME = "repo_functions"
SUMMARY_CACHE_FILE = "summaries.json"

SECTION_DEFINITIONS = [
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


# --- Summary Cache Functions ---
def read_summaries():
    """Reads the entire summary cache file."""
    if not os.path.exists(SUMMARY_CACHE_FILE):
        return {}
    try:
        with open(SUMMARY_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError):
        return {}


def write_summary(repo_url, commit_hash, summary_data):
    """Writes a summary for a specific repo to the cache file."""
    summaries = read_summaries()
    summaries[repo_url] = {
        "timestamp": time.time(),
        "commit": commit_hash,
        "summary": summary_data,
    }
    with open(SUMMARY_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)


def get_summary(repo_url):
    """Gets a specific summary from the cache."""
    summaries = read_summaries()
    return summaries.get(repo_url)


# --- GitHub API Functions (No changes here) ---
def parse_github_url(repo_url: str):
    path = urlparse(repo_url).path
    path = path.removesuffix(".git")
    owner, repo = path.lstrip("/").split("/")[:2]
    return owner, repo


def fetch_repo_tree(repo_url: str):
    # ... (code is unchanged)
    print("Fetching repository tree...")
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
        sha = ref["object"]["sha"]
        tree_url = f"{repo_api}/git/trees/{sha}?recursive=1"
        tree = requests.get(tree_url, headers=headers).json().get("tree", [])
        blobs = [item for item in tree if item.get("type") == "blob"]
        print(f"Successfully fetched {len(blobs)} file paths from the repository.")
        return blobs
    except Exception as e:
        print(f"Error fetching repo tree: {e}")
        return None


def get_file_content(repo_url: str, relative_file_path: str) -> str | None:
    # ... (code is unchanged)
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
    except Exception as e:
        print(f"Warning: GitHub API error fetching {relative_file_path}: {e}")
        return None


# --- LLM Interaction Functions (No changes here) ---
def get_important_files_by_category(repo_tree, sections=SECTION_DEFINITIONS):
    # ... (code is unchanged)
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

    # Helper: attempt to extract & sanitize JSON from model output
    def _extract_json_candidates(txt: str):
        # Strip code fences if present
        fenced = re.findall(r"```(?:json)?\n(.*?)```", txt, re.DOTALL | re.IGNORECASE)
        if fenced:
            for f in fenced:
                yield f.strip()
        # Bracket matching fallback
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
        # Remove trailing commas before } or ]
        txt = re.sub(r",\s*(\]|})", r"\1", txt)
        # Replace single quotes around keys/values with double quotes (simple heuristic)
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
        # Collapse duplicate whitespace
        return txt.strip()

    def _parse_identified(raw: str):
        tried = set()
        for cand in _extract_json_candidates(raw):
            if cand in tried:
                continue
            tried.add(cand)
            for attempt in range(2):  # raw then cleaned
                attempt_txt = cand if attempt == 0 else _clean_json(cand)
                try:
                    obj = json.loads(attempt_txt)
                    if isinstance(obj, dict):
                        return obj
                except Exception:
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
        # If still empty, attempt a corrective second pass asking model to fix JSON
        if not identified:
            try:
                fixer_prompt = ChatPromptTemplate.from_template(
                    """You attempted to output JSON but it was invalid or empty. Below is your previous response.\nReturn ONLY a valid JSON object following the schema: mapping of section id -> array of file path strings.\nPrevious response:\n{previous}\n"""
                )
                fixer_chain = fixer_prompt | llm_identify | StrOutputParser()
                fix_raw = fixer_chain.invoke({"previous": raw_response[:8000]})
                identified = _parse_identified(fix_raw) or {}
            except Exception:
                pass
        result = {s["id"]: [] for s in sections}
        if isinstance(identified, dict):
            for sid, val in identified.items():
                if sid in result and isinstance(val, list):
                    result[sid] = [p for p in val if isinstance(p, str) and p.strip()]
        return result
    except Exception as e:
        print(f"Error during file identification: {e}")
        return {s["id"]: [] for s in sections}


def generate_documentation(repo_url, files_by_category, sections=SECTION_DEFINITIONS):
    # ... (code is unchanged)
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

        generated_documentation = {}
        # Fallback for when the LLM doesn't follow instructions perfectly
        if "##" not in all_docs_str:
            return {s["id"]: markdown2.markdown(all_docs_str) for s in sections}

        for s in sections:
            title = s["title"]
            # Regex to find the content under a specific H2 heading until the next H2 or end of string
            pattern = f"## {re.escape(title)}(.*?)(\n## |$)"
            match = re.search(pattern, all_docs_str, re.DOTALL | re.IGNORECASE)
            content = (
                match.group(1).strip()
                if match
                else "Could not generate docs for this section."
            )
            generated_documentation[s["id"]] = markdown2.markdown(content)
        return generated_documentation
    except Exception as e:
        print(f"Error during documentation generation: {e}")
        return {s["id"]: "Error during generation." for s in sections}


# --- Django Views ---


def home(request):
    documentation = None
    error = None
    repo_url = ""
    cached_commit = None
    is_cached = False
    embedding_info = embedder.get_embedding_info()

    if request.method == "POST":
        repo_url = request.POST.get("repo_url")
        force_resummarize = request.POST.get("force_resummarize") == "true"

        if repo_url:
            if not GITHUB_TOKEN or not GOOGLE_API_KEY:
                error = "Server configuration error: API keys are missing."
            else:
                cached_summary = get_summary(repo_url)
                if cached_summary and not force_resummarize:
                    print(f"Displaying cached summary for {repo_url}")
                    documentation = cached_summary.get("summary")
                    cached_commit = cached_summary.get("commit")
                    is_cached = True
                else:
                    print(f"Generating new summary for {repo_url}")
                    repo_tree = fetch_repo_tree(repo_url)
                    if repo_tree:
                        # Get the latest commit hash to store with the summary
                        owner, repo = parse_github_url(repo_url)
                        headers = {"Authorization": f"token {GITHUB_TOKEN}"}
                        repo_api = f"https://api.github.com/repos/{owner}/{repo}"
                        repo_info = requests.get(repo_api, headers=headers).json()
                        default_branch = repo_info.get("default_branch", "main")
                        ref_url = f"{repo_api}/git/refs/heads/{default_branch}"
                        ref = requests.get(ref_url, headers=headers).json()
                        latest_commit = ref.get("object", {}).get("sha")

                        files_by_category = get_important_files_by_category(
                            repo_tree, SECTION_DEFINITIONS
                        )
                        documentation = generate_documentation(
                            repo_url, files_by_category, SECTION_DEFINITIONS
                        )
                        if latest_commit and documentation:
                            write_summary(repo_url, latest_commit, documentation)
                    else:
                        error = "Could not fetch the repository file structure. Please check the URL."

    return render(
        request,
        "home.html",
        {
            "documentation": documentation,
            "error": error,
            "repo_url": repo_url,
            "sections": SECTION_DEFINITIONS,
            "is_cached": is_cached,
            "cached_commit": cached_commit,
            "embedding_info": embedding_info,
        },
    )


def ask_question(request):
    # ... (code is unchanged)
    repo_url = request.GET.get("repo_url", "")
    embedding_info = embedder.get_embedding_info()
    return render(
        request,
        "ask.html",
        {
            "repo_url": repo_url,
            "embedding_info": embedding_info,
        },
    )


@csrf_exempt
def api_ask(request):
    if request.method != "POST":
        return HttpResponseBadRequest("POST required")

    try:
        body = json.loads(request.body.decode("utf-8"))
        question = body.get("question", "").strip()
        repo_url = body.get("repo_url", "").strip()
        action = body.get("action", "ask")  # 'ask', 'start_indexing', or 'get_config'
        embedding_mode = body.get("embedding_mode")  # Optional embedding mode override
    except json.JSONDecodeError:
        return HttpResponseBadRequest("Invalid JSON")

    if not repo_url and action not in ("get_config", "status"):
        return JsonResponse({"answer": "A repository URL is required."})

    # Handle configuration requests
    if action == "get_config":
        return JsonResponse(
            {
                "embedding_info": embedder.get_embedding_info(),
                "status": "config_retrieved",
            }
        )

    # Polling for indexing status only
    if action == "status":
        repo_status_data = embedder.get_indexing_status(
            repo_url, persist_dir=CHROMA_PERSIST_DIR
        )
        return JsonResponse(
            {
                "status": repo_status_data.get("status", "not_indexed"),
                "embedding_mode": repo_status_data.get(
                    "embedding_mode", embedder.EMBEDDING_MODE
                ),
            }
        )

    # Handle indexing requests
    if action == "start_indexing":
        print(f"Received request to start indexing for {repo_url}")
        if embedding_mode:
            print(f"Using embedding mode: {embedding_mode}")

        def _start_index_task():
            print(f"Starting indexing thread for {repo_url}")
            try:
                res = embedder.index_repository(
                    repo_url,
                    persist_dir=CHROMA_PERSIST_DIR,
                    collection_name=CHROMA_COLLECTION_NAME,
                    embedding_mode=embedding_mode,
                )
                print("Background indexing result:", res)
            except Exception as e:
                print(f"Indexing thread for {repo_url} failed: {e}")

        threading.Thread(target=_start_index_task, daemon=True).start()
        return JsonResponse(
            {
                "status": "indexing_started",
                "embedding_mode": embedding_mode or embedder.EMBEDDING_MODE,
            }
        )

    # Handle question answering
    if not question:
        return JsonResponse({"answer": "A question is required."})

    if "clear" in question.lower() or "reset" in question.lower():
        embedder.clear_indexing_state(repo_url, persist_dir=CHROMA_PERSIST_DIR)
        return JsonResponse(
            {
                "answer": f"Indexing state cleared for {repo_url}. You can now ask questions again."
            }
        )

    repo_status_data = embedder.get_indexing_status(
        repo_url, persist_dir=CHROMA_PERSIST_DIR
    )
    repo_status = repo_status_data.get("status")

    if repo_status == "running":
        return JsonResponse(
            {
                "answer": "Repository is currently being indexed. Please try again in a few moments.",
                "status": "indexing_in_progress",
            }
        )

    if repo_status != "done":
        return JsonResponse(
            {
                "answer": "This repository has not been indexed yet. Please click the 'Index Repository' button first.",
                "status": "not_indexed",
            }
        )

    try:
        print(f"Repository {repo_url} is indexed. Querying for: '{question}'")
        adv = embedder.query_repository_advanced(
            question,
            repo_url,
            persist_dir=CHROMA_PERSIST_DIR,
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_mode=embedding_mode,
        )
        retrieved_docs = adv.get("documents", [])
        classification = adv.get("classification")

        if not retrieved_docs:
            return JsonResponse(
                {
                    "answer": "I could not find relevant information in the repository to answer this question.",
                    "status": "no_results",
                    "classification": classification,
                }
            )

        # Build context using grouped docs; we'll append summary + repo map if available
        context = f"Query classification: {classification}\n\nContext from repository grouped by file:\n"
        for doc in retrieved_docs:
            metadata = doc["metadata"]
            context += f"=== FILE: {metadata['file_path']} (Lines: {metadata['start_line']}-{metadata['end_line']}) ===\n"
            context += doc["document"] + "\n\n"

        # Attempt to include cached summary (if exists) to enrich high-level answers
        cached_summary = get_summary(repo_url)
        summary_text = None
        if cached_summary and isinstance(cached_summary.get("summary"), dict):
            # Concatenate all section contents (already markdown) with headings
            sections_summary_parts = []
            for sid, content in cached_summary["summary"].items():
                # sid -> find matching title
                title = next(
                    (s["title"] for s in SECTION_DEFINITIONS if s["id"] == sid),
                    sid,
                )
                sections_summary_parts.append(f"## {title}\n{content}\n")
            summary_text = "\n".join(sections_summary_parts)
            context += "\n\n=== REPOSITORY SUMMARY (Precomputed) ===\n" + summary_text

        # Include repository map summary (lightweight structure overview)
        repo_map_summary = embedder.get_repo_map_summary(
            repo_url, persist_dir=CHROMA_PERSIST_DIR
        )
        if repo_map_summary and len(repo_map_summary) > 0:
            context += (
                "\n\n=== REPOSITORY MAP (Files & Units Overview) ===\n"
                + repo_map_summary
                + "\n"
            )

        # Metadata-only console logging
        print("\n" + "=" * 80)
        print(" PIPELINE METADATA SUMMARY ".center(80, "="))
        print(f"QUERY: {question}")
        print(f"EMBEDDING MODE: {embedding_mode or embedder.EMBEDDING_MODE}")
        print(f"CLASSIFICATION: {classification}")
        print(f"GROUPED DOC COUNT: {len(retrieved_docs)}")
        total_chars = sum(len(d["document"]) for d in retrieved_docs)
        print(f"TOTAL GROUPED CHARS: {total_chars}")
        if summary_text:
            print(
                f"SUMMARY INCLUDED: YES (chars={len(summary_text)}) - sections={len(cached_summary['summary'])}"
            )
        else:
            print("SUMMARY INCLUDED: NO")
        print("-" * 80)
        for idx, doc in enumerate(retrieved_docs, 1):
            m = doc["metadata"]
            print(
                f"[{idx}] file={m.get('file_path')} lines={m.get('start_line')}-{m.get('end_line')} gran={m.get('granularity')} size={len(doc['document'])}"
            )
        print("=" * 80 + "\n")

        # --------------------
        # Agentic Lookup Loop
        # --------------------
        additional_content = ""
        lookup_iterations = 0
        MAX_LOOKUP_ITERS = 2
        lookup_trace = []  # Collect decision-making trace for user visibility

        def _json_safe_parse(txt: str):
            try:
                match = re.search(r"\{(?:.|\n)*\}", txt)
                if match:
                    return json.loads(match.group(0))
            except Exception:
                return None
            return None

        repo_map = embedder.get_repo_map(repo_url, persist_dir=CHROMA_PERSIST_DIR)
        if not repo_map:
            print("[AGENT] No repository map found. Attempting on-demand generation...")
            repo_map = embedder.build_repo_map(repo_url, persist_dir=CHROMA_PERSIST_DIR)
            if repo_map:
                print("[AGENT] On-demand repository map generation succeeded.")
            else:
                print("[AGENT] On-demand repository map generation FAILED.")
                lookup_trace.append(
                    {
                        "iteration": 0,
                        "status": "no_repo_map_available",
                        "message": "Repository map not found (generation failed); planner skipped.",
                    }
                )
        if repo_map:
            while lookup_iterations < MAX_LOOKUP_ITERS:
                lookup_iterations += 1
                print(f"[AGENT] Lookup iteration {lookup_iterations} planning started")
                planner_llm = ChatGoogleGenerativeAI(
                    model=LLM_MODEL_NAME,
                    temperature=0.1,
                    google_api_key=GOOGLE_API_KEY,
                )
                plan_prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            "You are a repository Q&A planner. You receive: (1) user question, (2) current assembled context, (3) repository map listing files and unit names. Decide if MORE source code is required to answer confidently. If yes, choose up to 5 precise lookup requests.\nRules: output ONLY valid JSON object with keys: need_lookups (bool), reason (string), lookups (array). Each lookup item: {{type:'file'|'function', path:'relative/path.py', name: 'function_or_class_name' (omit for file)}}. Only pick functions that exist in the map. If existing context already shows the needed code, set need_lookups=false. Avoid speculative guesses.",
                        ),
                        (
                            "human",
                            "Question: {question}\n\nCURRENT_CONTEXT_SNIPPET (truncated to 4000 chars):\n{context_snippet}\n\nREPO_MAP (truncated):\n{repo_map_snippet}",
                        ),
                    ]
                )
                truncated_context = context[-4000:]
                truncated_map = repo_map_summary[:6000]
                planner_chain = plan_prompt | planner_llm | StrOutputParser()
                plan_raw = planner_chain.invoke(
                    {
                        "question": question,
                        "context_snippet": truncated_context,
                        "repo_map_snippet": truncated_map,
                    }
                )
                # Precompute sanitized preview to avoid backslash in f-string expression (Windows/Python parser quirk)
                _plan_preview = plan_raw[:500].replace("\n", " ").replace("\\n", " ")
                print(
                    f"[AGENT] Raw plan response (truncated 500 chars): {_plan_preview}"
                )
                plan = _json_safe_parse(plan_raw) or {}
                trace_entry = {
                    "iteration": lookup_iterations,
                    "raw_plan_preview": (
                        plan_raw[:300] + ("..." if len(plan_raw) > 300 else "")
                    ),
                    "parsed": plan,
                }
                if not plan or not plan.get("need_lookups"):
                    if plan:
                        print(
                            f"[AGENT] Planner decided no lookups needed. Reason: {plan.get('reason')}"
                        )
                        trace_entry.update(
                            {
                                "status": "no_lookups_needed",
                                "reason": plan.get("reason"),
                            }
                        )
                    else:
                        print(
                            "[AGENT] Planner parse failed or returned empty; stopping lookup loop"
                        )
                        trace_entry.update(
                            {
                                "status": "plan_parse_failed",
                            }
                        )
                    lookup_trace.append(trace_entry)
                    break
                lookups = plan.get("lookups", [])
                if not isinstance(lookups, list) or len(lookups) == 0:
                    print(
                        "[AGENT] Planner indicated lookups needed but provided none; aborting further iterations"
                    )
                    trace_entry.update(
                        {
                            "status": "empty_lookup_list",
                            "reason": plan.get("reason"),
                        }
                    )
                    lookup_trace.append(trace_entry)
                    break
                print(f"[AGENT] Planner requests {len(lookups)} lookups: {lookups}")
                fetched = embedder.lookup_repo_items(
                    repo_url, lookups, persist_dir=CHROMA_PERSIST_DIR
                )
                if not fetched:
                    print(
                        "[AGENT] Lookup fetch returned zero items; stopping iterations"
                    )
                    trace_entry.update(
                        {
                            "status": "lookups_fetched_zero",
                            "requested": lookups,
                            "fetched_count": 0,
                        }
                    )
                    lookup_trace.append(trace_entry)
                    break
                add_parts = []
                for item in fetched:
                    if item["type"] == "file":
                        print(
                            f"[AGENT] Fetched FILE {item['path']} length={len(item['content'])}"
                        )
                        add_parts.append(
                            f"=== LOOKUP FILE {item['path']} (full content) ===\n```\n{item['content']}\n```"
                        )
                    else:
                        print(
                            f"[AGENT] Fetched FUNCTION {item['name']} in {item['path']} lines {item.get('start_line')}-{item.get('end_line')} length={len(item['content'])}"
                        )
                        add_parts.append(
                            f"=== LOOKUP FUNCTION {item['name']} IN {item['path']} (lines {item.get('start_line')}-{item.get('end_line')}) ===\n```\n{item['content']}\n```"
                        )
                additional_chunk = "\n\n".join(add_parts)
                additional_content += "\n\n" + additional_chunk
                context += (
                    "\n\n=== ON-DEMAND LOOKUPS ITERATION {it} ===\n".format(
                        it=lookup_iterations
                    )
                    + additional_chunk
                )
                # If we fetched fewer than requested, further iterations likely not useful
                if len(fetched) < len(lookups):
                    print(
                        f"[AGENT] Fetched {len(fetched)} < requested {len(lookups)}; stopping further lookup iterations"
                    )
                    trace_entry.update(
                        {
                            "status": "partial_fetch",
                            "requested": lookups,
                            "fetched_count": len(fetched),
                        }
                    )
                    lookup_trace.append(trace_entry)
                    break
                # Successful full fetch
                trace_entry.update(
                    {
                        "status": "lookups_fetched",
                        "requested": lookups,
                        "fetched_count": len(fetched),
                    }
                )
                lookup_trace.append(trace_entry)

        final_llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL_NAME, google_api_key=GOOGLE_API_KEY, temperature=0.4
        )
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a helpful assistant that answers questions about a code repository based ONLY on the supplied context (retrieved chunks + optional on-demand lookups + summary + map).\n"
                        "If the answer is uncertain or code isn't shown, explicitly say so.\n\n"
                        "When referencing implementation details include minimal verbatim code in fenced blocks with language tags.\n"
                        "If on-demand lookup sections were added, you may use them. Do not hallucinate functions not present.\n"
                        "If multiple snippets are relevant, separate them and label with file path.\n"
                        "If after lookups the required code still isn't present, state limitations and suggest which file to inspect next (without fabricating)."
                    ),
                ),
                ("human", "Question: {question}\n\nFULL CONTEXT:\n{context}"),
            ]
        )
        answer_chain = answer_prompt | final_llm | StrOutputParser()
        answer = answer_chain.invoke({"question": question, "context": context})

        # Append trace to answer (user-visible)
        if lookup_trace:
            trace_lines = ["\n---\n**Agent Lookup Trace**"]
            for t in lookup_trace:
                status = t.get("status")
                itn = t.get("iteration")
                if status == "no_repo_map_available":
                    line = "- (init) No repository map available; planner skipped."
                    trace_lines.append(line)
                    print(f"[AGENT] TRACE {line}")
                else:
                    reason = t.get("reason")
                    fetched_count = t.get("fetched_count")
                    requested = t.get("requested")
                    if status == "no_lookups_needed":
                        line = f"- Iter {itn}: Planner decided no lookups needed. Reason: {reason}"
                    elif status == "plan_parse_failed":
                        line = f"- Iter {itn}: Planner output could not be parsed."
                    elif status == "empty_lookup_list":
                        line = f"- Iter {itn}: Planner asked for lookups but list was empty. Reason: {reason}"
                    elif status in (
                        "lookups_fetched",
                        "partial_fetch",
                        "lookups_fetched_zero",
                    ):
                        line = f"- Iter {itn}: status={status} requested={len(requested) if requested else 0} fetched={fetched_count}"
                    else:
                        line = f"- Iter {itn}: status={status}"
                    trace_lines.append(line)
                    print(f"[AGENT] TRACE {line}")
            answer += "\n" + "\n".join(trace_lines)

        # Return raw markdown so the frontend (Marked.js) can render consistently without double conversion
        return JsonResponse(
            {
                "answer": answer,
                "status": "success",
                "embedding_mode": repo_status_data.get("embedding_mode", "unknown"),
                "num_results": len(retrieved_docs),
                "classification": classification,
                "lookup_iterations": lookup_iterations,
                "lookup_trace": lookup_trace,
            }
        )

    except Exception as e:
        print(f"Error during question answering: {e}")
        return JsonResponse(
            {
                "answer": f"An error occurred while processing your question: {str(e)}",
                "status": "error",
            }
        )


@csrf_exempt
def api_embedding_config(request):
    """API endpoint to get and set embedding configuration."""
    if request.method == "GET":
        return JsonResponse(embedder.get_embedding_info())

    elif request.method == "POST":
        try:
            body = json.loads(request.body.decode("utf-8"))
            action = body.get("action")

            if action == "set_mode":
                new_mode = body.get("mode")
                if new_mode in ["local", "remote"]:
                    # Update environment variable (note: this only affects current process)
                    os.environ["EMBEDDING_MODE"] = new_mode
                    embedder.EMBEDDING_MODE = new_mode
                    return JsonResponse(
                        {
                            "status": "success",
                            "message": f"Embedding mode set to {new_mode}",
                            "embedding_info": embedder.get_embedding_info(),
                        }
                    )
                else:
                    return JsonResponse(
                        {
                            "status": "error",
                            "message": "Invalid embedding mode. Use 'local' or 'remote'.",
                        }
                    )

            elif action == "test_local":
                try:
                    # Test if local embedding model can be loaded
                    test_embeddings = embedder.get_embeddings_local(["test"])
                    return JsonResponse(
                        {
                            "status": "success",
                            "message": "Local embedding model is working",
                            "embedding_dimension": len(test_embeddings[0]),
                        }
                    )
                except Exception as e:
                    return JsonResponse(
                        {
                            "status": "error",
                            "message": f"Local embedding model test failed: {str(e)}",
                        }
                    )

            elif action == "test_remote":
                try:
                    # Test if remote embedding API is working
                    test_embeddings = embedder.get_embeddings_remote(
                        ["test"], "RETRIEVAL_QUERY"
                    )
                    return JsonResponse(
                        {
                            "status": "success",
                            "message": "Remote embedding API is working",
                            "embedding_dimension": len(test_embeddings[0]),
                        }
                    )
                except Exception as e:
                    return JsonResponse(
                        {
                            "status": "error",
                            "message": f"Remote embedding API test failed: {str(e)}",
                        }
                    )

            else:
                return JsonResponse(
                    {
                        "status": "error",
                        "message": "Invalid action. Use 'set_mode', 'test_local', or 'test_remote'.",
                    }
                )

        except json.JSONDecodeError:
            return HttpResponseBadRequest("Invalid JSON")

    else:
        return HttpResponseBadRequest("GET or POST required")
