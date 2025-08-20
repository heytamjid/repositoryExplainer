import os
import requests
import base64
import re
import json
from urllib.parse import urlparse
from collections import defaultdict
import time

from django.shortcuts import render
from django.http import JsonResponse, HttpResponseBadRequest
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import markdown2

# --- Configuration & Constants ---


GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

LLM_MODEL_NAME = "gemini-2.5-flash"
LLM_TEMPERATURE_IDENTIFY = 0.1
LLM_TEMPERATURE_GENERATE = 0.4

# Sections are now modular and configurable. Each section has a stable id, a human title and a short description.
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
        "title": "Core Components (Implementation Details)",
        "description": "Primary source code modules and packages implementing the main features.",
    },
    {
        "id": "data_model",
        "title": "Data Model",
        "description": "ORM models, schemas, and database interaction layers.",
    },
]

# Backwards-compatible constant for max context
MAX_TOTAL_CONTEXT_CHARS = 50000

# --- GitHub API Functions ---


def parse_github_url(repo_url: str):
    path = urlparse(repo_url).path
    path = path.removesuffix(".git")
    owner, repo = path.lstrip("/").split("/")[:2]
    return owner, repo


def fetch_repo_tree(repo_url: str):
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


# --- LLM Interaction Functions ---


def get_important_files_by_category(repo_tree, sections=SECTION_DEFINITIONS):
    """Identify important files per section. The LLM is instructed to return a strict JSON object
    mapping section ids to a list of file paths. This function validates and falls back to a
    best-effort bullet-parsing if JSON cannot be parsed.
    """
    print("Identifying important files by category (structured JSON expected)...")
    if not repo_tree:
        return {s["id"]: [] for s in sections}

    file_list_str = "\n".join([item["path"] for item in repo_tree])

    # Build a JSON description of the sections to pass to the LLM so keys are explicit and stable
    sections_payload = [
        {"id": s["id"], "title": s["title"], "description": s.get("description", "")}
        for s in sections
    ]

    prompt_template_files = """
You are an expert software architect. Given a repository file list, return a JSON object that maps
section IDs to an array of the most relevant file paths for that section.

Important rules:
- Output MUST be valid JSON and only the JSON object (no surrounding explanation).
- Keys must be the section ids provided in the `sections` input.
- Values must be arrays of strings with relative file paths (or an empty array if none).

Input:
Sections: {sections}
Repository File List:
{file_list}

Example expected output format:
{{
  "purpose_scope": ["README.md", "docs/overview.md"],
  "system_architecture": ["docker-compose.yml", "deploy/prod.yaml"],
  "core_components": ["src/main.py"],
  "data_model": ["models.py"]
}}
"""
    try:
        llm_identify = ChatGoogleGenerativeAI(
            model=LLM_MODEL_NAME,
            temperature=LLM_TEMPERATURE_IDENTIFY,
            google_api_key=GOOGLE_API_KEY,
        )
        file_identification_prompt = ChatPromptTemplate.from_template(
            prompt_template_files
        )
        chain_identify = file_identification_prompt | llm_identify | StrOutputParser()
        print("Invoking LLM to identify important files (JSON expected)...")
        raw_response = chain_identify.invoke(
            {"file_list": file_list_str, "sections": json.dumps(sections_payload)}
        )

        # Try to parse returned JSON strictly. If that fails, attempt to extract the first JSON object.
        identified = None
        try:
            identified = json.loads(raw_response)
        except Exception:
            # Attempt to extract a JSON substring
            match = re.search(r"\{(?:.|\n)*\}", raw_response)
            if match:
                try:
                    identified = json.loads(match.group(0))
                except Exception:
                    identified = None

        # Validate and normalize result into a mapping from section id -> list[str]
        result = {s["id"]: [] for s in sections}
        if isinstance(identified, dict):
            for sid, val in identified.items():
                if sid in result and isinstance(val, list):
                    result[sid] = [p for p in val if isinstance(p, str) and p.strip()]
        else:
            # Fall back to older bullet-parsing if JSON not provided: best-effort
            print(
                "LLM did not return valid JSON; attempting fallback parsing of bullets."
            )
            current_key = None
            # Map titles to ids for fallback detection
            title_to_id = {s["title"]: s["id"] for s in sections}
            for line in raw_response.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                # header detection
                if line in title_to_id:
                    current_key = title_to_id[line]
                    continue
                if line.startswith("-") and current_key:
                    p = line[1:].strip()
                    if p:
                        result[current_key].append(p)
        print("Successfully identified files (structured).")
        return result
    except Exception as e:
        print(f"Error during file identification: {e}")
        return {s["id"]: [] for s in sections}


def generate_documentation(repo_url, files_by_category, sections=SECTION_DEFINITIONS):
    """Generate documentation. files_by_category is expected to be a dict mapping section id -> list of file paths."""
    print("Generating documentation (using structured sections)...")
    generated_documentation = {}

    # Flatten all files
    all_files = set()
    for files in files_by_category.values():
        all_files.update(files)

    if not all_files:
        return {s["id"]: "No relevant files were identified." for s in sections}

    context_str = ""
    current_total_chars = 0
    print(f"Fetching content for {len(all_files)} files...")
    for file_path in all_files:
        if current_total_chars >= MAX_TOTAL_CONTEXT_CHARS:
            print(
                f"Warning: Reached max context size of {MAX_TOTAL_CONTEXT_CHARS} chars."
            )
            break
        content = get_file_content(repo_url, file_path)
        if content:
            snippet = f"\n\n--- Start of content from: {file_path} ---\n{content}\n--- End of content from: {file_path} ---\n"
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
        output_parser = StrOutputParser()

        # Use the modular sections when instructing the model
        sections_desc = "\n".join(
            [
                f"- {s['id']}: {s['title']} -- {s.get('description','')}"
                for s in sections
            ]
        )

        system_prompt_content_generation = """
You are an expert technical writer. Your task is to generate clear, concise documentation for a GitHub repository based *only* on the provided code snippets.
- For each section provided, write a clear explanation based only on the context.
- If the context is insufficient for a section, state that clearly. Do not invent information.
- Structure your entire response in Markdown.
- Use the section TITLE as a Level 2 Markdown header (e.g., `## Purpose & Scope`).
"""
        prompt_template_generate = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt_content_generation),
                (
                    "human",
                    """
            Based *only* on the following context (code snippets from relevant files), generate documentation for all the following sections:
            {sections}

            Context:
            {context}
            ---
            Task: Provide a detailed explanation for each section. Structure your response with each section as a markdown header.
            """,
                ),
            ]
        )
        chain_generate = prompt_template_generate | llm_generate | output_parser

        print("Invoking LLM to generate documentation...")
        all_docs_str = chain_generate.invoke(
            {"context": context_str, "sections": sections_desc}
        )

        # Parse the combined response per section title
        for s in sections:
            title = s["title"]
            pattern = f"## {re.escape(title)}(.*?)(\n## |$)"
            match = re.search(pattern, all_docs_str, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                generated_documentation[s["id"]] = markdown2.markdown(content)
            else:
                generated_documentation[s["id"]] = (
                    "Could not generate documentation for this section."
                )
        print("Successfully generated documentation.")
        return generated_documentation
    except Exception as e:
        print(f"Error during documentation generation: {e}")
        return {s["id"]: "Error during generation." for s in sections}


# --- Django View ---


def home(request):
    documentation = None
    error = None
    repo_url = ""
    if request.method == "POST":
        repo_url = request.POST.get("repo_url")
        if repo_url:
            print(f"Starting analysis for repository: {repo_url}")
            if not GITHUB_TOKEN or not GOOGLE_API_KEY:
                error = "Server configuration error: API keys are missing."
            else:
                print("Step 1: Fetching repository tree.")
                repo_tree = fetch_repo_tree(repo_url)
                if repo_tree:
                    print("Step 2: Identifying important files.")
                    files_by_category = get_important_files_by_category(
                        repo_tree, SECTION_DEFINITIONS
                    )
                    if files_by_category:
                        print("Waiting for 5 seconds to avoid rate limiting...")
                        time.sleep(5)
                        print("Step 3: Generating documentation.")
                        documentation = generate_documentation(
                            repo_url, files_by_category, SECTION_DEFINITIONS
                        )
                    else:
                        error = "Could not identify important files in the repository."
                else:
                    error = "Could not fetch the repository file structure. Please check the URL and repository permissions."

    # Render output with documentation keyed by section id. The template can map ids to titles using SECTION_DEFINITIONS if needed.
    return render(
        request,
        "home.html",
        {
            "documentation": documentation,
            "error": error,
            "repo_url": repo_url,
            "sections": SECTION_DEFINITIONS,
        },
    )


def ask_question(request):
    """Render the chat UI where users can ask questions about the repository.
    Embedding/indexing will be implemented later; this view only renders the page.
    """
    # Allow pre-filling the repo URL via query parameter (from home page)
    repo_url = request.GET.get("repo_url", "")
    return render(request, "ask.html", {"repo_url": repo_url})


def api_ask(request):
    """Placeholder API endpoint for handling question POSTs from the chat UI.
    Currently returns a static message; embedding/indexing logic will be added later.
    """
    if request.method != "POST":
        return HttpResponseBadRequest("POST required")
    try:
        body = json.loads(request.body.decode("utf-8"))
    except Exception:
        return HttpResponseBadRequest("Invalid JSON")
    question = body.get("question", "").strip()
    repo_url = body.get("repo_url", "").strip()
    if not question:
        return JsonResponse({"answer": "Please provide a question."})

    # Placeholder answer. Actual retrieval + LLM answer generation will be implemented later.
    placeholder = (
        "Embedding backend not yet enabled. When enabled, the server will index the repository and "
        "return grounded answers with provenance (file path + lines)."
    )
    # Include repo_url in placeholder response for client-side verification
    resp = {"answer": placeholder}
    if repo_url:
        resp["repo_url"] = repo_url
    return JsonResponse(resp)
