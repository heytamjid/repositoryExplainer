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

LLM_MODEL_NAME = "gemini-1.5-flash"
LLM_TEMPERATURE_IDENTIFY = 0.1
LLM_TEMPERATURE_GENERATE = 0.4
MAX_TOTAL_CONTEXT_CHARS = 50000

CHROMA_PERSIST_DIR = "chroma_persist"
CHROMA_COLLECTION_NAME = "repo_functions"

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
"""
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
        match = re.search(r"\{(?:.|\n)*\}", raw_response)
        identified = json.loads(match.group(0)) if match else {}
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
    # ... (code is unchanged)
    documentation = None
    error = None
    repo_url = ""
    if request.method == "POST":
        repo_url = request.POST.get("repo_url")
        if repo_url:
            if not GITHUB_TOKEN or not GOOGLE_API_KEY:
                error = "Server configuration error: API keys are missing."
            else:
                repo_tree = fetch_repo_tree(repo_url)
                if repo_tree:
                    files_by_category = get_important_files_by_category(
                        repo_tree, SECTION_DEFINITIONS
                    )
                    documentation = generate_documentation(
                        repo_url, files_by_category, SECTION_DEFINITIONS
                    )
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
        },
    )


def ask_question(request):
    # ... (code is unchanged)
    repo_url = request.GET.get("repo_url", "")
    return render(request, "ask.html", {"repo_url": repo_url})


@csrf_exempt
def api_ask(request):
    if request.method != "POST":
        return HttpResponseBadRequest("POST required")

    try:
        body = json.loads(request.body.decode("utf-8"))
        question = body.get("question", "").strip()
        repo_url = body.get("repo_url", "").strip()
    except json.JSONDecodeError:
        return HttpResponseBadRequest("Invalid JSON")

    if not all([question, repo_url]):
        return JsonResponse(
            {"answer": "Both a question and a repository URL are required."}
        )

    if "clear" in question.lower() or "reset" in question.lower():
        embedder.clear_indexing_state(repo_url)
        return JsonResponse(
            {
                "answer": f"Indexing state cleared for {repo_url}. You can now ask questions again."
            }
        )

    repo_status = embedder._indexing_state.get(repo_url)

    if repo_status == "running":
        return JsonResponse(
            {
                "answer": "Repository is currently being indexed. Please try again in a few moments."
            }
        )

    if repo_status != "done":
        print("Repository not indexed. Starting indexing in background...")

        def _start_index_task():
            print(f"Starting indexing thread for {repo_url}")
            try:
                res = embedder.index_repository(
                    repo_url,
                    persist_dir=CHROMA_PERSIST_DIR,
                    collection_name=CHROMA_COLLECTION_NAME,
                )
                print("Background indexing result:", res)
            except Exception as e:
                print(f"Indexing thread for {repo_url} failed: {e}")

        threading.Thread(target=_start_index_task, daemon=True).start()
        return JsonResponse(
            {
                "answer": "This repository hasn't been indexed yet. I'm starting the process now. Please ask your question again in a few moments."
            }
        )

    try:
        print(f"Repository {repo_url} is indexed. Querying for: '{question}'")
        retrieved_docs = embedder.query_repository(
            question,
            repo_url,
            persist_dir=CHROMA_PERSIST_DIR,
            collection_name=CHROMA_COLLECTION_NAME,
        )

        if not retrieved_docs:
            return JsonResponse(
                {
                    "answer": "I could not find relevant information in the repository to answer this question."
                }
            )

        context = "Context from repository:\n"
        for doc in retrieved_docs:
            metadata = doc["metadata"]
            context += f"--- Start: {metadata['file_path']} (Lines: {metadata['start_line']}-{metadata['end_line']}) ---\n"
            context += doc["document"]
            context += f"\n--- End ---\n\n"

        # ## ADDED ##: Cleanly print the full context to the console for scrutiny.
        print("\n" + "=" * 80)
        print(" FULL CONTEXT FOR SCRUTINY ".center(80, "="))
        print(f"QUERY: {question}")
        print("-" * 80)
        print(context)
        print("=" * 80 + "\n")

        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL_NAME, google_api_key=GOOGLE_API_KEY
        )
        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant that answers questions about a code repository based *only* on the provided context. Be concise and clear. If the context doesn't contain the answer, say so.",
                ),
                ("human", "Question: {question}\n\nContext:\n{context}"),
            ]
        )
        chain = prompt_template | llm | StrOutputParser()
        answer = chain.invoke({"question": question, "context": context})

        return JsonResponse({"answer": answer})

    except Exception as e:
        print(f"Error during question answering: {e}")
        return JsonResponse(
            {"answer": f"An error occurred while processing your question: {str(e)}"}
        )
