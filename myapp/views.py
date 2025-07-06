import os
import requests
import base64
import re
from urllib.parse import urlparse
from collections import defaultdict
import time

from django.shortcuts import render
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import markdown2

# --- Configuration & Constants ---

# Using environment variables is a good practice for API keys.
# Ensure GOOGLE_API_KEY and GITHUB_TOKEN are set in your environment.
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")

LLM_MODEL_NAME = "gemini-2.5-flash"
LLM_TEMPERATURE_IDENTIFY = 0.1
LLM_TEMPERATURE_GENERATE = 0.4

CATEGORIES = [
    "Purpose & Scope",
    "System Architecture Overview",
    "Core Components (Implementation Details)",
    "Data Model",
]

MAX_TOTAL_CONTEXT_CHARS = 30000

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


def get_important_files_by_category(repo_tree):
    print("Identifying important files by category...")
    if not repo_tree:
        return {}

    file_list_str = "\n".join([item["path"] for item in repo_tree])

    prompt_template_files = """
You are an expert software architect analyzing a GitHub repository's file structure.
Your goal is to identify the *most important and relevant* files for understanding each of the following categories.

Repository File List:
{file_list}

Based on the file paths, list the most relevant file paths for each category below.
Prioritize files that likely contain defining information.

Categories:
1.  **Purpose & Scope**: (README, high-level documentation, main application files)
2.  **System Architecture Overview**: (Configuration files, main application/server files, routing files, docker-compose.yml)
3.  **Core Components (Implementation Details)**: (Source code directories like 'src/', 'lib/', 'app/', key modules)
4.  **Data Model**: (Files named 'models.py', 'schemas.py', database interaction layers, ORM definitions)

Provide your answer *strictly* in the following format, listing the file paths under each category heading.

**Purpose & Scope:**
- path/to/relevant/file1.ext

**System Architecture Overview:**
- path/to/relevant/file2.ext

**Core Components (Implementation Details):**
- path/to/core/logic.py

**Data Model:**
- path/to/models.py
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
        print("Invoking LLM to identify important files...")
        llm_response_files = chain_identify.invoke({"file_list": file_list_str})

        identified_files_by_category = {category: [] for category in CATEGORIES}
        current_category_key = None
        category_header_map = {
            "**Purpose & Scope:**": "Purpose & Scope",
            "**System Architecture Overview:**": "System Architecture Overview",
            "**Core Components (Implementation Details):**": "Core Components (Implementation Details)",
            "**Data Model:**": "Data Model",
        }

        for line in llm_response_files.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            is_header = False
            for header, key in category_header_map.items():
                if line.startswith(header):
                    current_category_key = key
                    is_header = True
                    break
            if not is_header and current_category_key and line.startswith("-"):
                file_path = line[1:].strip()
                if file_path:
                    identified_files_by_category[current_category_key].append(file_path)
        print("Successfully identified files.")
        return identified_files_by_category
    except Exception as e:
        print(f"Error during file identification: {e}")
        return {}


def generate_documentation(repo_url, files_by_category):
    print("Generating documentation...")
    generated_documentation = {}
    all_files = set()
    for files in files_by_category.values():
        all_files.update(files)

    if not all_files:
        return {
            category: "No relevant files were identified." for category in CATEGORIES
        }

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
            category: "Failed to retrieve content for the identified files."
            for category in CATEGORIES
        }

    try:
        llm_generate = ChatGoogleGenerativeAI(
            model=LLM_MODEL_NAME,
            temperature=LLM_TEMPERATURE_GENERATE,
            google_api_key=GOOGLE_API_KEY,
        )
        output_parser = StrOutputParser()

        system_prompt_content_generation = """
You are an expert technical writer. Your task is to generate clear, concise documentation for a GitHub repository based *only* on the provided code snippets.
- For each category provided, write a detailed explanation based on the context.
- If the context is insufficient for a category, state that clearly. Do not invent information.
- Structure your entire response in Markdown.
- Use the category titles as Level 2 Markdown headers (e.g., `## Purpose & Scope`).
"""
        prompt_template_generate = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt_content_generation),
                (
                    "human",
                    """
            Based *only* on the following context (code snippets from relevant files), generate documentation for all the following categories:
            {categories}

            Context:
            {context}
            ---
            Task: Provide a detailed explanation for each category. Structure your response with each category as a markdown header.
            """,
                ),
            ]
        )
        chain_generate = prompt_template_generate | llm_generate | output_parser

        print("Invoking LLM to generate documentation...")
        all_docs_str = chain_generate.invoke(
            {"context": context_str, "categories": "\n".join(CATEGORIES)}
        )

        # Parse the combined response
        for category in CATEGORIES:
            # Use regex to find the content for each category
            pattern = f"## {re.escape(category)}(.*?)(\n## |$)"
            match = re.search(pattern, all_docs_str, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                generated_documentation[category] = markdown2.markdown(content)
            else:
                generated_documentation[category] = (
                    "Could not generate documentation for this category."
                )
        print("Successfully generated documentation.")
        return generated_documentation
    except Exception as e:
        print(f"Error during documentation generation: {e}")
        return {category: "Error during generation." for category in files_by_category}


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
                    files_by_category = get_important_files_by_category(repo_tree)
                    if files_by_category:
                        print("Waiting for 5 seconds to avoid rate limiting...")
                        time.sleep(5)
                        print("Step 3: Generating documentation.")
                        documentation = generate_documentation(
                            repo_url, files_by_category
                        )
                    else:
                        error = "Could not identify important files in the repository."
                else:
                    error = "Could not fetch the repository file structure. Please check the URL and repository permissions."

    return render(
        request,
        "home.html",
        {"documentation": documentation, "error": error, "repo_url": repo_url},
    )
