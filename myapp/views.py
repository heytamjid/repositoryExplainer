import os
import requests
import base64
import re
import json
import time
import threading
from urllib.parse import (
    urlparse,
)  # (kept if other code still references, will be removable later)

from django.shortcuts import render
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import markdown2  # (may be removable after full migration)

from . import embedder  # retains backward compatibility
from .agent import (
    run_agentic_lookups,
)  # kept for backward compatibility (qa_module uses it)
from .github_utils import (
    parse_github_url,
    fetch_repo_tree,
    get_file_content,
)
from .summar import (
    SECTION_DEFINITIONS,
    get_summary,
    generate_or_get_summary,
)
from .qa_module import answer_question

# Optional direct imports (not strictly necessary, but clarify modularization)
from . import retrieval  # noqa: F401
from . import repo_map as repo_map_utils  # avoid shadowing later variables

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

# SECTION_DEFINITIONS now imported from summar module


# Summary cache helpers now provided by summar module


# --- GitHub API Functions ---
# (Now imported from github_utils; previous inline implementations removed to avoid duplication.)


# --- LLM Interaction Functions (No changes here) ---
## get_important_files_by_category now provided by summar module


## generate_documentation now provided by summar module


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
                documentation, cached_commit, is_cached = generate_or_get_summary(
                    repo_url, force=force_resummarize
                )
                if documentation is None:
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

    try:
        result = answer_question(
            question,
            repo_url,
            persist_dir=CHROMA_PERSIST_DIR,
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_mode=embedding_mode,
            google_api_key=GOOGLE_API_KEY,
        )
        # Normalize for backward response shape
        if result.get("status") == "success":
            return JsonResponse(result)
        else:
            return JsonResponse(result)
    except Exception as e:  # noqa: BLE001
        return JsonResponse(
            {
                "answer": f"An error occurred while processing your question: {e}",
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
