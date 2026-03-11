"""Django view layer for the Repository Explainer application.

This module exposes:
    - Web views (home, ask_question) for rendering HTML templates.
    - JSON API endpoints (api_ask, api_embedding_config) for:
            * Repository indexing lifecycle (start, status, clear)
            * Question answering over a codebase using vector search + LLM
            * Embedding configuration (local vs remote modes, diagnostics)
    - Integration glue between GitHub utilities, summarization, embedding, and QA modules.

The goal is to keep business / retrieval / summarization logic in their own modules
(`embedder`, `summar`, `qa_module`, etc.) and let this file focus on HTTP orchestration
and response shaping. Comments are added inline to clarify intent of each block.
"""

import os  # Environment variable access for API keys & runtime config
import json  # JSON request/response handling
import threading  # For background indexing without blocking HTTP request

from django.shortcuts import render  # Template rendering
from django.http import JsonResponse, HttpResponseBadRequest  # HTTP responses
from django.views.decorators.csrf import (
    csrf_exempt,
)  # Allow API POSTs without CSRF token


from . import embedder, config
from .summar import (
    generate_or_get_summary,
)

# Core QA pipeline (retrieval + LLM synthesis)
from .qa_module import answer_question


# --- Django Views ---


def home(request):
    """Render landing page for repository summarization.

    GET: Display empty form and (optionally) current embedding engine info.
    POST: Accept a GitHub repository URL, trigger summary generation (or fetch from cache)
          and return structured documentation sections for display.

    The heavy lifting (tree fetch, summarization, caching) is delegated to
    `generate_or_get_summary` in the `summar` module, keeping view thin.
    """
    # Initialize default template context
    documentation = None
    error = None
    repo_url = ""
    cached_commit = None  # Commit SHA associated with cached summary (if any)
    is_cached = False  # Indicates whether summary was served from cache
    embedding_info = (
        embedder.get_embedding_info()
    )  # Surface current embedding mode/status

    if request.method == "POST":  # Form submission for new repository summarization
        repo_url = request.POST.get("repo_url")
        force_resummarize = request.POST.get("force_resummarize") == "true"

        if repo_url:
            if not config.GITHUB_TOKEN or not config.GOOGLE_API_KEY:
                # Fail early if runtime configuration incomplete
                error = "Server configuration error: API keys are missing."
            else:
                # Retrieve existing summary or compute new one (force flag overrides cache)
                documentation, cached_commit, is_cached = generate_or_get_summary(
                    repo_url, force=force_resummarize
                )
                if documentation is None:
                    # Upstream failure: invalid URL or network/API issue
                    error = (
                        "Could not fetch/generate the summary. Please check the URL."
                    )

    # Render HTML with full context (sections definitions drive dynamic template grouping)
    return render(
        request,
        "home.html",
        {
            "documentation": documentation,
            "error": error,
            "repo_url": repo_url,
            "sections": config.SECTION_DEFINITIONS,
            "is_cached": is_cached,
            "cached_commit": cached_commit,
            "embedding_info": embedding_info,
        },
    )


def ask_question(request):
    """Render the question-answering interface.

    This view does not perform retrieval/LLM work directly— it just
    passes along any pre-selected repo URL (query param) and embedding
    system metadata so the front-end can display status and send AJAX
    requests to `api_ask`.
    """
    repo_url = request.GET.get(
        "repo_url", ""
    )  # Prefill if user arrived from summary page
    embedding_info = (
        embedder.get_embedding_info()
    )  # Expose current embedding mode/settings
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
    """Unified API endpoint for config retrieval, indexing control, and QA.

    Accepts JSON POST with fields:
      - action: one of
          * 'ask' (default) – answer a question over the repo
          * 'start_indexing' – begin background embedding & vector store build
          * 'status' – poll current indexing state
          * 'get_config' – fetch embedding configuration/metadata
      - repo_url: GitHub repository URL (required except for 'get_config'/'status')
      - question: Natural language query (required for 'ask')
      - embedding_mode: Optional override (e.g., 'local' or 'remote')
    """
    if request.method != "POST":
        return HttpResponseBadRequest("POST required")

    # --- Parse & validate inbound JSON payload ---
    try:
        body = json.loads(request.body.decode("utf-8"))
        question = body.get("question", "").strip()
        repo_url = body.get("repo_url", "").strip()
        action = body.get("action", "ask")  # Fallback to standard QA workflow
        # Optional override per request
        embedding_mode = body.get("embedding_mode")
    except json.JSONDecodeError:
        return HttpResponseBadRequest("Invalid JSON")

    if not repo_url and action not in ("get_config", "status"):
        # 'ask' and 'start_indexing' require a repository context
        return JsonResponse({"answer": "A repository URL is required."})

    # --- Configuration discovery (front-end bootstrap) ---
    if action == "get_config":
        return JsonResponse(
            {
                "embedding_info": embedder.get_embedding_info(),
                "status": "config_retrieved",
            }
        )

    # --- Poll for existing indexing lifecycle status ---
    if action == "status":
        repo_status_data = embedder.get_indexing_status(
            repo_url, persist_dir=config.CHROMA_PERSIST_DIR
        )
        return JsonResponse(
            {
                "status": repo_status_data.get("status", "not_indexed"),
                "embedding_mode": repo_status_data.get(
                    "embedding_mode", embedder.config.EMBEDDING_MODE
                ),
            }
        )

    # --- Kick off asynchronous repository indexing ---
    if action == "start_indexing":
        print(f"Received request to start indexing for {repo_url}")
        if embedding_mode:
            print(f"Using embedding mode: {embedding_mode}")

        def _start_index_task():  # Background worker closure
            print(f"Starting indexing thread for {repo_url}")
            try:
                res = embedder.index_repository(
                    repo_url,
                    persist_dir=config.CHROMA_PERSIST_DIR,
                    collection_name=config.CHROMA_COLLECTION_NAME,
                    embedding_mode=embedding_mode,
                )
                print("Background indexing result:", res)
            except Exception as e:  # Log; surface errors later via status polling
                print(f"Indexing thread for {repo_url} failed: {e}")

        threading.Thread(target=_start_index_task, daemon=True).start()
        return JsonResponse(
            {
                "status": "indexing_started",
                "embedding_mode": embedding_mode or config.EMBEDDING_MODE,
            }
        )

    # --- Question answering flow ---
    if not question:
        return JsonResponse({"answer": "A question is required."})

    # Allow lightweight reset via explicit command-like phrases only.
    # Previous substring matching ("clear" in question) caused false positives
    # for legitimate questions containing those words.
    _q_stripped = question.lower().strip()
    if _q_stripped in ("clear", "reset", "clear index", "reset index", "clear indexing", "reset indexing"):
        embedder.clear_indexing_state(repo_url, persist_dir=config.CHROMA_PERSIST_DIR)
        return JsonResponse(
            {
                "answer": f"Indexing state cleared for {repo_url}. You can now ask questions again."
            }
        )

    try:
        # Delegate retrieval + answer synthesis to QA module
        result = answer_question(
            question,
            repo_url,
            persist_dir=config.CHROMA_PERSIST_DIR,
            collection_name=config.CHROMA_COLLECTION_NAME,
            embedding_mode=embedding_mode,
            google_api_key=config.GOOGLE_API_KEY,
        )
        # Response already normalized by qa_module; return directly
        return JsonResponse(result)
    except Exception as e:  # noqa: BLE001
        # Defensive catch-all to avoid exposing traceback to client
        return JsonResponse(
            {
                "answer": f"An error occurred while processing your question: {e}",
                "status": "error",
            }
        )


@csrf_exempt
def api_embedding_config(request):
    """Manage and diagnose embedding configuration.

    GET: Return current embedding engine metadata (mode, provider, dimensions if known).
    POST: Accepts JSON with an 'action' key:
        * set_mode   – Switch between 'local' and 'remote' embedding backends.
        * test_local – Attempt to generate a trivial embedding using the local model.
        * test_remote – Same, but invoking remote provider / API.

    Returns structured JSON suitable for UI control panels.
    """
    if request.method == "GET":
        return JsonResponse(embedder.get_embedding_info())

    elif request.method == "POST":
        try:
            body = json.loads(request.body.decode("utf-8"))
            action = body.get("action")

            # Hot-switch embedding backend (non-persistent)
            if action == "set_mode":
                new_mode = body.get("mode")
                if new_mode in ["local", "remote"]:
                    # Process-level override: update both env and config module
                    os.environ["EMBEDDING_MODE"] = new_mode
                    config.EMBEDDING_MODE = new_mode
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

            elif action == "test_local":  # Smoke test the local model pipeline
                try:
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

            elif action == "test_remote":  # Smoke test remote API reachability/contract
                try:
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

            else:  # Fallback for unsupported actions
                return JsonResponse(
                    {
                        "status": "error",
                        "message": "Invalid action. Use 'set_mode', 'test_local', or 'test_remote'.",
                    }
                )

        except json.JSONDecodeError:
            return HttpResponseBadRequest("Invalid JSON")

    else:  # Reject non-GET/POST verbs
        return HttpResponseBadRequest("GET or POST required")
