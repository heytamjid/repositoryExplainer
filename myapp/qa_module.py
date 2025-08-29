"""Question answering pipeline module.

Provides a high-level function `answer_question` that:
 1. Validates repository indexing status.
 2. Performs advanced retrieval via embedder.
 3. Builds context including retrieved docs, cached summary, and repo map summary.
 4. Optionally performs agentic on-demand lookups (via agent.run_agentic_lookups).
 5. Generates final answer with LLM and returns structured response dict.
"""

from __future__ import annotations

from typing import Dict, Any
import os
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from . import embedder  # still used for get_indexing_status (could later move)
from .retrieval import query_repository_advanced
from .repo_map import get_repo_map_summary
from .agent import run_agentic_lookups
from .summar import get_summary, SECTION_DEFINITIONS

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
LLM_MODEL_NAME = "gemini-2.5-flash"


def _assemble_summary_markdown(cached_summary: Dict[str, Any] | None) -> str | None:
    if not cached_summary or not isinstance(cached_summary.get("summary"), dict):
        return None
    summary_dict = cached_summary["summary"]
    parts = []
    for sid, content in summary_dict.items():
        title = next((s["title"] for s in SECTION_DEFINITIONS if s["id"] == sid), sid)
        parts.append(f"## {title}\n{content}\n")
    return "\n".join(parts)


def answer_question(
    question: str,
    repo_url: str,
    persist_dir: str = "chroma_persist",
    collection_name: str = "repo_functions",
    embedding_mode: str | None = None,
    google_api_key: str | None = None,
) -> Dict[str, Any]:
    google_api_key = google_api_key or GOOGLE_API_KEY

    repo_status_data = embedder.get_indexing_status(repo_url, persist_dir=persist_dir)
    repo_status = repo_status_data.get("status")
    if repo_status == "running":
        return {
            "status": "indexing_in_progress",
            "answer": "Repository is currently being indexed. Please try again in a few moments.",
        }
    if repo_status != "done":
        return {
            "status": "not_indexed",
            "answer": "This repository has not been indexed yet. Please index it first.",
        }

    adv = query_repository_advanced(
        question,
        repo_url,
        persist_dir=persist_dir,
        collection_name=collection_name,
        embedding_mode=embedding_mode,
    )
    retrieved_docs = adv.get("documents", [])
    classification = adv.get("classification")
    if not retrieved_docs:
        return {
            "status": "no_results",
            "answer": "I could not find relevant information in the repository to answer this question.",
            "classification": classification,
        }

    context = f"Query classification: {classification}\n\nContext from repository grouped by file:\n"
    for doc in retrieved_docs:
        m = doc["metadata"]
        context += (
            f"=== FILE: {m['file_path']} (Lines: {m['start_line']}-{m['end_line']}) ===\n"
            + doc["document"]
            + "\n\n"
        )

    cached_summary = get_summary(repo_url)
    summary_markdown = _assemble_summary_markdown(cached_summary)
    if summary_markdown:
        context += "\n\n=== REPOSITORY SUMMARY (Precomputed) ===\n" + summary_markdown

    repo_map_summary = get_repo_map_summary(repo_url, persist_dir=persist_dir)
    if repo_map_summary:
        context += (
            "\n\n=== REPOSITORY MAP (Files & Units Overview) ===\n"
            + repo_map_summary
            + "\n"
        )

    context, lookup_trace, lookup_iterations = run_agentic_lookups(
        question,
        context,
        repo_url,
        repo_map_summary or "",
        persist_dir=persist_dir,
        google_api_key=google_api_key,
    )

    final_llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME, google_api_key=google_api_key, temperature=0.4
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
    chain = answer_prompt | final_llm | StrOutputParser()
    answer = chain.invoke({"question": question, "context": context})

    if lookup_trace:
        trace_lines = ["\n---\n**Agent Lookup Trace**"]
        for t in lookup_trace:
            status = t.get("status")
            itn = t.get("iteration")
            if status == "no_repo_map_available":
                line = "- (init) No repository map available; planner skipped."
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
        answer += "\n" + "\n".join(trace_lines)

    return {
        "status": "success",
        "answer": answer,
        "embedding_mode": repo_status_data.get("embedding_mode", "unknown"),
        "num_results": len(retrieved_docs),
        "classification": classification,
        "lookup_iterations": lookup_iterations,
        "lookup_trace": lookup_trace,
    }


__all__ = ["answer_question"]
