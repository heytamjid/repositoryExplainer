"""Agent loop logic extracted from views for clarity.

Provides `run_agentic_lookups` which performs iterative planning + lookups
and returns (updated_context, lookup_trace, iterations_used).
"""

from __future__ import annotations
from typing import List, Dict, Tuple
import re
import json

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from . import repo_map as repo_map_utils

LLM_MODEL_NAME = "gemini-2.5-flash"
MAX_LOOKUP_ITERS = 2


def _json_safe_parse(txt: str):
    try:
        match = re.search(r"\{(?:.|\n)*\}", txt)
        if match:
            return json.loads(match.group(0))
    except Exception:  # noqa: BLE001
        return None
    return None


def run_agentic_lookups(
    question: str,
    context: str,
    repo_url: str,
    repo_map_summary: str,
    persist_dir: str = "chroma_persist",
    google_api_key: str | None = None,
) -> Tuple[str, List[Dict], int]:
    """Execute the iterative lookup planning / execution loop.

    Returns: (new_context, trace, iterations)
    """
    lookup_trace: List[Dict] = []
    lookup_iterations = 0

    repo_map_data = repo_map_utils.get_repo_map(repo_url, persist_dir=persist_dir)
    if not repo_map_data:
        print("[AGENT] No repository map found. Attempting on-demand generation...")
        repo_map_data = repo_map_utils.build_repo_map(repo_url, persist_dir=persist_dir)
        if repo_map_data:
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
            return context, lookup_trace, lookup_iterations

    while lookup_iterations < MAX_LOOKUP_ITERS:
        lookup_iterations += 1
        print(f"[AGENT] Lookup iteration {lookup_iterations} planning started")
        planner_llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL_NAME, temperature=0.1, google_api_key=google_api_key
        )
        plan_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a repository Q&A planner. You receive: (1) user question, (2) current assembled context, (3) repository map listing files and unit names. Decide if MORE source code is required to answer confidently. If yes, choose up to 5 precise lookup requests.\nRules: output ONLY valid JSON object with keys: need_lookups (bool), reason (string), lookups (array). Each lookup item: {{type}}:'file'|'function', path:'relative/path.py', name:'function_or_class_name' (omit for file). Only pick functions that exist in the map. If existing context already shows the needed code, set need_lookups=false. Avoid speculative guesses.",
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
        preview = plan_raw[:500].replace("\n", " ")
        print(f"[AGENT] Raw plan response (truncated 500 chars): {preview}")
        plan = _json_safe_parse(plan_raw) or {}
        trace_entry = {
            "iteration": lookup_iterations,
            "raw_plan_preview": plan_raw[:300] + ("..." if len(plan_raw) > 300 else ""),
            "parsed": plan,
        }
        if not plan or not plan.get("need_lookups"):
            if plan:
                print(
                    f"[AGENT] Planner decided no lookups needed. Reason: {plan.get('reason')}"
                )
                trace_entry.update(
                    {"status": "no_lookups_needed", "reason": plan.get("reason")}
                )
            else:
                print(
                    "[AGENT] Planner parse failed or returned empty; stopping lookup loop"
                )
                trace_entry.update({"status": "plan_parse_failed"})
            lookup_trace.append(trace_entry)
            break
        lookups = plan.get("lookups", [])
        if not isinstance(lookups, list) or len(lookups) == 0:
            print(
                "[AGENT] Planner indicated lookups needed but provided none; aborting further iterations"
            )
            trace_entry.update(
                {"status": "empty_lookup_list", "reason": plan.get("reason")}
            )
            lookup_trace.append(trace_entry)
            break
        print(f"[AGENT] Planner requests {len(lookups)} lookups: {lookups}")
        fetched = repo_map_utils.lookup_repo_items(
            repo_url, lookups, persist_dir=persist_dir
        )
        if not fetched:
            print("[AGENT] Lookup fetch returned zero items; stopping iterations")
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
        trace_entry.update(
            {
                "status": "lookups_fetched",
                "requested": lookups,
                "fetched_count": len(fetched),
            }
        )
        lookup_trace.append(trace_entry)

    return context, lookup_trace, lookup_iterations


__all__ = ["run_agentic_lookups"]
