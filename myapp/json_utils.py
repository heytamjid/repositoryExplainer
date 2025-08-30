"""JSON parsing utility functions."""

import json
import re
from typing import Any, Dict


def extract_json_candidates(txt: str):
    """Yield candidate JSON substrings (fenced blocks first, then brace-matched)."""
    fenced = re.findall(r"```(?:json)?\n(.*?)```", txt, re.DOTALL | re.IGNORECASE)
    if fenced:
        for f in fenced:
            yield f.strip()
    # Fallback: naive brace stack scanning for first complete top-level object
    stack = []
    start = None
    for i, ch in enumerate(txt):
        if ch == "{":
            if start is None:
                start = i
            stack.append(ch)
        elif ch == "}" and stack:
            stack.pop()
            if not stack and start is not None:  # Completed a JSON object candidate
                candidate = txt[start : i + 1]
                yield candidate.strip()
                start = None


def clean_json(txt: str) -> str:
    """Attempt minor repairs: dangling commas & single quotes -> double quotes."""
    txt = re.sub(r",\s*(\]|})", r"\1", txt)  # Remove trailing commas
    # Convert object keys/values in single quotes to JSON-compliant double quotes
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


def parse_identified(raw: str) -> Dict[str, Any] | None:
    """Return first successfully decoded dict from candidate JSON segments."""
    tried = set()
    for cand in extract_json_candidates(raw):
        if cand in tried:
            continue
        tried.add(cand)
        for attempt in range(2):  # Raw first, then cleaned
            attempt_txt = cand if attempt == 0 else clean_json(cand)
            try:
                obj = json.loads(attempt_txt)
                if isinstance(obj, dict):
                    return obj
            except Exception:  # noqa: BLE001 – tolerate and continue
                continue
    return None
