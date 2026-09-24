#!/usr/bin/env python3
"""
UserPromptSubmit hook: classifies the incoming prompt as a "question", an
"issue", a "pr", or "other" via a TypeSafe Choice question (see classifiers.py), then
injects additionalContext nudging the session toward the matching lane of the
git-workflow pipeline (or a direct answer, for questions). "other" (thanks,
acknowledgments, status reports) injects nothing. It never performs
the GitHub actions itself — git-workflow's own pipeline still owns opening
issues/PRs, this just tells it which lane applies.

Low-confidence classifications inject nothing: a missing nudge is harmless, a
wrong one steers the session the wrong way. The threshold defaults to
MIN_CONFIDENCE and can be overridden with PROMPT_CLASSIFIER_MIN_CONFIDENCE;
use eval/run_eval.py to pick a value from real data.

Recursion guard: the old backend (and the eval harness's claude backend) run
`claude -p`, which would otherwise re-trigger this hook. Those calls set
ROUTE_PROMPT_ACTIVE, which is checked first thing on entry.

Fails soft: any error (missing API key, HTTP error, timeout,
malformed response, ...) is logged to stderr and results in no
additionalContext being injected — the prompt just passes through
unclassified rather than blocking.
"""
import json
import os
import sys

from classifiers import classify_typesafe

MIN_CONFIDENCE = 0.5  # untuned starting point; see eval/run_eval.py

CONTEXT_BY_CATEGORY = {
    "question": (
        "[prompt-classifier] This prompt looks like a question. Answer it directly — "
        "no GitHub issue or PR is needed unless the answer itself reveals one is warranted."
    ),
    "issue": (
        "[prompt-classifier] This prompt looks like a new bug report or feature request. "
        "Follow the issue-driven workflow: clarify if needed, then open a GitHub issue "
        "capturing it before making any code changes."
    ),
    "pr": (
        "[prompt-classifier] This prompt looks like a request to implement or fix something. "
        "Follow the issue-driven workflow: make sure an issue exists (open one first if not), "
        "then branch, implement, and open a PR referencing it."
    ),
}


def eprint(*args):
    print(*args, file=sys.stderr)


def min_confidence():
    raw = os.environ.get("PROMPT_CLASSIFIER_MIN_CONFIDENCE")
    if raw is None:
        return MIN_CONFIDENCE
    try:
        return float(raw)
    except ValueError:
        eprint(f"route_prompt: ignoring bad PROMPT_CLASSIFIER_MIN_CONFIDENCE {raw!r}")
        return MIN_CONFIDENCE


def main():
    if os.environ.get("ROUTE_PROMPT_ACTIVE"):
        return  # nested classification call — skip, avoid recursion

    payload = json.load(sys.stdin)
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        return

    result = classify_typesafe(prompt)
    if result.category not in CONTEXT_BY_CATEGORY:
        return  # "other": nothing to steer
    if result.confidence < min_confidence():
        eprint(f"route_prompt: {result.category} at confidence "
               f"{result.confidence:.2f} is below threshold; not nudging")
        return

    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": CONTEXT_BY_CATEGORY[result.category],
        },
    }))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        eprint(f"route_prompt: unexpected error: {e}")
        sys.exit(0)
