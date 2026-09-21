"""
Classification backends for prompt-classifier. Each takes a prompt and returns a
Result; both are shared by the hook (route_prompt.py) and the comparison
harness (eval/run_eval.py) so they're measured exactly as they run.

- classify_typesafe: one Choice question to TypeSafe's Jev via /v1/systemone.
  Returns per-category probabilities and a confidence value.
- classify_claude: headless `claude -p` call to Haiku (the original backend).
  Returns a label only; the model gives no confidence.

Both are built from the same CATEGORIES descriptions so a comparison measures
the backend, not differences in wording.
"""
import json
import os
import re
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field

CATEGORIES = {
    "question": (
        "Can be answered directly, with no code change or GitHub issue needed. "
        "Includes asking whether something is feasible, asking for an assessment, "
        "explanation, or opinion, even when the message mentions code or says "
        "not to write it."
    ),
    "issue": (
        "Reports a bug, requests a feature, or describes a problem that should be "
        "tracked as a GitHub issue before any code changes happen."
    ),
    "pr": (
        "Asks to implement or fix something where the work should happen on a "
        "branch culminating in a pull request, including continuing "
        "already-scoped implementation work."
    ),
}

# Claude Code exports the plugin's `userConfig` values to hook processes as
# CLAUDE_PLUGIN_OPTION_<KEY>; see typesafe_api_key in plugin.json.
API_KEY_ENV = "CLAUDE_PLUGIN_OPTION_TYPESAFE_API_KEY"
TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"
TYPESAFE_MODEL = "jev-latest"
TYPESAFE_TIMEOUT = 10

CLAUDE_MODEL = "claude-haiku-4-5-20251001"
CLAUDE_TIMEOUT = 15


class ClassifierError(Exception):
    pass


@dataclass
class Result:
    category: str
    confidence: float | None = None
    probabilities: dict = field(default_factory=dict)


def classify_typesafe(prompt, timeout=TYPESAFE_TIMEOUT):
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise ClassifierError(
            f"no TypeSafe API key ({API_KEY_ENV} is unset); "
            "set typesafe_api_key when enabling the plugin"
        )

    body = json.dumps({
        "state": prompt,
        "model": TYPESAFE_MODEL,
        "questions": {
            "category": {
                "type": "choice",
                "instructions": (
                    "Which kind of request is this message to a coding assistant?"
                ),
                "criteria": CATEGORIES,
            },
        },
    }).encode()
    req = urllib.request.Request(
        TYPESAFE_URL, data=body, method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            payload = json.load(res)
    except urllib.error.HTTPError as e:
        # 401 bad key, 422 bad request, 429 rate limited, 529 overloaded.
        raise ClassifierError(f"typesafe HTTP {e.code}: {e.read()[:200]!r}") from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise ClassifierError(f"typesafe request failed: {e}") from e

    answer = payload["answers"]["category"]
    if answer["choice"] not in CATEGORIES:
        raise ClassifierError(f"unexpected category: {answer['choice']!r}")
    return Result(answer["choice"], answer["confidence"], answer["probabilities"])


def _claude_prompt(prompt):
    lines = "\n".join(f'- "{name}": {desc}' for name, desc in CATEGORIES.items())
    options = "|".join(CATEGORIES)
    return (
        "You are a fast prompt classifier for a coding assistant harness. "
        "Classify the user's message below into exactly one category:\n\n"
        f"{lines}\n\n"
        "Respond with ONLY a compact JSON object, no other text, no markdown "
        f'fences: {{"category": "{options}"}}\n\nUser message:\n{prompt}'
    )


def classify_claude(prompt, cwd=".", timeout=CLAUDE_TIMEOUT):
    # The child is itself a Claude Code session; the flag makes any installed
    # UserPromptSubmit hook (including route_prompt.py) skip it.
    env = dict(os.environ, ROUTE_PROMPT_ACTIVE="1")
    try:
        res = subprocess.run(
            ["claude", "-p", _claude_prompt(prompt),
             "--model", CLAUDE_MODEL, "--output-format", "json"],
            cwd=cwd, env=env, capture_output=True, text=True,
            timeout=timeout, stdin=subprocess.DEVNULL,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        raise ClassifierError(f"claude call failed: {e}") from e
    if res.returncode != 0:
        raise ClassifierError(f"claude exited {res.returncode}: {res.stderr[:300]}")

    try:
        result_text = json.loads(res.stdout).get("result", "")
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", result_text, re.DOTALL)
        category = json.loads(fence.group(1) if fence else result_text).get("category")
    except (json.JSONDecodeError, AttributeError) as e:
        raise ClassifierError(f"malformed claude output: {e}") from e
    if category not in CATEGORIES:
        raise ClassifierError(f"unexpected category: {category!r}")
    return Result(category)
