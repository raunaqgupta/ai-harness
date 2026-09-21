# prompt-classifier

A Claude Code plugin that classifies every prompt as a **question**, an **issue**, or a **pr**, and nudges the session toward the matching lane of the [git-workflow](../git-workflow/README.md) pipeline — without performing any GitHub actions itself.

## What it does

It installs a `UserPromptSubmit` hook (`hooks/route_prompt.py`). On every prompt, the hook:

1. Sends the prompt to [TypeSafe](https://docs.typesafe.ai)'s `jev-latest` model as a single [Choice](https://docs.typesafe.ai/primitives/choice.md) question over `question` / `issue` / `pr` (`hooks/classifiers.py`). The response carries a probability per category and a confidence value.
2. If confidence is at or above the threshold, injects `additionalContext` matching the category:
   - **question** — answer directly, no issue/PR needed.
   - **issue** — clarify if needed, then open a GitHub issue before making any code changes.
   - **pr** — make sure an issue exists (open one first if not), then branch, implement, and open a PR referencing it.
3. Below the threshold it injects nothing: a missing nudge is harmless, a wrong one steers the session the wrong way.

The hook only classifies and nudges — it never calls `gh` itself. Opening the actual issue/PR is still owned by whatever pipeline is in effect (e.g. `git-workflow`'s injected instructions, or a repo's own `CLAUDE.md`).

## Confidence threshold

Defaults to `0.5` (`MIN_CONFIDENCE` in `route_prompt.py`), an untuned starting point. Override with the `PROMPT_CLASSIFIER_MIN_CONFIDENCE` environment variable. Run the [comparison harness](#comparison-harness) to see the coverage/accuracy trade-off on labeled prompts and pick a value.

## Privacy

Every prompt you submit is sent to TypeSafe's API (`https://api.typesafe.ai`) for classification. Do not install this plugin if that is not acceptable for the work you do in Claude Code.

## Requirements

- A TypeSafe API key (create one at [console.typesafe.ai](https://console.typesafe.ai/)). Claude Code asks for it when you enable the plugin — see [Setup](#setup).
- `python3` on `PATH`. No third-party Python packages — the hook uses the standard library's `urllib`.
- Claude Code v2.1.271 or later (plugin `userConfig` support).

## Install

```
claude plugin marketplace add raunaqgupta/ai-harness
claude plugin install prompt-classifier@ai-harness
```

Or point at a local checkout:

```
claude plugin marketplace add /path/to/ai-harness
claude plugin install prompt-classifier@ai-harness
```

## Setup

The plugin declares a `typesafe_api_key` option (`userConfig` in `plugin.json`, marked `sensitive`). Claude Code prompts for it when the plugin is enabled, masks the input, and stores it in secure storage (the macOS Keychain, falling back to `~/.claude/.credentials.json`) rather than in `settings.json`. It is passed to the hook as `CLAUDE_PLUGIN_OPTION_TYPESAFE_API_KEY`.

For a scripted install, supply it up front:

```
claude plugin install prompt-classifier@ai-harness --config typesafe_api_key=<your key>
```

The plugin reads only that option. It deliberately ignores a `TYPESAFE_API_KEY` environment variable, so a key used for other tools is never picked up by accident. Without a key the hook does nothing.

## Comparison harness

`eval/run_eval.py` runs a labeled prompt set (`eval/prompts.jsonl`) through both backends — TypeSafe and the original headless `claude -p` call to Haiku — and reports:

- accuracy overall and split into `clear` / `hard` (ambiguous) prompts
- confusion matrix, mismatches, and errors
- latency p50 / p95 / mean
- for TypeSafe, accuracy vs. coverage at each confidence threshold
- where the two backends disagree

```
python3 plugins/prompt-classifier/eval/run_eval.py                      # both backends
python3 plugins/prompt-classifier/eval/run_eval.py --backends typesafe  # one backend
python3 plugins/prompt-classifier/eval/run_eval.py --out results.json   # save raw results
```

It runs sequentially by default so latency isn't distorted by concurrency (`--workers N` speeds it up). Outside Claude Code there is no `userConfig` to supply the key, so export it as `CLAUDE_PLUGIN_OPTION_TYPESAFE_API_KEY` for the run. The `typesafe` backend is skipped with a message if that is unset; the `claude` backend needs the `claude` CLI on `PATH`.

The labels in `prompts.jsonl` are one person's judgment against the category definitions in `hooks/classifiers.py`; ambiguous prompts are marked `"hard": true`. Edit the file or add prompts from your own history — the more it resembles what you actually type, the more the numbers mean. Both backends are built from the same category descriptions so the comparison measures the backend, not the wording.

## Notes

- Fails soft: any error (missing key, HTTP 401/422/429/529, timeout, malformed response, ...) is logged to stderr and results in no `additionalContext` being injected — the prompt just passes through unclassified rather than blocking. There are no retries: the hook is on the critical path of every prompt.
- `ROUTE_PROMPT_ACTIVE` still short-circuits the hook. The hook itself no longer spawns Claude, but the harness's `claude` backend does, and the guard stops that nested session from triggering this hook.
- Adds one network round trip to every prompt. Pair with `git-workflow` for the actual issue/PR pipeline; on its own this plugin only classifies.
