#!/usr/bin/env python3
"""
Compare prompt-classifier backends on a labeled prompt set.

    python3 eval/run_eval.py                     # both backends, all prompts
    python3 eval/run_eval.py --backends typesafe # one backend
    python3 eval/run_eval.py --out results.json  # also dump raw per-prompt results

Reports per backend: accuracy, confusion matrix, latency (p50/p95/mean), errors
and mismatches; for TypeSafe also an accuracy-vs-coverage curve over confidence
thresholds, to help pick PROMPT_CLASSIFIER_MIN_CONFIDENCE. Then agreement
between backends.

Runs sequentially by default so latency numbers aren't distorted by
concurrency; use --workers to speed up (latency is then only indicative).
The typesafe backend needs the key in CLAUDE_PLUGIN_OPTION_TYPESAFE_API_KEY (the
variable Claude Code sets for hooks); the claude backend needs `claude`
on PATH.
"""
import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "hooks"))
import classifiers  # noqa: E402

LABELS = list(classifiers.CATEGORIES)
THRESHOLDS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def load_dataset(path, limit):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    for row in rows:
        if row["label"] not in LABELS:
            sys.exit(f"bad label {row['label']!r} in {path}")
    return rows[:limit] if limit else rows


def backend_fn(name):
    if name == "typesafe":
        if not os.environ.get(classifiers.API_KEY_ENV):
            return None, f"{classifiers.API_KEY_ENV} is not set"
        return classifiers.classify_typesafe, None
    if name == "claude":
        return lambda p: classifiers.classify_claude(p, cwd=str(HERE)), None
    sys.exit(f"unknown backend {name!r}")


def run_backend(fn, rows, workers):
    def one(row):
        start = time.perf_counter()
        try:
            res = fn(row["prompt"])
            return {"result": res, "error": None, "seconds": time.perf_counter() - start}
        except Exception as e:  # noqa: BLE001 - record and continue
            return {"result": None, "error": str(e), "seconds": time.perf_counter() - start}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, rows))


def pct(x, n):
    return f"{100 * x / n:5.1f}%" if n else "  n/a"


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def report(name, rows, outs):
    n = len(rows)
    answered = [(r, o) for r, o in zip(rows, outs) if o["result"]]
    correct = [(r, o) for r, o in answered if o["result"].category == r["label"]]
    errors = n - len(answered)

    print(f"\n=== {name} ===")
    print(f"answered {len(answered)}/{n}  errors {errors}")
    print(f"accuracy  overall {pct(len(correct), n)}  (errors count as wrong)"
          f"   answered-only {pct(len(correct), len(answered))}")
    for tier, hard in (("clear", False), ("hard", True)):
        tier_rows = [(r, o) for r, o in zip(rows, outs) if r["hard"] == hard]
        ok = sum(1 for r, o in tier_rows if o["result"] and o["result"].category == r["label"])
        print(f"          {tier:5} {pct(ok, len(tier_rows))}  ({ok}/{len(tier_rows)})")

    secs = [o["seconds"] for o in outs if o["result"]]
    if secs:
        print(f"latency   p50 {percentile(secs, .5):.2f}s  p95 {percentile(secs, .95):.2f}s"
              f"  mean {statistics.mean(secs):.2f}s")

    print("confusion (rows = label, cols = predicted):")
    print("  " + " " * 10 + "".join(f"{p:>10}" for p in LABELS))
    for label in LABELS:
        counts = [sum(1 for r, o in answered
                      if r["label"] == label and o["result"].category == p) for p in LABELS]
        print(f"  {label:<10}" + "".join(f"{c:>10}" for c in counts))

    print("mismatches:")
    shown = False
    for r, o in zip(rows, outs):
        if o["result"] and o["result"].category != r["label"]:
            conf = o["result"].confidence
            conf_s = f" conf={conf:.2f}" if conf is not None else ""
            hard = " [hard]" if r["hard"] else ""
            print(f"  {r['label']}->{o['result'].category}{conf_s}{hard}  {r['prompt'][:80]}")
            shown = True
    for r, o in zip(rows, outs):
        if o["error"]:
            print(f"  ERROR  {o['error'][:100]}  ({r['prompt'][:50]})")
            shown = True
    if not shown:
        print("  none")

    if any(o["result"] and o["result"].confidence is not None for o in outs):
        print("confidence threshold (nudge only if confidence >= t):")
        print("  t     coverage  accuracy-when-nudged")
        for t in THRESHOLDS:
            kept = [(r, o) for r, o in answered if o["result"].confidence >= t]
            ok = sum(1 for r, o in kept if o["result"].category == r["label"])
            print(f"  {t:.1f}   {pct(len(kept), n)}   {pct(ok, len(kept))}")


def report_agreement(rows, all_outs):
    names = list(all_outs)
    if len(names) < 2:
        return
    a, b = names[:2]
    both = [(r, x, y) for r, x, y in zip(rows, all_outs[a], all_outs[b])
            if x["result"] and y["result"]]
    agree = sum(1 for _, x, y in both if x["result"].category == y["result"].category)
    print(f"\n=== agreement {a} vs {b} ===")
    print(f"same label on {agree}/{len(both)} prompts")
    for r, x, y in both:
        if x["result"].category != y["result"].category:
            print(f"  label={r['label']} {a}={x['result'].category} {b}={y['result'].category}"
                  f"  {r['prompt'][:70]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--backends", default="claude,typesafe",
                    help="comma-separated: claude, typesafe (default: both)")
    ap.add_argument("--dataset", default=str(HERE / "prompts.jsonl"))
    ap.add_argument("--limit", type=int, help="only the first N prompts")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", help="write raw per-prompt results to this JSON file")
    args = ap.parse_args()

    rows = load_dataset(args.dataset, args.limit)
    print(f"{len(rows)} prompts from {args.dataset}")

    all_outs = {}
    for name in args.backends.split(","):
        fn, skip_reason = backend_fn(name.strip())
        if fn is None:
            print(f"\nskipping {name}: {skip_reason}")
            continue
        print(f"\nrunning {name} ...", flush=True)
        all_outs[name] = run_backend(fn, rows, args.workers)
        report(name, rows, all_outs[name])
    if not all_outs:
        sys.exit("no backend could run")
    report_agreement(rows, all_outs)

    if args.out:
        dump = {
            name: [
                {"prompt": r["prompt"], "label": r["label"], "hard": r["hard"],
                 "category": o["result"].category if o["result"] else None,
                 "confidence": o["result"].confidence if o["result"] else None,
                 "probabilities": o["result"].probabilities if o["result"] else None,
                 "error": o["error"], "seconds": round(o["seconds"], 3)}
                for r, o in zip(rows, outs)
            ]
            for name, outs in all_outs.items()
        }
        Path(args.out).write_text(json.dumps(dump, indent=2))
        print(f"\nraw results written to {args.out}")


if __name__ == "__main__":
    main()
