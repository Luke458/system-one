"""Compare batched next-token scoring with greedy prompt-only JSON generation."""
import argparse
import json
from pathlib import Path
import platform
import statistics
import time

import torch
import transformers
from .engine import DecisionModel, schema_valid


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(device, fn):
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    result = fn()
    synchronize(device)
    ms = (time.perf_counter() - start) * 1000
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return result, ms, peak


def summary(rows, method):
    selected = [r for r in rows if r["method"] == method]
    times = sorted(r["latency_ms"] for r in selected)
    return {"n": len(times), "median_ms": statistics.median(times),
            "p95_ms": times[max(0, __import__('math').ceil(.95 * len(times)) - 1)],
            "schema_rate": sum(r["schema_valid"] for r in selected) / len(selected),
            "exact_match_rate": sum(r["exact_match"] for r in selected) / len(selected),
            "field_accuracy": sum(r["correct_fields"] for r in selected) / sum(r["field_count"] for r in selected)}


def main():
    from .qc import CASES, FIELDS
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openbmb/MiniCPM5-2B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device", default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", default="reports/benchmark.json")
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0 or args.max_new_tokens < 1:
        parser.error("repeats/max-new-tokens must be positive and warmups nonnegative")
    model = DecisionModel.from_pretrained(args.model, revision=args.revision, device=args.device)
    methods = {
        "decide": lambda c: model.decide(c, FIELDS),
        "generate": lambda c: model.generate_structured(c, FIELDS, args.max_new_tokens),
    }
    for _ in range(args.warmups):
        for fn in methods.values():
            timed(model.device, lambda: fn(CASES[0]["context"]))
    rows = []
    for repeat in range(args.repeats):
        for index, case in enumerate(CASES):
            # Alternate order to reduce systematic thermal/order bias.
            for method in (list(methods) if (repeat + index) % 2 == 0 else list(reversed(methods))):
                result, ms, peak = timed(model.device, lambda: methods[method](case["context"]))
                details, raw, output_tokens, parse_error = None, None, 0, None
                if method == "decide":
                    value = {k: v.value for k, v in result.items()}
                    details = {k: v.to_dict() for k, v in result.items()}
                    lengths = [len(model.tokenizer.encode(model.field_prompt(case["context"], f)[0], add_special_tokens=False)) for f in FIELDS.values()]
                else:
                    raw, output_tokens = result
                    try:
                        value = json.loads(raw)
                    except (ValueError, TypeError) as exc:
                        value, parse_error = None, str(exc)
                    lengths = [len(model.tokenizer.encode(model.generation_prompt(case["context"], FIELDS), add_special_tokens=False))]
                valid = schema_valid(value, FIELDS)
                correct = sum(type(value.get(k)) is type(v) and value.get(k) == v for k, v in case["expected"].items()) if type(value) is dict else 0
                rows.append({"case": case["id"], "repeat": repeat, "method": method,
                             "latency_ms": ms, "peak_allocated_bytes": peak,
                             "input_tokens": lengths, "output_tokens": output_tokens,
                             "schema_valid": valid, "exact_match": valid and value == case["expected"],
                             "correct_fields": correct, "field_count": len(FIELDS),
                             "value": value, "raw_text": raw, "parse_error": parse_error, "decisions": details})
                print(f"{method:8} {case['id']:8} {ms:8.1f} ms schema={valid}", flush=True)
    report = {"model": model.model_id, "revision": model.revision,
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "transformers": transformers.__version__, "hip": torch.version.hip,
                              "device": str(model.device), "gpu": torch.cuda.get_device_name(model.device) if model.device.type == "cuda" else None,
                              "dtype": str(next(model.model.parameters()).dtype), "attention": "sdpa"},
              "settings": vars(args), "dataset": CASES,
              "notes": ["Synthetic smoke benchmark, not evidence of production accuracy/calibration.",
                        "End-to-end warm latency includes prompt/tokenization/validation/transfers; excludes model load.",
                        "decide repeats context per field; no shared KV prefix in Phase 1.",
                        "Generation is ordinary prompt-only greedy JSON, not grammar-constrained; no retries/repair.",
                        "Decision schema validity is guaranteed by construction; semantic correctness is separate."],
              "summary": {m: summary(rows, m) for m in methods}, "rows": rows}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))

if __name__ == "__main__":
    main()
