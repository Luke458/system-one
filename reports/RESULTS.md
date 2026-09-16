# Measured local smoke results

Measured 2026-09-16 on AMD Radeon RX 9070 XT, bfloat16, SDPA. Checkpoint: `12a3808a956f869c767195e9266b59c4d21d92e2`. Torch `2.13.0+rocm10.0.0`, Transformers `5.8.0`. Four unique synthetic QC cases, three repeats each, three fields per case, one warmup per method. Repeats are not additional accuracy samples.

| Measure | Typed logits | Ordinary greedy JSON |
|---|---:|---:|
| Median end-to-end latency | 36.46 ms | 600.11 ms |
| p95 latency (12 measurements) | 38.21 ms | 620.97 ms |
| Valid output schema | 100% | 100% |
| Exact case accuracy | 25% | 50% |
| Per-field accuracy | 58.3% | 50.0% |

The median ratio is **16.5×** on this tiny warm workload. It does not establish a general speed advantage, especially for longer contexts or optimized generation backends. Classification returns valid types by construction; both paths can return wrong answers.

The clearest failure: the missing-stores case selected `insufficient_evidence` with roughly 97.5% conditional confidence, despite explicit ingestion failures in the input. The generation baseline also missed this case. No prompts were tuned to these cases after observing results. The engine is operational, but this checkpoint/prompt combination is not ready for dependable QC automation.

## Numerical verification

Pinned full-checkpoint checks passed: batched and singleton decisions agree on the first QC case; eager and SDPA attention select the same answers; greedy first-token generation returns `D`, matching the cause scorer. All 26 label candidates pass exact-boundary validation. Maximum candidate-logit differences were 0.25 in bfloat16 across the checked execution paths. This is not exact score parity; near ties and calibration can be sensitive to execution settings. The separate tiny-Llama float32 batching test passes at 1e-6 absolute tolerance.

## Uncalibrated diagnostics

Only four unique synthetic cases per field; these are debugging diagnostics, **not calibration evidence**. No temperature was fitted to these examples.

| Field | Multiclass Brier | Top-label ECE |
|---|---:|---:|
| data_issue | 0.6572 | 0.3906 |
| cause | 0.9354 | 0.5732 |
| escalate | 0.5130 | 0.3289 |

`smoke-calibration.json` contains reliability-bin data. `synthetic-calibration.json` demonstrates temperature fitting and disjoint-split evaluation using artificial logits and labels in `examples/synthetic-*.json`; those scores are not model outputs and establish no model quality claim.

## Scope and next experiment

22 automated tests pass (see `tests.txt`). Raw prompts are reproducible from the code and fixed dataset; raw model answers, per-field distributions, token counts, device metadata and timing rows are in `benchmark.json`. Model path checks are in `model-checks.json`.

Before considering QC deployment, obtain representative labelled records and separately assess prompt design, label-order bias, non-thinking behavior, checkpoint variants, field consistency, and held-out calibration. Phase 2 cache work should preserve measured logits and be benchmarked at longer contexts. Temperature scaling cannot fix the observed wrong class rankings.
