# Phase 2 results: shared-prefix inference

Implemented the next step in the supplied project direction: an opt-in shared-prefix execution path with an unchanged typed field API. Added explicit selected-class `probability`, `raw_probability`, and `probability_kind` metadata. Temperatures are described as scaling, not proof of empirical calibration. Models without `logits_to_keep` use a compatible full-logit forward.

## Verification

34 tests pass, including float32 cached/full-prompt parity on tiny Llama and GPT-2 models, unequal and identical suffixes, field reordering, microbatches and repeated requests. Local float32 candidate logit/probability tolerance is 2e-6. The full MiniCPM5-2B checkpoint was then measured on the RX 9070 XT with Torch 2.13.0+rocm10.0.0, Transformers 5.8.0 and bfloat16. Model revision: `12a3808a956f869c767195e9266b59c4d21d92e2`.

## Warm end-to-end latency

Three repeats per shape, one warmup per execution strategy per shape, batch size 4. Times include prompt/token validation, prefill, cache duplication, suffix forward, transfers and result construction. Cache state is rebuilt on every call. Speedup is batched median divided by shared-prefix median.

| Context tokens | Fields | Batched ms | Shared-prefix ms | Speedup |
|---:|---:|---:|---:|---:|
| 128 | 1 | 29.8 | 55.9 | 0.53× |
| 128 | 4 | 57.6 | 69.2 | 0.83× |
| 128 | 16 | 231.3 | 192.9 | 1.20× |
| 1024 | 1 | 79.2 | 104.8 | 0.76× |
| 1024 | 4 | 294.7 | 141.3 | 2.09× |
| 1024 | 16 | 1184.3 | 348.9 | 3.39× |
| 4096 | 1 | 314.8 | 340.0 | 0.93× |
| 4096 | 4 | 1474.1 | 452.6 | 3.26× |
| 4096 | 16 | 5936.4 | 928.2 | 6.40× |

The single-field calls lose because splitting prefill adds work without sharing it across fields. Multi-field/long-context calls benefit. These are synthetic repeated-field/filler workloads, not a production throughput study or an accuracy benchmark. No automatic execution selector was added; `batched` remains the default.

## Numerical differences

No selected-answer disagreements occurred across the measured scaling matrix or the four original QC cases. Bfloat16 scores are **not interchangeable** between execution paths: maximum QC probability differences were:

| QC case | Max absolute probability delta | Max candidate-logit delta |
|---|---:|---:|
| stores | 0.015211 | 0.250 |
| products | 0.062126 | 0.375 |
| market | 0.010099 | 0.250 |
| unknown | 0.060283 | 0.500 |

The largest probability shift is roughly 6.2 percentage points. This exceeds the float32 reference tolerance and matters for calibration and confidence thresholds even when argmax is unchanged. Near-tied decisions may change on other inputs. Fit and validate calibration for a fixed model revision, prompt, class order, dtype and execution mode. The original confidently wrong QC answers remain; caching is a performance feature, not an accuracy remedy.

## Scope

Cache copies have independent mutable storage and live only within a request. Prefix computation is shared; KV storage is materialized per field batch. Full-attention Transformers DynamicCache models are supported; sliding/hybrid and other cache classes are explicitly rejected. No production cache service, learned head, new calibration family, or escalation action was introduced.

The user-authored direction document remains unchanged. `docs/direction.md` separates implemented functionality from future levels. The next quality milestone is representative labelled evaluation, prompt/label-order auditing and separately validated calibration; the next performance milestone is broader model/context testing and assessing production cache backends.

Raw measurements: `cache-benchmark.json`. Tests: `phase2-tests.txt`. Implementation details: `../docs/phase-2-kv-cache.md`.
