# Phase 2: shared-prefix KV inference

Implemented as an opt-in execution strategy:

```python
result = model.decide(context, fields, execution="shared_prefix")
```

The default remains `execution="batched"`. Shared prefill can lose at short contexts or with one field. It does not improve semantic correctness or calibrate confidence.

## Implementation

1. Render complete field prompts with the existing chat template and thinking disabled. Validate candidate tokens at each exact prompt boundary. Tokenize full prompts and take their longest common **token** prefix. Leave at least one token in every suffix, including identical prompts and a singleton field.
2. Validate all prompt lengths before any forward pass. Prefill the common prefix once with `use_cache=True`. Reject an empty common prefix rather than silently changing execution mode.
3. Keep the resulting `DynamicCache` immutable within the call. Each suffix microbatch gets a deep copy and the supported `batch_repeat_interleave` operation. Prefix KV memory is materialized per row: this is shared **computation**, not zero-copy shared storage. Copies are included in latency/memory measurements.
4. Left-pad the suffixes after the cached prefix. Build an attention mask covering prefix, masked gap and real suffix tokens. Explicit logical position IDs count only unmasked tokens; physical cache length includes padding. Read final-position logits, which always correspond to a real suffix token.
5. Reuse Phase 1's candidate distributions and typed output builder. Discard all cache objects after the call. There is no global cache, cross-request reuse, eviction policy or retained user context.

The public API accepts ordinary Hugging Face causal LMs, but this execution backend currently requires full-attention `DynamicCache` models. Sliding-window, hybrid/recurrent, and shared-layer cache configurations are rejected. Llama and GPT-2 paths have small float32 tests; MiniCPM5-2B has GPU checks. Other architectures, versions and attention implementations are not certified. Models without `logits_to_keep` can use the full logits output but may consume considerably more memory.

## Verification

Tests compare full-prompt and cached candidate logits, probabilities and candidate mass at 2e-6 absolute tolerance on small float32 models. They cover one/many fields, identical and unequal suffix lengths, reordered fields, microbatch boundaries, empty/long context, repeated calls separated by an unrelated request, explicit overflow, and unsupported cache configuration. Existing candidate validation and schema tests remain in place. Full-model bfloat16 differences are measured rather than expected to match float32 tolerance; see `reports/PHASE2.md`.

Run the scaling benchmark:

```sh
OMP_NUM_THREADS=4 .venv/bin/python -m system_one.cache_benchmark \
  --context-tokens 128 1024 4096 --field-counts 1 4 16 --batch-size 4
```

It times complete warmed `decide` calls, alternates execution order, includes cache-copy and tokenization costs, and records median/p95, GPU peak allocated memory, actual prompt/prefix lengths, computed token positions, candidate-score differences and decision disagreements. Context and fields in the scaling matrix are synthetic; original QC cases are evaluated separately for path parity. Model loading is excluded. The token-position count is an accounting measure, not a FLOP or memory estimate.

## Remaining work

- Representative labelled task evaluation and independent calibration for each execution configuration.
- More context lengths (including 16k subject to memory), models, dtypes and field distributions; dedicated concurrent-use tests before serving.
- Measure immutable or paged prefix storage against materialized copies. Never treat writable `expand` views as safe caches.
- Evaluate LMCache/vLLM/SGLang integration only after backend needs and real workload justify it. No production cache infrastructure is reimplemented here.
- Runtime selection should use measured break-even data; no automatic heuristic is enabled yet.

Reference: [Transformers cache strategies](https://huggingface.co/docs/transformers/main/en/kv_cache). The implementation was also checked against the locally installed Transformers 5.8.0 cache API.
