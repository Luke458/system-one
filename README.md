# System One

An experimental semantic decision runtime for ordinary Hugging Face causal language models. Return typed Boolean/Choice distributions directly from logits, reuse context prefill across fields, or compile repeated fields into lightweight classifiers over a frozen model's hidden states.

MiniCPM5-2B is the first full checkpoint tested. This is not a reproduction of Jev, TypeSafe's architecture, or RLCD. The original zero-shot method is prompt-sensitive; all reported quality measurements are small synthetic experiments.

## Install

Python 3.11+ is required; development uses Python 3.12. Install the PyTorch build appropriate to your CPU/GPU first, then:

```sh
python -m venv .venv
. .venv/bin/activate
# CPU example; for ROCm use the compatible wheel index for your hardware.
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[test]'
OMP_NUM_THREADS=4 python -m pytest -q
```

[PyTorch installation instructions](https://pytorch.org/get-started/locally/). ROCm uses `torch.cuda` APIs. Tested GPU runtime: RX 9070 XT, Torch 2.13.0+rocm10.0.0, Transformers 5.8.0, bfloat16. No CUDA-only extensions are required. The `.venv`, model weights and fitted head artifacts are excluded from Git.

`bash scripts/setup.sh` supports `uv` and an explicit `TORCH_INDEX_URL`; it refuses to overwrite an existing environment. `scripts/use-existing-runtime.py` can create an overlay on an existing runtime when a fully separate environment is unnecessary.

## Dynamic decisions

```python
from system_one import Boolean, Choice, DecisionModel

model = DecisionModel.from_pretrained(
    "openbmb/MiniCPM5-2B",
    revision="12a3808a956f869c767195e9266b59c4d21d92e2",
)
fields = {
    "data_issue": Boolean("Does the evidence establish a data completeness issue?"),
    "cause": Choice("What is the best-supported cause?", (
        "missing_stores", "missing_products", "real_market_movement",
        "insufficient_evidence",
    )),
}
context = "Twenty expected stores failed to upload; product coverage is complete elsewhere."
result = model.decide(context, fields)
print(result["data_issue"].value)        # Python bool
print(result["cause"].probabilities)    # Canonical class order
```

Dynamic inference uses one next-token score per field, without generating or parsing JSON. The host constructs the typed result. Arbitrary class strings are mapped to distinct single-token candidates, validated at the exact prompt boundary. Boolean order is `(False, True)`; Choice accepts 2–26 unique nonempty strings. Failed inference or invalid schema raises an exception, rather than manufacturing an answer.

`Decision` exposes `value`, `values`, `probabilities`, `raw_probabilities`, `probability`, `raw_probability`, candidate `logits`, `token_ids`, `candidate_mass`, `temperature`, `probability_kind`, and `strategy`. Candidate mass measures unscaled full-vocabulary probability assigned to the allowed answer tokens; it does not measure correctness. Distribution records preserve Boolean types during JSON serialization with `to_dict()`.

Inputs exceeding `max_input_tokens` (4096 by default) fail without silent truncation. `field_batch_size` defaults to 8. Prompt instructions are not a security boundary against adversarial context.

## Shared-prefix execution

```python
result = model.decide(context, fields, execution="shared_prefix")
```

Prefills the exact common token prefix once and copies its KV cache into field microbatches. Caches live only within the call. This shares computation, not zero-copy storage. Currently supports full-attention Transformers DynamicCache models; sliding/hybrid caches are rejected. Tiny Llama/GPT-2 tests and full MiniCPM GPU measurements cover the implemented paths, not every Hugging Face architecture.

On the measured synthetic workload, 4,096 context tokens and 16 fields ran 6.4× faster than full-prompt batching. Single-field calls were slower. Bfloat16 probabilities differed across paths even where selected values agreed; calibrate for fixed execution settings. [Implementation and results](docs/phase-2-kv-cache.md).

## Balanced scoring

```python
result = model.decide_balanced(context, fields)
print(result["cause"].agreement_fraction)
print(result["cause"].order_stable)
```

Averages class-aligned raw probabilities across all answer orders for up to four classes, or cyclic rotations for larger sets. Costs additional forwards and exposes all variant probabilities and probability spans. `probability_kind` is `order_averaged_raw`; no calibration is implied. Mean ties resolve to the first canonical class. Cyclic rotations are not invariant to every reordering.

On a small synthetic evaluation, field accuracy improved from 50% to 61.1%. A subsequent 16-case set scored 64.6%. Unanimous variants sometimes agreed on wrong answers. `order_stable` is diagnostic, not an authorization to act. [Reliability experiments](reports/RELIABILITY.md).

For controlled audits, `decide(..., option_orders={"data_issue": (1, 0)})` changes presentation while preserving canonical class order. [Dataset format and audit CLI](docs/quality-evaluation.md).

## Compile repeated fields

```python
training_examples = [
    {"id": "train-1", "context": "A store upload failed.", "value": True},
    {"id": "train-2", "context": "All records reconcile exactly.", "value": False},
    # Supply many representative independently labelled examples in real use.
]
head = model.compile(
    field="data_issue",
    specification=fields["data_issue"],
    training_examples=training_examples,
    layer=-1,
    components=32,
)
head.save("artifacts/data-issue-v1")  # Refuses to overwrite an existing directory.
result = model.decide(context, fields)
assert result["data_issue"].strategy == "compiled"
assert result["cause"].strategy == "dynamic"
```

The frozen backbone produces mean-pooled hidden features. PCA/whitening and an L2-regularized linear softmax head are fitted only on the supplied training examples. Every class must appear. No backbone parameters are trained. Repeated compiled fields at the same layer share one context feature forward; layers count from embedding output (0), with `-1` selecting the final hidden state. All unmasked prompt tokens participate in mean pooling.

Heads are stored as safetensors plus JSON. `model.load_head(directory)` checks model/revision/tokenizer-template/dtype identity, and `decide` checks field question/value semantics. Anonymous in-memory models are bound to their instance. `model.remove_head(field)` restores the dynamic path. Do not mutate a loaded model's weights or tokenizer in place while using its heads.

Compiled results have `candidate_mass=None`, empty `token_ids`, classifier logits and `strategy="compiled"`. They are not LM candidate-token scores. Balanced permutations apply only to dynamic fields; attempting them on compiled names fails explicitly.

Use separate training, development, calibration and evaluation data. The runtime cannot determine whether your supplied labels leaked from an evaluation set. Save/load preserves training IDs for provenance. No automatic continual learning or layer selection is implemented. [Compiled GPU demo](reports/compiled-demo.json).

## Calibration

`system_one.calibration` provides multiclass Brier (sum over classes, range 0–2), top-label equal-bin ECE, reliability data, NLL and bounded scalar temperature fitting. `system_one.calibrate` enforces disjoint case IDs between supplied calibration and evaluation files.

```sh
OMP_NUM_THREADS=4 python -m system_one.calibrate \
  --calibration calibration.json --evaluation evaluation.json
```

Rows contain `id`, `field`, canonical `values`, `logits`, and integer `label`. Apply the fitted mapping using `model.decide(..., temperatures=report["temperatures"])`. `probability_kind` becomes `temperature_scaled`, which does not certify empirical calibration. Scaling cannot fix wrong class rankings. Fit a separate artifact when the model, head, prompt, class mapping, dtype or execution settings change. Class-specific/isotonic methods and managed calibration registries remain future work.

## Explicit fallback / escalation

```python
from system_one import EscalationPolicy, decide_with_fallback

# larger_model is a separately configured DecisionModel/provider adapter.
def fallback(context, requested_fields):
    return larger_model.decide(context, requested_fields)

routed = decide_with_fallback(
    model, context, fields,
    policy=EscalationPolicy(min_probability=0.9),  # Illustrative, not validated.
    fallback=fallback,
)
print(routed.sources)  # Per-field local/fallback provenance
```

Only flagged fields go to the explicit callback. Optional candidate-mass thresholds and balanced-order disagreement also trigger fallback. The replacement must provide exactly the requested fields with valid typed `Decision` distributions. Errors propagate; the library does not silently treat a failed fallback as success. Initial results and reasons are retained. Fallback outputs are returned once, without recursive retries.

No network provider, API key handling, external messages or automatic business actions are built in. A callback can use another local model or an application-controlled remote provider. Thresholds require representative evaluation; no universal safe threshold is claimed. [Escalation details](docs/runtime-buildout.md).

## Reproduce experiments

```sh
python -m examples.qc
python -m system_one.benchmark --revision 12a3808a956f869c767195e9266b59c4d21d92e2
python -m system_one.cache_benchmark
python -m system_one.quality_audit
OMP_NUM_THREADS=4 python -m scripts.demo_compiled_heads
```

These commands download/load weights as needed and overwrite their named report files; the compiled demo creates a new timestamped ignored artifact directory. Tests use fake models and tiny randomly initialized Transformers, with no checkpoint download. GPU experiments are separate from CPU CI.

Historical reports record the actual tested settings and limitations. They are not production benchmarks or direct comparisons with Jev. [Project direction](docs/direction.md) distinguishes implemented work from remaining research. The user-authored [related-work document](Related%20Work%20and%20Updated%20Project%20Direction.md) contains research leads whose external claims have not all been independently verified.
