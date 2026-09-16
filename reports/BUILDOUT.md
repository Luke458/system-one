# Runtime build-out verification

The initial architecture now includes compiled frozen-LM heads and an explicit fallback callback, alongside dynamic scoring, shared-prefix inference and calibration utilities.

## GPU integration

Three heads were fitted from 12 synthetic development contexts, with fixed final-layer mean pooling, 8 PCA components and L2-regularized softmax classifiers. All three heads were saved as safetensors/JSON, removed from the model registry and reloaded. They served automatically through `model.decide`.

On 16 previously used synthetic evaluation cases, compiled-head field accuracy was **77.1%** (37/48 labels). The prior balanced zero-shot configuration scored 31/48 (64.6%) on that set. This dataset was already used in earlier experiments; the result is integration evidence and an exploratory comparison, not a fresh generalization claim.

Median recorded end-to-end compiled call: **25.3 ms** across these 16 short contexts, for all three fields sharing one feature forward. This is not a controlled performance study.

## Implementation boundaries

PCA and classifier fitting use only the supplied training examples; the backbone is frozen. Heads validate field semantics and model identity, and use non-pickle persistence. Calibration remains an explicit separate workflow. Policy-driven escalation invokes only an application-supplied callback, validates its outputs, and preserves provenance; no external provider or business action is built in.

Raw training/evaluation provenance and per-field probabilities: `compiled-demo.json`. Fresh model-instance reload verification: `head-reload-check.json`. CPU test record: `buildout-tests.txt`. Source and wheel builds both succeed.
