# Compiled fields and explicit escalation

The runtime now has experimental implementations of the four planned levels: dynamic candidate scoring, calibration tools, compiled hidden-state classifiers, and policy-triggered fallback callbacks. This completes the initial architecture, not production reliability or a reproduction of proprietary systems.

## Compiled head boundary

`compile(field, specification, training_examples, layer=-1, components=32, l2=.01)` fits PCA whitening and a linear softmax classifier on frozen mean-pooled backbone features. Training labels are exact typed values, with unique IDs and all classes represented. Fit uses a deterministic SVD and zero-initialized L-BFGS linear fit. The component count is bounded by sample count and feature width; eigenvalue scaling has a small numerical floor. No hyperparameter search or validation-set selection is performed.

The feature format includes a fixed classification instruction, JSON-encoded context and the checkpoint's non-thinking chat template. Mean pooling covers all unmasked tokens. The field question itself is not encoded in features: it is fixed head semantics and is validated against metadata at inference. Heads sharing a layer share feature extraction in `decide`; different layers currently require separate backbone calls. This is not an optimized early-exit encoder.

Safetensors store four numeric arrays (mean, projection, weights, bias); metadata stores semantics, model identity, training IDs and fit parameters. Loading validates tensor shapes and finiteness and binds the head to model/revision/template/dtype. Anonymous models are instance-bound; externally changing weights/tokenizer state in place is not supported. Artifacts are ignored by Git because training IDs and task-specific models may contain application-sensitive information.

Calibration remains separate: collect classifier logits on a disjoint calibration split, fit temperatures, and evaluate on another split. No representative real-world calibration has been established.

## Fallback boundary

`decide_with_fallback` executes the local path, applies an explicitly constructed policy and calls an explicitly supplied callback once with only affected fields. Reasons can include low probability, low/unavailable candidate mass, or disagreement in balanced dynamic scoring. Compiled heads have no LM candidate mass; a nonzero mass requirement therefore sends those results to fallback.

Returned replacements must exactly match requested keys, typed allowed values, canonical class order and valid normalized probabilities. Invalid replacements or callback exceptions fail the request. Result metadata preserves initial predictions, reasons, final decisions and local/fallback sources. There is no recursive retry, timeout manager, service deployment, provider integration or external action.

A confidence cutoff expresses application policy, not a correctness guarantee. The included synthetic experiments show that even unanimous option orders can be wrong. Evaluation of coverage/risk/cost and real-world fallback behavior remains application work.

## Validation

Unit tests cover frozen-feature extraction, padding parity, PCA/classifier fit, tensor persistence, mixed compiled/dynamic dispatch, changed schema/model rejection, strict typed training labels, fallback triggering/skipping, invalid provider output and provider failure propagation.

`scripts.demo_compiled_heads` runs real MiniCPM5-2B feature extraction, trains all three QC heads from 12 existing synthetic development cases, saves/reloads each, and evaluates on 16 previously used synthetic cases. This is integration evidence only, not a fresh generalization claim. Raw outputs are in `reports/compiled-demo.json`.
