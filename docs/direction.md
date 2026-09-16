# Semantic decision runtime: implementation status

The user-authored `Related Work and Updated Project Direction.md` is the project direction. Its related-work claims are research leads, not independently verified findings of this repository. This project makes no novelty or Jev/RLCD reproduction claim.

The stable abstraction is `context + typed questions → distributions`. MiniCPM5-2B is the first tested full checkpoint, not a mandatory architecture. Preserve explicit question strings and class ordering while the API matures; the shorthand constructors and attribute-access examples in the direction document are proposals rather than implemented compatibility contracts.

| Level | Implemented now | Next evidence needed |
|---|---|---|
| Dynamic decisions | Boolean/Choice candidate scoring, exact token validation, batched execution, opt-in shared prefix | Option-order audit implemented; next: prompt sensitivity, checkpoint comparison, representative labels |
| Calibration | Brier, ECE, reliability data, separate-split scalar temperature fitting; explicit raw/scaled result metadata | Independent task evaluation, versioned calibration artifacts tied to model/prompt/class order/execution/dtype |
| Compiled heads | Frozen hidden features, PCA whitening, linear heads, save/load and automatic field dispatch implemented | Representative labels, pooling/layer ablations, disjoint calibration and OOD evaluation |
| Escalation | Explicit policy and validated fallback callback implemented; no built-in network provider | Held-out risk/coverage/cost curves, application provider adapters and validated policies |

`Decision.probability` is the selected-class probability. `raw_probability` remains the original conditional probability. `probability_kind` says `raw` or `temperature_scaled`. Supplying a temperature, even a fitted one, is not sufficient evidence to call the result empirically calibrated. No confidence threshold automatically authorizes an action.

## Next priority after shared-cache measurements

The Phase 1 smoke benchmark found confidently wrong answers. Build a larger labelled corpus with a predefined calibration/evaluation split and audit label permutation, prompt wording and Base/SFT/final checkpoints before optimizing further. Evaluate class accuracy, Brier/ECE and risk/coverage together; do not tune on the evaluation split. Keep the four original QC examples as smoke tests.

For future learned heads, fit PCA/whitening and classifiers only on training data; choose layer/hyperparameters on validation data; reserve calibration and final test data separately. Include strong cheap baselines and feature-extraction latency in the end-to-end comparison. Do not promise that a small head eliminates the cost of running its LM backbone.

Keep production cache integration, alternate calibration families, compiled heads and escalation as separate reviewable increments. Plain PyTorch/Transformers remains the current backend. Synthetic head training has been run; no escalation service or automatic external action has been started.
