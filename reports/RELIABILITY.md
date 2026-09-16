# Reliability improvement experiment

Added optional `model.decide_balanced(context, fields)` to average canonical class probabilities across answer presentations. It exposes disagreement and per-class probability spans, leaves the ordinary `decide` default unchanged, and makes no empirical calibration claim.

## Method

Before inference, wrote 24 new hand-authored synthetic cases: 12 development and 12 evaluation, each split containing three examples of each QC cause. These differ from the four earlier smoke cases. Both splits share the same author, task definitions and similar evidence patterns, so this is not an independent real-world benchmark.

Three variants were fixed before examining their outputs:

1. Original prompt, original class order.
2. Original prompt, mean probabilities over every permutation (2 per Boolean, 24 for cause).
3. More explicit evidence-oriented prompt and answer definitions, with the same averaging.

Selection used development field accuracy, then exact-case accuracy; the baseline wins exact ties. Only the baseline and development winner were then run on the evaluation split. No temperatures were fitted and no order was chosen after looking at its score.

## Development selection

| Variant | Field accuracy | Fully correct cases |
|---|---:|---:|
| baseline | 44.4% | 0.0% |
| balanced | 63.9% | 16.7% |
| defined_balanced | 50.0% | 25.0% |

The original-prompt balanced variant won. The new explicit-definition prompt was not adopted.

## Evaluation

| Measure | Original single order | Balanced orders |
|---|---:|---:|
| Field accuracy (36 labels) | 50.0% | 61.1% |
| Fully correct cases (12) | 8.3% | 25.0% |
| data_issue Brier (lower is better) | 0.654 | 0.407 |
| cause Brier (lower is better) | 1.052 | 0.933 |
| escalate Brier (lower is better) | 0.912 | 0.417 |

This is four additional correct field predictions (22/36 versus 18/36), and three fully correct cases versus one. The sample is too small and synthetic to establish a dependable operating accuracy.

Measured per-case median elapsed time in this experiment was roughly 223 ms for balanced scoring versus 36 ms for the baseline. These timings include the complete calls but were not a dedicated warmed latency benchmark. Balanced scoring trades additional computation for modest quality improvement. Shared-prefix mode is supported by the API but was not used in this evaluation.

## Agreement does not establish correctness

Only 7 of 36 balanced evaluation predictions had unanimous argmax across orders. Of those seven, four were correct and **three were wrong**. Therefore, simply trusting unanimous predictions would still be unreliable. No automatic action or validated abstention threshold is provided.

`BalancedDecision` returns a proposed typed value, mean raw probability distribution, agreement fraction, stability flag, probability spans, all variant distributions and candidate-mass minimum. Probabilities are explicitly marked `order_averaged_raw`. No fictional averaged logits or shared token IDs are returned: token-to-class mappings differ between variants.

For <=4 classes, exhaustive averaging removes pure fixed-letter bias in a controlled test and is class-reordering invariant up to numerical tolerance. Larger choices use cyclic rotations to bound cost and are not invariant to arbitrary order changes. Mean ties resolve to the first canonical class. Cache mode, batch shape and bfloat16 execution can still introduce score differences.

## Validation and next work

46 tests pass, including remapping, an artificial fixed-letter-bias model, float32 class-reordering invariance and cached/full balanced parity. A separate full-GPU check compares the packaged API against all 36 saved evaluation field outputs.

For more substantial gains, the next experiment should compare checkpoint variants and task-specific learning with representative labelled records and independent calibration/evaluation splits. Do not treat post-hoc temperature scaling as a fix for incorrect class rankings. Keep the current implementation experimental and retain human review for decisions that matter.

Files: `reliability-dataset.json`, `reliability-experiment.json`, `balanced-api-check.json`, `balanced-tests.txt`. Reproduction: `python -m scripts.reliability_experiment`. GPU API check: `python -m scripts.check_balanced`.
