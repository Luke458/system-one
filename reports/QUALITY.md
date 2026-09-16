# Quality audit: option-order sensitivity

Ran the pinned MiniCPM5-2B checkpoint on the RX 9070 XT in bfloat16 with the batched execution strategy, one field at a time. Every two-class and four-class permutation was evaluated: 112 field evaluations over four unique synthetic QC cases. No model training, temperature fitting or prompt selection occurred.

## Findings

| Field | Cases changing answer across orders | Original-order accuracy | Accuracy across all orders (descriptive) | Largest class-probability swing |
|---|---:|---:|---:|---:|
| data_issue | 4/4 | 50% | 50.0% | 92.9 percentage points |
| cause | 3/4 | 50% | 52.1% | 99.6 percentage points |
| escalate | 3/4 | 75% | 62.5% | 96.3 percentage points |

Reversing the Boolean option presentation flipped `data_issue` on **all four cases**. The returned class order was held fixed and remapping is covered by unit tests, so this is sensitivity to the presented answer mapping, not a changed result-array interpretation.

Cause predictions changed on three of four cases. For the missing-stores case, 8 of 24 orders selected `missing_stores` and 16 selected `insufficient_evidence`. In the market case, only 4 of 24 orders selected `real_market_movement`; the other 20 selected `insufficient_evidence`.

At an original-order confidence threshold of 0.95, all three fields still included errors among accepted predictions (one of three for each Boolean, one of two for cause). These tiny counts are diagnostic only. Apparently strong conditional softmax scores are not a validated basis for automated actions or escalation thresholds.

## Interpretation

The engine produces typed distributions correctly, but this checkpoint/prompt setup is fragile under a semantically equivalent reordering of answers. Temperature scaling preserves class ranking and cannot repair this behavior. Selecting whichever option order performed best on these four examples would be tuning on the smoke evaluation, so no such change was made. No order ensemble was silently substituted for the existing scoring method.

The single-field identity scores can differ from the earlier multi-field benchmark because batch shape and bfloat16 arithmetic differ, especially around ties. They are not evidence of a model improvement. Permutations repeat the same examples; 112 measurements do not constitute 112 independently labelled cases.

## Implemented

- `option_orders` on `decide`, with strict permutation validation and canonical class-aligned outputs for Boolean/Choice.
- Audit CLI with built-in smoke data or external typed labelled datasets.
- Accuracy, Brier, ECE, reliability bins, descriptive threshold risk/coverage, answer instability and probability-span reports.
- Full input/label/order/token-ID records and model/runtime metadata for reproducibility.
- 43 passing tests, including remapping and permuted cached-versus-full parity.

Next quality work should compare Base/SFT/final checkpoints and independently specified prompt variants on representative labelled data, with development and held-out evaluation separated before choosing a configuration. Compiled heads remain a separate future experiment requiring labels. The new user direction document remains unchanged.

Raw data: `quality-audit.json`. Usage and dataset schema: `../docs/quality-evaluation.md`. Test results: `quality-tests.txt`.
