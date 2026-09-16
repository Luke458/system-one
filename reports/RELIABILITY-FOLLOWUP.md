# Further reliability investigation

The best tested configuration remains the final MiniCPM5-2B checkpoint with balanced letter scoring. No default model or prompt was changed after these experiments.

## Development comparisons

All configurations used the same 12 synthetic development cases, with 36 field labels. Final-checkpoint development measurements were reused from the pinned reliability experiment; the packaged balanced API had already reproduced its saved evaluation outputs.

| Checkpoint / scoring | Development field accuracy |
|---|---:|
| final/single | 44.4% |
| final/balanced | 63.9% |
| sft/single | 44.4% |
| sft/balanced | 61.1% |
| final / meaningful tokens / single | 44.4% |
| final / meaningful tokens / balanced | 44.4% |

SFT did not beat the incumbent. Meaningful answer tokens (`no`, `yes`, `store`, `product`, `market`, `unknown`) also did not help. These were valid single-token candidates at the actual prompt boundary, but their semantics did not overcome the classification weakness.

## Fresh synthetic evaluation

Sixteen new examples were written and saved before comparison inference, covering missing-store feeds, missing-product records, legitimate closures/assortment changes, repaired historical incidents, distractors, and insufficient current evidence.

The development-selected incumbent (`final/balanced`) scored **31/48 fields correct (64.6%)**, with **3/16 fully correct cases**. The selected meaningful-token variant scored **23/48 fields correct (47.9%)**, with **1/16 fully correct cases**. The evaluation set was reused for the latter exploratory comparison, so this is not an independent confirmatory trial. No SFT evaluation was run after SFT lost on development data.

These are hand-authored synthetic examples sharing the same task definitions and author, not production validation. Comparing a few alternatives on tiny data does not establish statistical superiority across tasks.

## Numerical check

The original single-order development run was repeated using **float32 with eager attention**, instead of bfloat16 with SDPA. It produced **exactly the same 36 selected values** and the same **44.4% accuracy**. Thus these observed classification failures are not explained by reduced-precision arithmetic or the SDPA path. Probability differences still exist and calibration must remain execution-specific.

## Current state

- Optional `decide_balanced` is implemented and full-GPU verified against saved outputs.
- 46 automated tests pass.
- No temperatures, confidence thresholds, automatic actions, or escalation policies were fitted/enabled.
- SFT weights are cached locally for further controlled experiments; the original checkpoint remains the default.
- The user's project-direction document is unchanged.

The next substantive step needs a concrete task and representative labels: compare a stronger model baseline, then train/evaluate a lightweight task-specific head if low latency remains essential. Arbitrary runtime questions can keep the dynamic path, but a learned head would have fixed trained semantics. Existing four-case smoke tests and these synthetic comparisons are not enough to choose deployment thresholds.

## Reproduction

```sh
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 .venv/bin/python -m scripts.compare_checkpoints
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 .venv/bin/python -m scripts.compare_verbalizers
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 .venv/bin/python -m scripts.check_precision
```

Pinned checkpoints:

- `openbmb/MiniCPM5-2B`: `12a3808a956f869c767195e9266b59c4d21d92e2`
- `openbmb/MiniCPM5-2B-SFT`: `3e0690c57f6ac772180498467c82e56170cc00ac`

The scripts overwrite their named result files. Raw artifacts: `checkpoint-comparison.json`, `checkpoint-fresh-evaluation.json`, `verbalizer-comparison.json`, `precision-check.json`. The original selection experiment is in `reliability-experiment.json`.
