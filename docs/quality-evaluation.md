# Option-order and confidence audit

Run the fixed smoke audit:

```sh
OMP_NUM_THREADS=4 .venv/bin/python -m system_one.quality_audit
```

Or pass a labelled dataset with `--dataset path.json`. This audit is descriptive: it does not fit temperatures, select the best option order, tune prompts, train a head, or trigger escalation. Fields are evaluated individually with a fixed batch shape so changing unrelated field batches does not confound option-order comparisons. It is not a latency benchmark.

For two through four classes, every permutation is scored; for larger class sets the audit scores cyclic rotations, so every class occupies every position once. The report records the full order, actual candidate token IDs, class-aligned probabilities and logits, candidate mass, expected class index, dataset contents/hash, checkpoint revision, dtype, execution mode and library versions.

Results stay in the caller's canonical class order even when presentation order changes:

```python
result = model.decide(
    context, fields,
    option_orders={"data_issue": (1, 0), "cause": (3, 1, 0, 2)},
)
```

Each tuple lists the original class indices in display order. For a Boolean, `(1,0)` displays `A: true`, `B: false`, but result values/distributions remain ordered `(False, True)`. Missing fields keep their default order. Unknown field names, duplicate/out-of-range indices, and Boolean indices are rejected. Single-token validation is applied to the actual remapped candidate labels. Shared-prefix execution also supports these orders.

## Dataset format

```json
{
  "name": "labelled_qc_evaluation",
  "fields": {
    "data_issue": {
      "type": "boolean",
      "question": "Does the evidence establish a data completeness issue?"
    },
    "cause": {
      "type": "choice",
      "question": "What is the best-supported cause?",
      "values": ["missing_stores", "missing_products", "market", "unknown"]
    }
  },
  "cases": [
    {
      "id": "unique-case-id",
      "context": "The actual observed evidence goes here.",
      "expected": {"data_issue": false, "cause": "unknown"}
    }
  ]
}
```

The example is a format illustration, not a real labelled record. IDs must be unique. Expected labels must match exact Boolean/string types and choices. Do not include private context in reports intended for sharing: the report deliberately retains complete input records for reproducibility. Inputs still obey the engine's context-length limit; oversized records fail explicitly.

## Reading the report

- **Identity metrics:** one prediction per unique case with the original class order: accuracy, multiclass Brier, ECE, reliability bins and risk/coverage at fixed confidence thresholds.
- **Unstable cases:** unique cases whose selected value changes under at least one order.
- **Disagreement with identity:** fraction of non-identity permutations that disagree with the original order.
- **Probability span:** largest max-minus-min probability for any canonical class across permutations of one case.
- **All-order metrics:** descriptive averages over repeated measurements. These are not independent examples and must not be used to inflate sample-size or significance claims.
- **Risk/coverage:** error rate among predictions meeting a confidence threshold, plus fraction accepted. No accepted cases yields null risk. These curves are diagnostics, not validated operating policies.

The four original smoke examples cannot establish reliability or safe escalation thresholds. Keep them as regressions. For actual evaluation, obtain independently labelled representative records, separate train/development/calibration/evaluation splits by source or time as appropriate, and freeze prompt/order/backend settings before evaluating the held-out split. Changing the option mapping changes the experimental configuration and invalidates assumptions behind an old fitted temperature.
