"""Synthetic QC smoke fixtures, not production evaluation data."""
from .engine import Boolean, Choice

FIELDS = {
    "data_issue": Boolean("Does the evidence indicate a data completeness or pipeline issue?"),
    "cause": Choice("What is the best-supported cause?", (
        "missing_stores", "missing_products", "real_market_movement", "insufficient_evidence")),
    "escalate": Boolean("Should data engineering investigate a likely pipeline issue?"),
}
CASES = [
    {"id": "stores", "context": "Compared with last week, reported stores fell from 500 to 360. Products per store are unchanged. Sales fell 28%. There were no store closures, and ingestion logs show 140 stores failed to upload.",
     "expected": {"data_issue": True, "cause": "missing_stores", "escalate": True}},
    {"id": "products", "context": "All 500 stores uploaded. Half the product categories are absent after a failed product mapping deployment. Sales totals fell 48%. No assortment changes occurred.",
     "expected": {"data_issue": True, "cause": "missing_products", "escalate": True}},
    {"id": "market", "context": "All stores and products are present. Source receipts reconcile exactly with the warehouse. A promotion ended yesterday; transactions fell 15%, consistent with independent receipt counts. No pipeline errors occurred.",
     "expected": {"data_issue": False, "cause": "real_market_movement", "escalate": False}},
    {"id": "unknown", "context": "Sales fell 20%. No store counts, product counts, source receipts, or pipeline logs are available. There is insufficient evidence to identify the cause or establish a pipeline fault.",
     "expected": {"data_issue": False, "cause": "insufficient_evidence", "escalate": False}},
]

