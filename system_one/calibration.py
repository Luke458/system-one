"""Multiclass Brier (sum over classes), top-label ECE, held-out NLL scaling."""
import torch


def _validate(scores, labels, probabilities=False):
    x = torch.as_tensor(scores, dtype=torch.float64, device="cpu")
    y0 = torch.as_tensor(labels, device="cpu")
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] < 2 or not torch.isfinite(x).all():
        raise ValueError("Expected finite nonempty [N,C] scores, C >= 2")
    if y0.ndim != 1 or len(y0) != len(x) or y0.dtype == torch.bool or y0.is_floating_point():
        raise ValueError("Expected one integer class index per row")
    y = y0.long()
    if (y < 0).any() or (y >= x.shape[1]).any():
        raise ValueError("Class index out of range")
    if probabilities and ((x < 0).any() or (x > 1).any() or
                          not torch.allclose(x.sum(1), torch.ones(len(x), dtype=x.dtype), atol=1e-6, rtol=0)):
        raise ValueError("Invalid probability distributions")
    return x, y


def brier_score(probabilities, labels):
    p, y = _validate(probabilities, labels, True)
    target = torch.nn.functional.one_hot(y, p.shape[1])
    return float(((p - target) ** 2).sum(1).mean())


def reliability_data(probabilities, labels, bins=10):
    p, y = _validate(probabilities, labels, True)
    if type(bins) is not int or bins < 1:
        raise ValueError("bins must be a positive integer")
    confidence, prediction = p.max(1)
    correct = prediction.eq(y).double()
    assignment = (confidence * bins).long().clamp(max=bins - 1)
    result = []
    for b in range(bins):
        mask = assignment.eq(b)
        count = int(mask.sum())
        result.append({"lower": b / bins, "upper": (b + 1) / bins, "count": count,
                       "confidence": float(confidence[mask].mean()) if count else None,
                       "accuracy": float(correct[mask].mean()) if count else None})
    return result


def expected_calibration_error(probabilities, labels, bins=10):
    data = reliability_data(probabilities, labels, bins)
    n = sum(row["count"] for row in data)
    return sum(row["count"] / n * abs(row["accuracy"] - row["confidence"])
               for row in data if row["count"])


def fit_temperature(logits, labels):
    """Fit bounded scalar T on calibration-only data, never the evaluation split.

    Deterministic log-space grid includes T=1, so calibration NLL cannot worsen.
    Bounds [0.05,20] regularize degenerate/perfectly separable small datasets.
    """
    x, y = _validate(logits, labels)
    grid = torch.cat([torch.logspace(-1.30103, 1.30103, 401, dtype=x.dtype).clamp(0.05, 20.0), torch.ones(1)])
    losses = torch.stack([torch.nn.functional.cross_entropy(x / t, y) for t in grid])
    return float(grid[losses.argmin()])


def calibration_report(logits, labels, temperature=1.0, bins=10):
    import math
    x, y = _validate(logits, labels)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Invalid temperature")
    p = (x / temperature).softmax(1)
    return {"temperature": temperature, "n": len(y), "brier": brier_score(p, y),
            "ece": expected_calibration_error(p, y, bins),
            "nll": float(torch.nn.functional.cross_entropy(x / temperature, y)),
            "reliability": reliability_data(p, y, bins)}
