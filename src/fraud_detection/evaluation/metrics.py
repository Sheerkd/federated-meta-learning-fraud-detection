"""Consistent validation thresholds and rare-event classification metrics."""

import numpy as np
from sklearn.metrics import (
    average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score,
)


def choose_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, dict]:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    _validate(labels, scores)
    if len(np.unique(labels)) < 2:
        raise ValueError("Validation threshold needs both classes")
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = np.divide(2 * precision[:-1] * recall[:-1], precision[:-1] + recall[:-1],
                   out=np.zeros_like(thresholds), where=(precision[:-1] + recall[:-1]) > 0)
    best = np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))
    best_recall = recall[best].max()
    tied = best[np.isclose(recall[best], best_recall, rtol=0, atol=1e-12)]
    # Thresholds are ascending. Lowest threshold is the final deterministic tie rule.
    index = int(tied[0])
    return float(thresholds[index]), {"rule": "maximize validation F1, then recall, then lower threshold",
                                      "validation_f1": float(f1[index]),
                                      "validation_recall": float(recall[index])}


def classification_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    _validate(labels, scores)
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    prediction = (scores >= threshold).astype(np.int8)
    tn, fp, fn, tp = (int(n) for n in confusion_matrix(labels, prediction, labels=[0, 1]).ravel())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else None
    both_classes = len(np.unique(labels)) == 2
    return {
        "rows": len(labels), "fraud": int(labels.sum()),
        "fraud_prevalence": float(labels.mean()),
        "threshold": float(threshold),
        "precision": precision, "recall": recall, "f1": f1,
        "false_positive_rate": fpr,
        "roc_auc": float(roc_auc_score(labels, scores)) if both_classes else None,
        "average_precision": float(average_precision_score(labels, scores)) if both_classes else None,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "undefined_reason": None if both_classes else "Only one class in this evaluation set",
    }


def macro_average(by_client: dict[str, dict]) -> dict:
    fields = ("precision", "recall", "f1", "false_positive_rate", "roc_auc", "average_precision")
    return {field: float(np.mean([row[field] for row in by_client.values() if row[field] is not None]))
            if any(row[field] is not None for row in by_client.values()) else None
            for field in fields}


def _validate(labels: np.ndarray, scores: np.ndarray) -> None:
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores) or len(labels) == 0:
        raise ValueError("Labels and scores must be nonempty 1D arrays of equal length")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Labels or scores are invalid")
    if not ((scores >= 0) & (scores <= 1)).all():
        raise ValueError("Scores must be in [0, 1]")
