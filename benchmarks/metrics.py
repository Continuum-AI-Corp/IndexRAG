"""
Evaluation metrics for IndexRAG benchmarks.

Supports: Exact Match (EM), F1, Substring Match.
"""

import re
from typing import List


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = " ".join(s.split())
    return s


def exact_match(predicted: str, ground_truth: str) -> bool:
    """Check if normalized prediction matches ground truth."""
    return normalize_answer(predicted) == normalize_answer(ground_truth)


def substring_match(predicted: str, ground_truth: str) -> bool:
    """Check if ground truth is a substring of prediction."""
    return normalize_answer(ground_truth) in normalize_answer(predicted)


def f1_score(predicted: str, ground_truth: str) -> float:
    """Compute token-level F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer(predicted).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)

    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_metrics(
    predictions: List[str],
    ground_truths: List[str],
) -> dict:
    """
    Compute aggregate metrics over a list of predictions.

    Returns:
        Dict with 'em', 'f1', 'substring_match' averages.
    """
    n = len(predictions)
    if n == 0:
        return {"em": 0.0, "f1": 0.0, "substring_match": 0.0, "n": 0}

    em_total = sum(exact_match(p, g) for p, g in zip(predictions, ground_truths))
    f1_total = sum(f1_score(p, g) for p, g in zip(predictions, ground_truths))
    sub_total = sum(substring_match(p, g) for p, g in zip(predictions, ground_truths))

    return {
        "em": em_total / n,
        "f1": f1_total / n,
        "substring_match": sub_total / n,
        "n": n,
    }
