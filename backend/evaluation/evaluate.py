"""
Evaluation metrics for retrieval quality and board quality benchmarking.

Section 1: Standard IR metrics (Recall@K, Precision@K, AP, nDCG, MRR)
Section 2: Routine-aware board metrics (coverage, conflict rate, tier consistency)
"""

import math
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)


# ── Section 1: Standard IR Metrics ────────────────────────────────

def recall_at_k(predicted: Sequence, truth: Sequence, k: int = 10) -> float:
    """Fraction of relevant items found in the top-k predictions."""
    return len(set(predicted[:k]) & set(truth)) / max(1, len(truth))


def precision_at_k(predicted: Sequence, truth: Sequence, k: int = 10) -> float:
    """Fraction of top-k predictions that are relevant."""
    return len(set(predicted[:k]) & set(truth)) / k


def average_precision(predicted: Sequence, truth: Sequence) -> float:
    """Average precision across all relevant positions."""
    truth_set = set(truth)
    hits, total = 0, 0.0
    for i, item in enumerate(predicted):
        if item in truth_set:
            hits += 1
            total += hits / (i + 1)
    return total / max(1, len(truth))


def ndcg_at_k(predicted: Sequence, truth: Sequence, k: int = 10) -> float:
    """Normalized discounted cumulative gain at k."""
    truth_set = set(truth)
    dcg = sum(
        1.0 / math.log2(i + 2) for i, item in enumerate(predicted[:k])
        if item in truth_set
    )
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(truth))))
    return dcg / ideal if ideal > 0 else 0.0


def mrr(predicted: Sequence, truth: Sequence) -> float:
    """Mean reciprocal rank — 1/rank of the first relevant result."""
    truth_set = set(truth)
    for i, item in enumerate(predicted):
        if item in truth_set:
            return 1.0 / (i + 1)
    return 0.0


# ── Section 2: Routine-Aware Board Metrics ────────────────────────

ROUTINE_STEPS = {"cleanser", "toner", "serum", "treatment", "moisturizer", "spf"}


def routine_coverage(board: dict) -> dict:
    """Compute routine step coverage from a board_to_dict output."""
    metrics = board.get("metrics", {})
    return {
        "covered": metrics.get("routine_coverage", 0),
        "max": metrics.get("routine_coverage_max", 5),
        "pct": metrics.get("routine_coverage_pct", 0.0),
    }


def conflict_rate(boards: list[dict]) -> float:
    """Fraction of boards that contain at least one ingredient conflict."""
    if not boards:
        return 0.0
    conflicted = sum(
        1 for b in boards if not b.get("metrics", {}).get("conflict_free", True)
    )
    return conflicted / len(boards)


def tier_consistency(board: dict) -> float:
    """Price tier standard deviation across board items (lower = more cohesive)."""
    prices = [item["price"] for item in board.get("board", [])]
    if len(prices) < 2:
        return 0.0
    mean = sum(prices) / len(prices)
    variance = sum((p - mean) ** 2 for p in prices) / len(prices)
    return math.sqrt(variance)


def avg_similarity(board: dict) -> float:
    """Mean CLIP similarity score across board items."""
    items = board.get("board", [])
    if not items:
        return 0.0
    scores = [item.get("complementarity_score", 0) for item in items]
    return sum(scores) / len(scores)


# ── Benchmark runner ──────────────────────────────────────────────

def run_benchmark(boards: list[dict]) -> dict:
    """Compute aggregate metrics across multiple generated boards."""
    if not boards:
        return {}

    coverages = [routine_coverage(b) for b in boards]
    return {
        "n_boards": len(boards),
        "avg_routine_coverage_pct": round(
            sum(c["pct"] for c in coverages) / len(coverages), 3
        ),
        "conflict_rate": round(conflict_rate(boards), 3),
        "avg_tier_consistency": round(
            sum(tier_consistency(b) for b in boards) / len(boards), 2
        ),
        "avg_complementarity_score": round(
            sum(avg_similarity(b) for b in boards) / len(boards), 4
        ),
        "perfect_coverage_rate": round(
            sum(1 for c in coverages if c["pct"] == 1.0) / len(coverages), 3
        ),
    }
