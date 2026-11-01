"""Standard retrieval metrics for labeled query/document relevance pairs."""

from __future__ import annotations

import math


def recall_at_k(found: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant documents retrieved within the first k results."""
    return len(set(found[:k]) & relevant) / len(relevant) if relevant else 0.0


def reciprocal_rank(found: list[str], relevant: set[str], k: int) -> float:
    for position, doc_id in enumerate(found[:k], start=1):
        if doc_id in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(found: list[str], relevant: set[str], k: int) -> float:
    """Binary normalized DCG (higher rank carries greater value)."""
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, doc in enumerate(found[:k]) if doc in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / ideal if ideal else 0.0
