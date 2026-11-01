"""Pure metric tests run without model downloads or network access."""

import pytest

from localrag.metrics import ndcg_at_k, recall_at_k, reciprocal_rank


def test_perfect_ranking():
    found = ["a", "b", "c"]
    relevant = {"a", "b"}
    assert recall_at_k(found, relevant, 2) == 1.0
    assert reciprocal_rank(found, relevant, 2) == 1.0
    assert ndcg_at_k(found, relevant, 2) == pytest.approx(1.0)


def test_missing_relevance_and_empty_truth():
    assert recall_at_k(["x"], {"y"}, 1) == 0.0
    assert reciprocal_rank(["x"], {"y"}, 1) == 0.0
    assert ndcg_at_k(["x"], set(), 2) == 0.0


def test_ranking_quality():
    assert ndcg_at_k(["x", "a"], {"a"}, 2) < ndcg_at_k(["a", "x"], {"a"}, 2)
