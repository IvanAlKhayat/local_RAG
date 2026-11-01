"""Verify the committed synthetic benchmark can run without model downloads."""

import json

import pytest

from evaluation.benchmark_synthetic import ROOT, load_cases, run_benchmark


def test_synthetic_gold_labels():
    cases = load_cases()
    assert len(cases) == 70
    assert any(case["category"] == "hard_paraphrase" for case in cases)
    assert all(case["answer_contains"] for case in cases)


@pytest.mark.asyncio
async def test_sparse_benchmark_is_offline_and_measured():
    report = await run_benchmark("sparse", "http://127.0.0.1:1", "not-installed")
    assert report["docs"] == 10
    assert report["questions"] == 70
    assert report["mode"] == "sparse"
    assert report["embedding_model"] == "not used (sparse-only)"
    assert report["metrics"]["recall_at_5"] >= 0.8
    assert report["query_latency_ms"]["p95"] >= 0
    committed = json.loads((ROOT / "results" / "synthetic_sparse.json").read_text())
    assert report["metrics"] == committed["metrics"]
