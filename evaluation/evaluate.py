"""Run retrieval-only evaluation on a JSONL corpus with known relevant docs.

Each line: {"query": "...", "relevant_doc_ids": ["..."], "source_ids": [...]}
Get document IDs from GET /api/documents after indexing your files.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from localrag.config import load_config
from localrag.engine import RAGEngine
from localrag.metrics import ndcg_at_k, recall_at_k, reciprocal_rank


async def evaluate(dataset: Path) -> dict:
    engine = RAGEngine(load_config())
    cases = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if not cases:
        raise ValueError("Dataset contains no examples")
    scores = {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0}
    for case in cases:
        selected = case.get("source_ids") or [s.id for s in engine.config.sources if s.enabled]
        # Retrieval returns chunks; measure relevance at the document level.
        found = list(dict.fromkeys(row["doc_id"] for row in await engine.retrieve(case["query"], selected)))
        relevant = set(case["relevant_doc_ids"])
        k = engine.config.retrieval.final_top_k
        scores["recall"] += recall_at_k(found, relevant, k)
        scores["mrr"] += reciprocal_rank(found, relevant, k)
        scores["ndcg"] += ndcg_at_k(found, relevant, k)
    return {"cases": len(cases), "k": engine.config.retrieval.final_top_k,
            **{name: round(value / len(cases), 4) for name, value in scores.items()}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate document retrieval quality")
    parser.add_argument("dataset", type=Path, help="JSONL path with labeled examples")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(evaluate(args.dataset)), indent=2))


if __name__ == "__main__":
    main()
