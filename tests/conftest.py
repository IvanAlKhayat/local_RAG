"""Unit-test vector-store substitute.

These are API/unit tests, not Chroma integration benchmarks. No native Rust
extension, vector downloads, network or live model server is required.
"""

import chromadb
import pytest


class MemoryCollection:
    def __init__(self):
        self.rows = {}

    def upsert(self, ids, documents, embeddings, metadatas):
        for key, content, vector, meta in zip(ids, documents, embeddings, metadatas):
            self.rows[key] = (content, vector, meta)

    def delete(self, ids):
        for key in ids:
            self.rows.pop(key, None)

    def count(self):
        return len(self.rows)

    def query(self, query_embeddings, n_results, where):
        selected = [(key, row) for key, row in self.rows.items()
                    if row[2]["source_id"] in where["source_id"]["$in"]]
        vector = query_embeddings[0]
        selected.sort(key=lambda pair: sum(a * b for a, b in zip(vector, pair[1][1])), reverse=True)
        return {"ids": [[key for key, _ in selected[:n_results]]]}


class MemoryClient:
    def __init__(self):
        self.collections = {}

    def get_or_create_collection(self, name):
        return self.collections.setdefault(name, MemoryCollection())

    def delete_collection(self, name):
        self.collections.pop(name, None)


@pytest.fixture(autouse=True)
def stub_native_vector_store(monkeypatch):
    monkeypatch.setattr(chromadb, "PersistentClient", lambda **kwargs: MemoryClient())
