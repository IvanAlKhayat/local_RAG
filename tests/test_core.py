from pathlib import Path

import pytest
import httpx

from localrag.config import Config
from localrag.engine import RAGEngine, safe_local_file
from localrag.parsing import Passage, chunk_passages, stable_id
from localrag.storage import Storage


def test_chunker_respects_page_boundaries_and_overlap():
    parts = [Passage("alpha " * 200, 4), Passage("beta " * 100, 5)]
    chunks = chunk_passages(parts, 300, 40)
    assert len(chunks) > 4
    assert all(len(chunk.text) <= 300 for chunk in chunks)
    assert chunks[0].page == 4
    assert chunks[-1].page == 5


def test_config_rejects_bad_source_ids():
    with pytest.raises(ValueError):
        Config.model_validate({"sources": [{"id": "../secrets", "type": "local", "path": "/tmp"}]})


def test_local_root_cannot_be_escaped_or_follow_symlinks(tmp_path: Path):
    root = tmp_path / "allowed"
    root.mkdir()
    inside = root / "paper.txt"
    inside.write_text("text")
    outside = tmp_path / "private.txt"
    outside.write_text("private")
    (root / "shortcut.txt").symlink_to(outside)
    assert safe_local_file(root, "paper.txt") == inside
    with pytest.raises(ValueError):
        safe_local_file(root, "../private.txt")
    with pytest.raises(ValueError):
        safe_local_file(root, "shortcut.txt")


def test_index_search_replace_and_delete(tmp_path: Path):
    db = Storage(tmp_path / "test.sqlite3")
    doc = {"id": stable_id("file"), "source_id": "notes", "relative_path": "example.md", "title": "example.md", "fingerprint": "123"}
    a = {"id": "a", "text": "semantic retrieval with document citations", "page": 3}
    db.replace_document(doc, [a])
    assert db.sparse_search("document citations", 5)[0]["id"] == "a"
    assert db.sparse_search("document citations", 5, allowed={"other"}) == []
    db.replace_document(doc, [{"id": "b", "text": "new data", "page": None}])
    assert db.sparse_search("document citations", 5) == []
    with db.connect() as con:
        assert db.delete_document(con, doc["id"]) == ["b"]
    assert db.documents() == []


@pytest.mark.asyncio
async def test_local_sync_and_retrieval(tmp_path: Path, monkeypatch):
    # Keep tiny file operations synchronous in tests for deterministic execution.
    async def inline_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr("localrag.engine.asyncio.to_thread", inline_thread)
    root = tmp_path / "papers"
    root.mkdir()
    (root / "study.md").write_text("Dense semantic retrieval locates similar passages. Hybrid search also uses BM25.")
    config = Config.model_validate({
        "app": {"data_dir": str(tmp_path / "appdata")},
        "sources": [{"id": "papers", "type": "local", "path": str(root)}],
    })
    service = RAGEngine(config)

    async def fake_embed(texts):
        return [[1.0 if "semantic" in text.lower() else 0.1, 0.3] for text in texts]

    service.models.embed = fake_embed
    first = await service.sync_source("papers")
    assert first["indexed"] == 1
    assert (await service.sync_source("papers"))["indexed"] == 0
    results = await service.retrieve("semantic retrieval", ["papers"])
    assert results and results[0]["title"] == "study.md"
    (root / "study.md").unlink()
    assert (await service.sync_source("papers"))["removed"] == 1
    assert not await service.retrieve("semantic retrieval", ["papers"])


@pytest.mark.asyncio
async def test_api_basic_endpoints(monkeypatch):
    from localrag.api import app

    async def models(_provider):
        return ["qwen3.5:9b-q4_K_M"]

    monkeypatch.setattr(app.state.engine.registry, "available", models)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/api/health")).status_code == 200
        assert (await client.get("/api/sources")).status_code == 200
        listed = (await client.get("/v1/models")).json()
        assert listed["object"] == "list"
        assert listed["data"][0]["id"] == "chat/ollama/qwen3.5:9b-q4_K_M"


def test_config_multiple_models():
    cfg = Config.model_validate({"llm": {"default": "ollama", "providers": [
        {"id": "ollama", "provider": "ollama"},
        {"id": "vllm-hpc", "provider": "openai", "enabled": True, "model": "example"},
    ]}})
    assert cfg.llm.get_provider("vllm-hpc").provider == "openai"
    with pytest.raises(ValueError):
        Config.model_validate({"llm": {"default": "missing"}})
