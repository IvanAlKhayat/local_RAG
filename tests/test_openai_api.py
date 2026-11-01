"""Open WebUI compatibility tests with stubbed retrieval and generation."""

import pytest
import httpx


@pytest.fixture
def app(monkeypatch):
    # Import only after the test-specific in-memory Chroma substitute is ready.
    from localrag.api import app as application

    async def installed_models(provider):
        return ["qwen3.5:9b-q4_K_M"]

    monkeypatch.setattr(application.state.engine.registry, "available", installed_models)
    return application


@pytest.mark.asyncio
async def test_no_index_returns_safe_answer(app, monkeypatch):
    async def empty_retrieve(*args):
        return []

    monkeypatch.setattr(app.state.engine, "retrieve", empty_retrieve)
    async def fail_if_called(*_args):
        raise AssertionError("RAG must not hallucinate when no chunks match")

    monkeypatch.setattr(app.state.engine.models, "chat_stream", fail_if_called)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/chat/completions", json={
            "model": "rag/ollama/qwen3.5:9b-q4_K_M", "messages": [{"role": "user", "content": "What is in my files?"}]
        })
        assert response.status_code == 200
        assert "No indexed passages" in response.json()["choices"][0]["message"]["content"]


@pytest.mark.asyncio
async def test_chat_without_documents_and_streaming(app, monkeypatch):
    async def retrieval_must_not_run(*_args):
        raise AssertionError("Chat mode must skip retrieval")

    async def generate(messages, provider_id, model_name):
        assert provider_id == "ollama"
        assert model_name == "qwen3.5:9b-q4_K_M"
        assert messages == [{"role": "system", "content": "Be concise."},
                            {"role": "user", "content": "Hi!"}]
        yield "Hello!"

    monkeypatch.setattr(app.state.engine, "retrieve", retrieval_must_not_run)
    monkeypatch.setattr(app.state.engine.models, "chat_stream", generate)
    payload = {"model": "chat/ollama/qwen3.5:9b-q4_K_M", "messages": [
        {"role": "system", "content": "Be concise."}, {"role": "user", "content": "Hi!"}]}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "Hello!"
        streamed = await client.post("/v1/chat/completions", json={**payload, "stream": True})
        assert streamed.status_code == 200
        assert "Hello!" in streamed.text and "[DONE]" in streamed.text
        invalid = await client.post("/v1/chat/completions", json={**payload, "source_ids": []})
        assert invalid.status_code == 400


@pytest.mark.asyncio
async def test_chat_citations_and_sse(app, monkeypatch):
    async def retrieve(*args):
        return [{"doc_id": "d1", "title": "guide.md", "source_id": "documents", "page": 2,
                 "content": "The service runs on port 8000."}]

    async def generate(messages, provider_id, model_name):
        assert provider_id == "ollama"
        assert model_name == "qwen3.5:9b-q4_K_M"
        assert "guide.md" in messages[0]["content"]
        yield "Port 8000 [1]."

    monkeypatch.setattr(app.state.engine, "retrieve", retrieve)
    monkeypatch.setattr(app.state.engine.models, "chat_stream", generate)
    payload = {"model": "rag/ollama/qwen3.5:9b-q4_K_M", "messages": [{"role": "user", "content": "What port?"}]}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        result = await client.post("/v1/chat/completions", json=payload)
        assert result.status_code == 200
        assert "Open document" in result.json()["choices"][0]["message"]["content"]
        streamed = await client.post("/v1/chat/completions", json={**payload, "stream": True})
        assert streamed.status_code == 200
        assert "[DONE]" in streamed.text
        assert "Port 8000" in streamed.text


@pytest.mark.asyncio
async def test_invalid_model_and_multimodal_rejected(app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.post("/v1/chat/completions", json={
            "model": "localrag/nope", "messages": [{"role": "user", "content": "Hi"}]
        })).status_code == 404
        assert (await client.post("/v1/chat/completions", json={
            "model": "localrag/ollama/qwen3.5:9b-q4_K_M", "messages": [{"role": "user", "content": [{"type": "image_url"}]}]
        })).status_code == 400


@pytest.mark.asyncio
async def test_manage_page_points_to_native_chat(app, monkeypatch):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/manage")
        assert response.status_code == 200
        assert 'href="http://127.0.0.1:8080"' in response.text
        monkeypatch.setenv("LOCALRAG_WEBUI_URL", "http://127.0.0.1:3000")
        assert 'href="http://127.0.0.1:3000"' in (await client.get("/manage")).text
