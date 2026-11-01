"""Provider contract tests; no model server or network is required."""

import json

import httpx
import pytest

from localrag.config import Config
from localrag.providers import ModelClient


@pytest.mark.asyncio
async def test_openai_embeddings_and_streaming(monkeypatch):
    seen = []

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        seen.append((request.url.path, body, request.headers.get("authorization")))
        if request.url.path == "/v1/embeddings":
            return httpx.Response(200, json={"data": [
                {"index": 1, "embedding": [0.0, 1.0]},
                {"index": 0, "embedding": [1.0, 0.0]},
            ]})
        return httpx.Response(200, text=(
            'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
            'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
            'data: [DONE]\n\n'
        ), headers={"content-type": "text/event-stream"})

    transport = httpx.MockTransport(handler)
    original_client = httpx.AsyncClient
    monkeypatch.setattr("localrag.providers.httpx.AsyncClient",
                        lambda **kwargs: original_client(transport=transport, **kwargs))
    monkeypatch.setenv("LOCALRAG_TEST_API_KEY", "test-token")
    cfg = Config.model_validate({
        "llm": {"default": "vllm", "providers": [{
            "id": "vllm", "provider": "openai", "base_url": "http://fake/v1", "model": "example",
            "api_key_env": "LOCALRAG_TEST_API_KEY"}]},
        "embeddings": {"provider": "openai", "base_url": "http://fake/v1", "model": "vectors",
                       "api_key_env": "LOCALRAG_TEST_API_KEY"},
    })
    model = ModelClient(cfg)
    assert await model.embed(["one", "two"]) == [[1.0, 0.0], [0.0, 1.0]]
    assert "".join([piece async for piece in model.chat_stream([{"role": "user", "content": "Hi"}])]) == "Hello world"
    assert [call[0] for call in seen] == ["/v1/embeddings", "/v1/chat/completions"]
    assert all(call[2] == "Bearer test-token" for call in seen)


@pytest.mark.asyncio
async def test_native_ollama_kept_as_default(monkeypatch):
    paths = []

    def handler(request: httpx.Request):
        paths.append(request.url.path)
        if request.url.path == "/api/embed":
            return httpx.Response(200, json={"embeddings": [[0.5, 0.5]]})
        return httpx.Response(200, text=(
            '{"message":{"content":"OK"},"done":false}\n'
            '{"done":true}\n'
        ))

    transport = httpx.MockTransport(handler)
    original_client = httpx.AsyncClient
    monkeypatch.setattr("localrag.providers.httpx.AsyncClient",
                        lambda **kwargs: original_client(transport=transport, **kwargs))
    model = ModelClient(Config())
    assert await model.embed(["test"]) == [[0.5, 0.5]]
    assert "".join([part async for part in model.chat_stream([{"role": "user", "content": "Hi"}])]) == "OK"
    assert paths == ["/api/embed", "/api/chat"]
