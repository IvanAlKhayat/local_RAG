"""Model discovery contract tests; no external model server is needed."""

import json

import httpx
import pytest

from localrag.config import Config
from localrag.discovery import ModelRegistry


@pytest.mark.asyncio
async def test_ollama_discovery_filters_embeddings_and_caches(monkeypatch):
    requests = []

    def handler(request: httpx.Request):
        requests.append(request.url.path)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [
                {"name": "qwen3.5:9b-q4_K_M"},
                {"name": "huihui_ai/qwen3.5-abliterated:9b"},
                {"name": "embeddinggemma:latest"},
            ]})
        model = json.loads(request.content)["model"]
        capability = "embedding" if model.startswith("embedding") else "completion"
        return httpx.Response(200, json={"capabilities": [capability]})

    transport = httpx.MockTransport(handler)
    original = httpx.AsyncClient
    monkeypatch.setattr("localrag.discovery.httpx.AsyncClient",
                        lambda **kwargs: original(transport=transport, **kwargs))
    registry = ModelRegistry(Config())
    listed = await registry.list_models()
    ids = {item["id"] for item in listed}
    assert ids == {f"{mode}/ollama/{name}" for mode in ("chat", "rag")
                   for name in ("qwen3.5:9b-q4_K_M", "huihui_ai/qwen3.5-abliterated:9b")}
    assert await registry.resolve("chat/ollama/qwen3.5:9b-q4_K_M") == (
        "chat", registry.config.llm.get_provider("ollama"), "qwen3.5:9b-q4_K_M")
    assert (await registry.resolve("rag/ollama/qwen3.5:9b-q4_K_M"))[0] == "rag"
    assert (await registry.resolve("localrag/ollama/qwen3.5:9b-q4_K_M"))[0] == "rag"
    with pytest.raises(KeyError):
        await registry.resolve("chat/ollama/embeddinggemma:latest")
    before = len(requests)
    assert await registry.list_models() == listed
    assert len(requests) == before  # Repeat /models calls use the TTL cache.


@pytest.mark.asyncio
async def test_openai_compatible_vllm_discovery(monkeypatch):
    def handler(request: httpx.Request):
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": "mlx-community/Qwen3.5-9B-MLX-4bit"}]})

    transport = httpx.MockTransport(handler)
    original = httpx.AsyncClient
    monkeypatch.setattr("localrag.discovery.httpx.AsyncClient",
                        lambda **kwargs: original(transport=transport, **kwargs))
    config = Config.model_validate({"llm": {"default": "vllm-metal", "providers": [{
        "id": "vllm-metal", "provider": "openai", "enabled": True,
        "discover": True, "base_url": "http://fake:8001/v1",
    }]}})
    registry = ModelRegistry(config)
    listed = await registry.list_models()
    assert [item["id"] for item in listed] == [
        "chat/vllm-metal/mlx-community/Qwen3.5-9B-MLX-4bit",
        "rag/vllm-metal/mlx-community/Qwen3.5-9B-MLX-4bit"]
    assert (await registry.resolve(listed[0]["id"]))[2] == "mlx-community/Qwen3.5-9B-MLX-4bit"


@pytest.mark.asyncio
async def test_unavailable_vllm_does_not_hide_static_ollama(monkeypatch):
    transport = httpx.MockTransport(lambda request: httpx.Response(503))
    original = httpx.AsyncClient
    monkeypatch.setattr("localrag.discovery.httpx.AsyncClient",
                        lambda **kwargs: original(transport=transport, **kwargs))
    config = Config.model_validate({"llm": {"default": "ollama", "providers": [
        {"id": "ollama", "model": "qwen3.5:9b-q4_K_M", "discover": False},
        {"id": "cluster", "provider": "openai", "discover": True, "enabled": True,
         "base_url": "http://offline:8001/v1"},
    ]}})
    listed = await ModelRegistry(config).list_models()
    assert [entry["id"] for entry in listed] == ["chat/ollama", "rag/ollama"]
