"""Minimal model transport: Ollama's native API or the OpenAI HTTP standard.

vLLM, vLLM-Metal and many hosted model servers expose OpenAI-compatible APIs.
No server-specific SDK is required in the RAG application.
"""

from __future__ import annotations

import json
import os

import httpx

from .config import Config


def authorization_headers(env_name: str) -> dict[str, str]:
    """Resolve a configured environment variable without persisting secrets."""
    key = os.getenv(env_name) if env_name else None
    return {"Authorization": f"Bearer {key}"} if key else {}


class ModelClient:
    def __init__(self, config: Config):
        self.config = config

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        cfg = self.config.embeddings
        base = cfg.base_url.rstrip("/")
        vectors: list[list[float]] = []
        async with httpx.AsyncClient(timeout=180, headers=authorization_headers(cfg.api_key_env)) as client:
            for index in range(0, len(texts), cfg.batch_size):
                batch = texts[index:index + cfg.batch_size]
                if cfg.provider == "ollama":
                    response = await client.post(f"{base}/api/embed", json={"model": cfg.model, "input": batch})
                    response.raise_for_status()
                    vectors.extend(response.json()["embeddings"])
                else:
                    response = await client.post(f"{base}/embeddings", json={"model": cfg.model, "input": batch})
                    response.raise_for_status()
                    data = sorted(response.json()["data"], key=lambda row: row["index"])
                    vectors.extend(row["embedding"] for row in data)
        return vectors

    async def chat_stream(self, messages: list[dict], provider_id: str | None = None,
                          model_name: str | None = None):
        """Yield only visible text; upstream reasoning fields are ignored."""
        cfg = self.config.llm.get_provider(provider_id or self.config.llm.default)
        selected_model = model_name or cfg.model
        base = cfg.base_url.rstrip("/")
        if cfg.provider == "ollama":
            url = f"{base}/api/chat"
            payload = {"model": selected_model, "messages": messages, "stream": True,
                       "options": {"num_ctx": cfg.context_length, "temperature": cfg.temperature}}
            if cfg.thinking is not None:
                payload["think"] = cfg.thinking
        else:
            url = f"{base}/chat/completions"
            payload = {"model": selected_model, "messages": messages, "stream": True,
                       "temperature": cfg.temperature, "max_tokens": min(2048, cfg.context_length // 2)}
            if cfg.thinking is not None:
                # vLLM's OpenAI-compatible server accepts chat template kwargs.
                # Set thinking: null for a server that does not recognize them.
                payload["chat_template_kwargs"] = {"enable_thinking": cfg.thinking}

        async with httpx.AsyncClient(timeout=httpx.Timeout(20, read=600),
                                     headers=authorization_headers(cfg.api_key_env)) as client:
            async with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if cfg.provider == "ollama":
                        if not line:
                            continue
                        packet = json.loads(line)
                        content = packet.get("message", {}).get("content")
                        if content:
                            yield content
                        if packet.get("done"):
                            break
                    elif line.startswith("data:"):
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        if not data:
                            continue
                        packet = json.loads(data)
                        for choice in packet.get("choices", []):
                            content = choice.get("delta", {}).get("content")
                            if isinstance(content, str) and content:
                                yield content
