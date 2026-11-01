"""Discover chat-capable models without coupling the RAG engine to a UI.

Ollama models come from /api/tags and their capabilities from /api/show.
OpenAI-compatible model servers, including vLLM, expose /v1/models.
"""

from __future__ import annotations

import time

import httpx

from .config import ChatProvider, Config
from .providers import authorization_headers


class ModelRegistry:
    """Small TTL cache of models exposed by enabled generation servers."""

    def __init__(self, config: Config, ttl_seconds: int = 30):
        self.config = config
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, tuple[float, list[str]]] = {}

    async def available(self, provider: ChatProvider) -> list[str]:
        """Return actual provider model names, never embedding-only models."""
        if not provider.discover:
            return [provider.model]

        cached = self._cache.get(provider.id)
        if cached and time.monotonic() - cached[0] < self.ttl_seconds:
            return cached[1]

        base = provider.base_url.rstrip("/")
        names: list[str] = []
        try:
            async with httpx.AsyncClient(timeout=12, headers=authorization_headers(provider.api_key_env)) as client:
                if provider.provider == "ollama":
                    response = await client.get(f"{base}/api/tags")
                    response.raise_for_status()
                    for item in response.json().get("models", []):
                        name = item.get("name") or item.get("model")
                        if not isinstance(name, str) or not name:
                            continue
                        # Ollama's capabilities are more reliable than name-based
                        # guesses such as "embedding*" or "*embed*".
                        try:
                            info = await client.post(f"{base}/api/show", json={"model": name})
                            info.raise_for_status()
                        except httpx.HTTPError:
                            continue
                        if "completion" in info.json().get("capabilities", []):
                            names.append(name)
                else:
                    response = await client.get(f"{base}/models")
                    response.raise_for_status()
                    names = [item["id"] for item in response.json().get("data", [])
                             if isinstance(item.get("id"), str) and item["id"]]
        except (httpx.HTTPError, ValueError, KeyError):
            # A stopped vLLM server must not hide models from working backends.
            names = []

        names = sorted(set(names))
        self._cache[provider.id] = (time.monotonic(), names)
        return names

    async def list_models(self) -> list[dict]:
        """Expose separate chat and grounded-RAG modes for each live model."""
        entries = []
        for provider in self.config.llm.providers:
            if not provider.enabled:
                continue
            names = await self.available(provider)
            for name in names:
                # Preserve the actual model name (including namespace and tag).
                suffix = f"/{name}" if provider.discover else ""
                for mode in ("chat", "rag"):
                    entries.append({"id": f"{mode}/{provider.id}{suffix}", "object": "model",
                                    "created": 0, "owned_by": "localrag"})
        return entries

    async def resolve(self, model_id: str) -> tuple[str, ChatProvider, str]:
        """Resolve an allowed model and its explicit chat/RAG mode.

        Legacy localrag/* identifiers continue to mean strict RAG, so existing
        saved conversations cannot silently become ungrounded chat.
        """
        mode, separator, remainder = model_id.partition("/")
        if not separator or mode not in {"chat", "rag", "localrag"}:
            raise KeyError("Invalid chat/RAG model identifier")
        for provider in self.config.llm.providers:
            if not provider.enabled:
                continue
            prefix = provider.id
            if provider.discover and remainder.startswith(prefix + "/"):
                selected = remainder[len(prefix) + 1:]
                if selected in await self.available(provider):
                    return ("rag" if mode == "localrag" else mode), provider, selected
            elif not provider.discover and remainder == prefix:
                return ("rag" if mode == "localrag" else mode), provider, provider.model
        raise KeyError("Unknown, disabled or unavailable chat/RAG model")
