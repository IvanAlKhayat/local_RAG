from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class AppConfig(BaseModel):
    title: str = "LocalRAG"
    data_dir: str = "./data"
    webui_url: str = "http://127.0.0.1:8080"
    max_upload_mb: int = Field(default=25, ge=1, le=500)


class ChatProvider(BaseModel):
    """One selectable generation backend (not an agent/tool provider)."""

    id: str
    provider: Literal["ollama", "openai"] = "ollama"
    enabled: bool = True
    discover: bool = False  # Query the provider's model list instead of fixing one model.
    base_url: str = "http://localhost:11434"
    api_key_env: str = ""  # Environment variable name, never the key itself.
    model: str = "qwen3.5:9b-q4_K_M"
    context_length: int = Field(default=8192, ge=1024, le=262144)
    temperature: float = Field(default=0.2, ge=0, le=2)
    thinking: bool | None = False  # False disables thinking where supported; None omits the parameter.

    @field_validator("id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        if not value or not all(c.isascii() and (c.isalnum() or c in "_-") for c in value):
            raise ValueError("Provider IDs must use ASCII letters, digits, _ or -")
        return value


class LLMConfig(BaseModel):
    default: str = "ollama"
    providers: list[ChatProvider] = Field(default_factory=lambda: [ChatProvider(
        id="ollama", discover=True
    )])

    @model_validator(mode="after")
    def check_providers(self):
        ids = [item.id for item in self.providers]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate chat provider IDs")
        if self.default not in ids or not self.get_provider(self.default).enabled:
            raise ValueError("The default chat provider must exist and be enabled")
        return self

    def get_provider(self, provider_id: str) -> ChatProvider:
        for item in self.providers:
            if item.id == provider_id and item.enabled:
                return item
        raise KeyError(f"Chat provider {provider_id!r} is unknown or disabled")


class EmbeddingsConfig(BaseModel):
    provider: Literal["ollama", "openai"] = "ollama"
    base_url: str = "http://localhost:11434"
    api_key_env: str = ""
    model: str = "embeddinggemma"
    batch_size: int = Field(default=8, ge=1, le=64)


class RetrievalConfig(BaseModel):
    mode: Literal["hybrid", "dense", "sparse"] = "hybrid"
    dense_top_k: int = Field(default=20, ge=1, le=100)
    sparse_top_k: int = Field(default=20, ge=1, le=100)
    fusion_k: int = Field(default=60, ge=1, le=1000)
    final_top_k: int = Field(default=5, ge=1, le=30)
    chunk_chars: int = Field(default=1300, ge=200, le=10000)
    chunk_overlap: int = Field(default=180, ge=0, le=2000)

    @model_validator(mode="after")
    def check_overlap(self):
        if self.chunk_overlap >= self.chunk_chars:
            raise ValueError("chunk_overlap must be smaller than chunk_chars")
        return self


class RerankerConfig(BaseModel):
    enabled: bool = False
    provider: Literal["sentence_transformers"] = "sentence_transformers"
    model: str = "BAAI/bge-reranker-v2-m3"
    device: Literal["cpu", "mps", "cuda"] = "cpu"


class SourceConfig(BaseModel):
    id: str
    name: str | None = None
    type: Literal["local", "upload", "google_drive"]
    enabled: bool = True
    path: str | None = None
    recursive: bool = True
    extensions: list[str] = Field(default_factory=lambda: [".pdf", ".docx", ".md", ".txt"])
    folder_ids: list[str] = Field(default_factory=list)
    credentials_file: str = "./credentials.json"
    token_file: str = "./data/google_token.json"

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not value or not all(ch.isascii() and (ch.isalnum() or ch in "_-") for ch in value):
            raise ValueError("Source IDs must use ASCII letters, digits, _ or -")
        return value

    @model_validator(mode="after")
    def validate_source(self):
        if self.type == "local" and not self.path:
            raise ValueError(f"Local source {self.id} requires path")
        return self


class Config(BaseModel):
    """Validated, checked-in-example settings; never persist secrets to YAML."""
    app: AppConfig = Field(default_factory=AppConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    sources: list[SourceConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_sources(self):
        ids = [s.id for s in self.sources]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate source IDs")
        if len([s for s in self.sources if s.type == "upload"]) > 1:
            raise ValueError("At most one upload source is supported")
        return self

    def resolve_path(self, path: str) -> Path:
        # Paths belong to the chosen config file, not to Python's site-packages.
        # This is critical when the package is installed as a wheel in Docker.
        p = Path(path).expanduser()
        return (p if p.is_absolute() else config_path().parent / p).resolve()

    @property
    def data_path(self) -> Path:
        return self.resolve_path(self.app.data_dir)

    def get_source(self, source_id: str) -> SourceConfig:
        for source in self.sources:
            if source.id == source_id:
                return source
        raise KeyError(source_id)


def config_path() -> Path:
    provided = os.getenv("LOCALRAG_CONFIG", "./config.yaml")
    p = Path(provided).expanduser()
    return (p if p.is_absolute() else Path.cwd() / p).resolve()


def load_config() -> Config:
    path = config_path()
    if not path.is_file():
        path = path.parent / "config.example.yaml"
    if not path.is_file():
        raise FileNotFoundError("Missing config.yaml; copy config.example.yaml to config.yaml")
    config = Config.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    # Docker reaches host model servers through host.docker.internal; native
    # installs use localhost. Overrides avoid a second, duplicated YAML file.
    for provider in config.llm.providers:
        key = "LOCALRAG_URL_" + provider.id.upper().replace("-", "_")
        if os.getenv(key):
            provider.base_url = os.environ[key]
    if os.getenv("LOCALRAG_EMBEDDING_URL"):
        config.embeddings.base_url = os.environ["LOCALRAG_EMBEDDING_URL"]
    return config


