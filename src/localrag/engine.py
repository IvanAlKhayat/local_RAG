from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings

from .config import Config, SourceConfig
from .discovery import ModelRegistry
from .google_drive import download_item, get_service, list_items
from .providers import ModelClient
from .parsing import SUPPORTED, chunk_passages, parse_file, sha256_file, stable_id
from .storage import Storage


@dataclass
class FileItem:
    relative_path: str
    doc_id: str
    fingerprint: str
    local_path: Path | None = None
    remote_id: str | None = None
    drive_item: object | None = None


def safe_local_file(root: Path, relative_path: str) -> Path:
    """Never follow symlinks or allow paths to escape a configured root."""
    if not root.is_dir():
        raise ValueError(f"Source directory does not exist: {root}")
    candidate = root / relative_path
    if not candidate.is_file() or candidate.is_symlink():
        raise ValueError("File not found or symlink rejected")
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError("Path outside authorized root")
    current = candidate.parent
    while current != root:
        if current.is_symlink():
            raise ValueError("Symlink directory rejected")
        current = current.parent
    return candidate


class RAGEngine:
    def __init__(self, config: Config):
        self.config = config
        self.config.data_path.mkdir(parents=True, exist_ok=True)
        self.storage = Storage(config.data_path / "localrag.sqlite3")
        self.chroma = chromadb.PersistentClient(
            path=str(config.data_path / "chroma"),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.chroma.get_or_create_collection(name="localrag_chunks")
        self.models = ModelClient(config)
        self.registry = ModelRegistry(config)
        self.index_lock = asyncio.Lock()
        self._cross_encoder = None

    def source_root(self, source: SourceConfig) -> Path:
        if source.type == "upload":
            root = self.config.data_path / "uploads"
            root.mkdir(parents=True, exist_ok=True)
            return root
        if source.type != "local" or not source.path:
            raise ValueError("Not a local source")
        return self.config.resolve_path(source.path)

    def signature(self) -> str:
        payload = {
            "embedding_provider": self.config.embeddings.provider,
            "embedding_base_url": self.config.embeddings.base_url,
            "embedding_model": self.config.embeddings.model,
            "chunk_chars": self.config.retrieval.chunk_chars,
            "chunk_overlap": self.config.retrieval.chunk_overlap,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def verify_signature(self):
        file = self.config.data_path / "index_signature.json"
        if file.is_file() and json.loads(file.read_text()) != {"signature": self.signature()}:
            raise RuntimeError("Embedding/chunk configuration changed: rebuild the index in Settings")

    def save_signature(self):
        (self.config.data_path / "index_signature.json").write_text(
            json.dumps({"signature": self.signature()}), encoding="utf-8")

    def reset_index(self):
        """Rebuild is required after changing the embedding model or chunker."""
        self.chroma.delete_collection("localrag_chunks")
        self.collection = self.chroma.get_or_create_collection(name="localrag_chunks")
        self.storage.clear_index()
        self.save_signature()

    def _local_items(self, source: SourceConfig) -> list[FileItem]:
        root = self.source_root(source)
        if not root.is_dir():
            raise RuntimeError(f"Source path does not exist: {root}")
        candidates = root.rglob("*") if source.recursive else root.glob("*")
        allowed = {x.lower() for x in source.extensions} & SUPPORTED
        found = []
        for path in candidates:
            if any(part.startswith(".") for part in path.relative_to(root).parts):
                continue
            if path.suffix.lower() not in allowed or not path.is_file() or path.is_symlink():
                continue
            relative = str(path.relative_to(root))
            try:
                valid = safe_local_file(root, relative)
            except ValueError:
                continue
            if path.stat().st_size > 100 * 1024 * 1024:
                continue
            found.append(FileItem(
                relative_path=relative,
                doc_id=stable_id(f"{source.id}:{relative}"),
                fingerprint=sha256_file(valid),
                local_path=valid,
            ))
        return found

    async def _index_file(self, source: SourceConfig, item: FileItem, path: Path):
        passages = await asyncio.to_thread(parse_file, path)
        snippets = chunk_passages(passages, self.config.retrieval.chunk_chars, self.config.retrieval.chunk_overlap)
        if not snippets:
            raise ValueError("No extractable text found; scanned files require OCR")
        chunks = [{
            "id": stable_id(f"{item.doc_id}:{idx}:{part.page}:{part.text}"),
            "text": part.text,
            "page": part.page,
        } for idx, part in enumerate(snippets)]
        vectors = await self.models.embed([c["text"] for c in chunks])
        old_ids = self.storage.chunk_ids(item.doc_id)
        new_ids = [c["id"] for c in chunks]
        if chunks:
            await asyncio.to_thread(self.collection.upsert,
                ids=new_ids,
                documents=[c["text"] for c in chunks],
                embeddings=vectors,
                metadatas=[{"doc_id": item.doc_id, "source_id": source.id} for _ in chunks])
        doc = {
            "id": item.doc_id, "source_id": source.id, "relative_path": item.relative_path,
            "title": Path(item.relative_path).name, "fingerprint": item.fingerprint,
            "remote_id": item.remote_id,
        }
        self.storage.replace_document(doc, chunks)
        stale_ids = list(set(old_ids) - set(new_ids))
        if stale_ids:
            await asyncio.to_thread(self.collection.delete, ids=stale_ids)

    async def sync_source(self, source_id: str, update=lambda **kwargs: None):
        source = self.config.get_source(source_id)
        if not source.enabled:
            raise RuntimeError(f"Source {source_id} is disabled in configuration")
        async with self.index_lock:
            self.verify_signature()
            old = {x["id"]: x for x in self.storage.documents(source.id)}
            service = None
            if source.type == "google_drive":
                service = await asyncio.to_thread(get_service, self.config, source)
                remote_items = await asyncio.to_thread(list_items, service, source.folder_ids)
                items = [FileItem(
                    relative_path=x.relative_path,
                    doc_id=stable_id(f"{source.id}:{x.id}"),
                    fingerprint=x.fingerprint,
                    remote_id=x.id,
                    drive_item=x,
                ) for x in remote_items]
            else:
                items = await asyncio.to_thread(self._local_items, source)
            update(total=len(items), done=0, indexed=0, failed=[])
            current = set()
            indexed = 0
            failed = []
            for position, item in enumerate(items, 1):
                current.add(item.doc_id)
                existing = old.get(item.doc_id)
                if existing and existing["fingerprint"] == item.fingerprint and existing["relative_path"] == item.relative_path:
                    update(done=position)
                    continue
                try:
                    if item.drive_item:
                        with tempfile.TemporaryDirectory(prefix="localrag-drive-") as temp:
                            path = Path(temp) / f"download{item.drive_item.extension}"
                            await asyncio.to_thread(download_item, service, item.drive_item, path)
                            await self._index_file(source, item, path)
                    else:
                        await self._index_file(source, item, item.local_path)
                    indexed += 1
                except Exception as exc:
                    failed.append({"path": item.relative_path, "error": str(exc)[:400]})
                update(done=position, indexed=indexed, failed=failed)
            # Remove deleted items only after a completed listing. Keep old versions of failed items.
            deleted = set(old) - current
            for doc_id in deleted:
                ids = self.storage.chunk_ids(doc_id)
                if ids:
                    await asyncio.to_thread(self.collection.delete, ids=ids)
                with self.storage.connect() as db:
                    self.storage.delete_document(db, doc_id)
            self.save_signature()
            return {"indexed": indexed, "removed": len(deleted), "failed": failed, "scanned": len(items)}

    async def retrieve(self, question: str, source_ids: list[str]) -> list[dict]:
        """Fuse dense and BM25 rankings with reciprocal rank fusion (RRF)."""
        self.verify_signature()
        allowed = set(source_ids)
        if not allowed:
            return []
        config = self.config.retrieval
        dense_ids: list[str] = []
        sparse: list[dict] = []
        if config.mode in {"dense", "hybrid"} and self.collection.count():
            vector = (await self.models.embed([question]))[0]
            result = await asyncio.to_thread(self.collection.query,
                query_embeddings=[vector],
                n_results=min(config.dense_top_k, self.collection.count()),
                where={"source_id": {"$in": list(allowed)}},
            )
            dense_ids = result["ids"][0]
        if config.mode in {"sparse", "hybrid"}:
            sparse = self.storage.sparse_search(question, config.sparse_top_k, allowed)

        scores: dict[str, float] = {}
        for ranking in (dense_ids, [entry["id"] for entry in sparse]):
            for rank, chunk_id in enumerate(ranking, 1):
                scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (config.fusion_k + rank)
        ordered_ids = sorted(scores, key=scores.get, reverse=True)
        documents = self.storage.get_chunks(ordered_ids)
        results = [documents[x] for x in ordered_ids if x in documents]

        if self.config.reranker.enabled and results:
            from sentence_transformers import CrossEncoder
            if self._cross_encoder is None:
                self._cross_encoder = await asyncio.to_thread(CrossEncoder,
                    self.config.reranker.model, device=self.config.reranker.device)
            candidates = results[:max(config.final_top_k * 4, 20)]
            values = await asyncio.to_thread(self._cross_encoder.predict,
                [(question, row["content"]) for row in candidates])
            results = [entry for _, entry in sorted(zip(values, candidates), key=lambda x: float(x[0]), reverse=True)]
        return results[:config.final_top_k]
