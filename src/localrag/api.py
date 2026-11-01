"""Operational HTTP API: source discovery, indexing jobs and document uploads.

Open WebUI owns the conversation UI; see openai_api.py for its model protocol.
This is a single-user/local-first service, intentionally without public auth.
"""

from __future__ import annotations

import asyncio
import html
import os
import re
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from .config import load_config
from .engine import RAGEngine, safe_local_file
from .openai_api import router as model_router
from .parsing import SUPPORTED

app = FastAPI(title="LocalRAG", version="1.2.1", description="Local-first chat and RAG; not an agent")
app.include_router(model_router)
app.state.config = load_config()
app.state.engine = RAGEngine(app.state.config)
jobs: dict[str, dict] = {}


def start_job(work):
    """Run ingestion in the background, reporting progress by opaque job ID."""
    job_id = uuid.uuid4().hex
    jobs[job_id] = {"status": "running", "total": 0, "done": 0, "indexed": 0, "failed": []}

    async def runner():
        try:
            result = await work(lambda **values: jobs[job_id].update(values))
            jobs[job_id].update(status="complete", result=result)
        except Exception as exc:
            jobs[job_id].update(status="error", error=str(exc))

    asyncio.create_task(runner())
    return {"job_id": job_id}


@app.get("/health")
@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "1.2.1", "type": "rag"}


@app.get("/manage", include_in_schema=False)
async def management_page():
    """Use a deployment-specific chat URL for Docker, native and remote UI."""
    template = (Path(__file__).parent / "static" / "manage.html").read_text(encoding="utf-8")
    destination = os.getenv("LOCALRAG_WEBUI_URL", app.state.config.app.webui_url)
    return HTMLResponse(template.replace("__LOCALRAG_WEBUI_URL__", html.escape(destination, quote=True)))


@app.get("/api/sources")
async def sources():
    config = app.state.config
    counts: dict[str, int] = {}
    for document in app.state.engine.storage.documents():
        counts[document["source_id"]] = counts.get(document["source_id"], 0) + 1
    return [dict(item.model_dump(mode="json"), document_count=counts.get(item.id, 0))
            for item in config.sources]


@app.get("/api/documents")
async def documents(source_id: str | None = None):
    return app.state.engine.storage.documents(source_id)


@app.get("/api/documents/{doc_id}/open")
async def open_document(doc_id: str):
    engine = app.state.engine
    document = engine.storage.get_document(doc_id)
    if document is None:
        raise HTTPException(404, "Document not found")
    source = app.state.config.get_source(document["source_id"])
    if not source.enabled:
        raise HTTPException(403, "Source disabled")
    if source.type == "google_drive":
        if not document.get("remote_id"):
            raise HTTPException(404, "Missing remote ID")
        return RedirectResponse(f"https://drive.google.com/open?id={document['remote_id']}")
    try:
        target = safe_local_file(engine.source_root(source), document["relative_path"])
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return FileResponse(target, filename=document["title"], content_disposition_type="inline")


@app.get("/api/sources/{source_id}/browse")
async def browse(source_id: str, path: str = Query(default="")):
    """Browse an allowed local directory or the indexed Google Drive hierarchy."""
    try:
        source = app.state.config.get_source(source_id)
    except KeyError as exc:
        raise HTTPException(404, "Unknown source") from exc
    if not source.enabled:
        raise HTTPException(403, "Source disabled")
    engine = app.state.engine
    known = {d["relative_path"]: d["id"] for d in engine.storage.documents(source_id)}
    if source.type == "google_drive":
        prefix = path.strip("/")
        entries = {}
        for relative, doc_id in known.items():
            if prefix and not relative.startswith(prefix + "/"):
                continue
            remainder = relative[len(prefix) + 1:] if prefix else relative
            first = remainder.split("/", 1)[0]
            entries[first] = {"name": first, "path": f"{prefix}/{first}".strip("/"),
                              "is_directory": "/" in remainder,
                              "doc_id": doc_id if "/" not in remainder else None,
                              "indexed": "/" not in remainder}
        return sorted(entries.values(), key=lambda x: (not x["is_directory"], x["name"].lower()))
    root = engine.source_root(source)
    candidate = root / path
    if not candidate.resolve().is_relative_to(root.resolve()) or not candidate.is_dir():
        raise HTTPException(400, "Invalid directory")
    current = candidate
    while current != root:
        if current.is_symlink():
            raise HTTPException(400, "Symlinks are not allowed")
        current = current.parent
    allowed = {ext.lower() for ext in source.extensions} & SUPPORTED
    entries = []
    for child in sorted(candidate.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        if child.is_symlink() or child.name.startswith("."):
            continue
        if child.is_file() and child.suffix.lower() not in allowed:
            continue
        relative = str(child.relative_to(root))
        entries.append({"name": child.name, "path": relative, "is_directory": child.is_dir(),
                        "doc_id": known.get(relative), "indexed": relative in known})
    return entries


@app.post("/api/sources/{source_id}/sync")
async def sync_source(source_id: str):
    try:
        source = app.state.config.get_source(source_id)
    except KeyError as exc:
        raise HTTPException(404, "Unknown source") from exc
    if not source.enabled:
        raise HTTPException(400, "Source disabled")
    if app.state.engine.index_lock.locked():
        raise HTTPException(409, "Indexing already running")
    return start_job(lambda update: app.state.engine.sync_source(source_id, update))


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Unknown job")
    return jobs[job_id]


@app.post("/api/reindex")
async def reindex():
    engine = app.state.engine
    if engine.index_lock.locked():
        raise HTTPException(409, "Indexing already running")

    async def rebuild(update):
        async with engine.index_lock:
            engine.reset_index()
        results = []
        for source in app.state.config.sources:
            if source.enabled:
                results.append(await engine.sync_source(source.id, update))
        return results

    return start_job(rebuild)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """Upload directly to the RAG index; Open WebUI attachments are separate."""
    source = next((s for s in app.state.config.sources if s.type == "upload" and s.enabled), None)
    if source is None:
        raise HTTPException(400, "No enabled upload source")
    name = re.sub(r"[^\w.() -]", "_", Path(file.filename or "document").name).strip(". ")[:120]
    if Path(name).suffix.lower() not in SUPPORTED:
        raise HTTPException(400, f"Supported: {', '.join(sorted(SUPPORTED))}")
    target = app.state.engine.source_root(source) / f"{uuid.uuid4().hex[:8]}_{name}"
    size = 0
    try:
        with target.open("wb") as output:
            while block := await file.read(1024 * 1024):
                size += len(block)
                if size > app.state.config.app.max_upload_mb * 1024 * 1024:
                    raise HTTPException(413, "File too large")
                output.write(block)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    return {"filename": target.name,
            "job": start_job(lambda update: app.state.engine.sync_source(source.id, update))}
