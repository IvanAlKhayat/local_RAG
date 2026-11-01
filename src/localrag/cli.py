"""Small operational CLI; web UI is supplied by Open WebUI."""

from __future__ import annotations

import argparse
import asyncio
import json

from .config import load_config
from .engine import RAGEngine


def main() -> None:
    parser = argparse.ArgumentParser(prog="localrag", description="Local-first RAG operations")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Run the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    sync = commands.add_parser("sync", help="Ingest one configured source")
    sync.add_argument("source_id")
    commands.add_parser("reindex", help="Rebuild all sources after changing embedding settings")
    commands.add_parser("status", help="Show configured source document counts")
    args = parser.parse_args()

    if args.command == "serve":
        import uvicorn
        uvicorn.run("localrag.api:app", host=args.host, port=args.port)
        return

    config = load_config()
    engine = RAGEngine(config)
    if args.command == "status":
        documents = engine.storage.documents()
        counts = {source.id: sum(doc["source_id"] == source.id for doc in documents)
                  for source in config.sources}
        print(json.dumps(counts, indent=2))
    elif args.command == "sync":
        print(json.dumps(asyncio.run(engine.sync_source(args.source_id)), indent=2))
    elif args.command == "reindex":
        # Reset before re-ingestion only when explicitly requested by the user.
        engine.reset_index()
        for source in config.sources:
            if source.enabled:
                print(json.dumps({source.id: asyncio.run(engine.sync_source(source.id))}, indent=2))


if __name__ == "__main__":
    main()
