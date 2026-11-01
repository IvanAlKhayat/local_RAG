from __future__ import annotations

import sqlite3
from pathlib import Path


class Storage:
    """Document metadata and lexical search stored in portable SQLite tables."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.initialize()

    def connect(self):
        con = sqlite3.connect(self.path, timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
        return con

    def initialize(self):
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    title TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    remote_id TEXT,
                    indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_id);
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    content TEXT NOT NULL,
                    page INTEGER
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED, content, tokenize='unicode61'
                );
            """)

    def documents(self, source_id: str | None = None):
        with self.connect() as db:
            sql = "SELECT * FROM documents"
            if source_id:
                return [dict(r) for r in db.execute(sql + " WHERE source_id=? ORDER BY relative_path", (source_id,))]
            return [dict(r) for r in db.execute(sql + " ORDER BY source_id, relative_path")]

    def get_document(self, doc_id: str):
        with self.connect() as db:
            row = db.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
            return dict(row) if row else None

    def chunk_ids(self, doc_id: str):
        with self.connect() as db:
            return [row[0] for row in db.execute("SELECT id FROM chunks WHERE doc_id=?", (doc_id,))]

    def get_chunks(self, chunk_ids: list[str]):
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        with self.connect() as db:
            rows = db.execute(f"""
                SELECT c.id, c.content, c.page, d.id doc_id, d.source_id,
                       d.relative_path, d.title FROM chunks c
                JOIN documents d ON d.id=c.doc_id
                WHERE c.id IN ({placeholders})
            """, chunk_ids).fetchall()
        return {row["id"]: dict(row) for row in rows}

    def clear_index(self):
        with self.connect() as db:
            db.execute("DELETE FROM chunks_fts")
            db.execute("DELETE FROM documents")

    def delete_document(self, db: sqlite3.Connection, doc_id: str) -> list[str]:
        ids = [row[0] for row in db.execute("SELECT id FROM chunks WHERE doc_id=?", (doc_id,))]
        for chunk_id in ids:
            db.execute("DELETE FROM chunks_fts WHERE chunk_id=?", (chunk_id,))
        db.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        return ids

    def replace_document(self, doc: dict, chunks: list[dict]):
        """Replace source metadata and FTS rows in a single transaction."""
        with self.connect() as db:
            self.delete_document(db, doc["id"])
            db.execute("INSERT INTO documents(id,source_id,relative_path,title,fingerprint,remote_id) VALUES(?,?,?,?,?,?)",
                       (doc["id"], doc["source_id"], doc["relative_path"], doc["title"], doc["fingerprint"], doc.get("remote_id")))
            for chunk in chunks:
                db.execute("INSERT INTO chunks(id,doc_id,content,page) VALUES(?,?,?,?)",
                           (chunk["id"], doc["id"], chunk["text"], chunk["page"]))
                db.execute("INSERT INTO chunks_fts(chunk_id,content) VALUES(?,?)", (chunk["id"], chunk["text"]))

    def sparse_search(self, query: str, limit: int, allowed: set[str] | None = None):
        """BM25 lexical ranking, using quoted terms to avoid FTS query syntax."""
        import re
        terms = re.findall(r"\w+", query, flags=re.UNICODE)[:12]
        if not terms:
            return []
        expression = " OR ".join('"' + t.replace('"', '""') + '"' for t in terms)
        with self.connect() as db:
            rows = db.execute("""
                SELECT c.id, c.content, c.page, d.id doc_id, d.source_id,
                       d.relative_path, d.title, bm25(chunks_fts) rank
                FROM chunks_fts
                JOIN chunks c ON c.id=chunks_fts.chunk_id
                JOIN documents d ON d.id=c.doc_id
                WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?
            """, (expression, max(limit * 10, 100) if allowed else limit)).fetchall()
        results = [dict(row) for row in rows]
        if allowed is not None:
            results = [row for row in results if row["source_id"] in allowed]
        return results[:limit]

