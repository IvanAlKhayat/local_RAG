from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from docx import Document
from pypdf import PdfReader

SUPPORTED = {".pdf", ".docx", ".md", ".txt"}


@dataclass(frozen=True)
class Passage:
    text: str
    page: int | None = None


def parse_file(path: Path) -> list[Passage]:
    """Extract text only. Scanned PDFs/images require OCR (not included)."""
    extension = path.suffix.lower()
    if extension == ".pdf":
        reader = PdfReader(str(path))
        return [Passage(text=page.extract_text() or "", page=i + 1) for i, page in enumerate(reader.pages)]
    if extension == ".docx":
        doc = Document(str(path))
        blocks = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                blocks.append(" | ".join(cell.text for cell in row.cells))
        return [Passage("\n".join(blocks))]
    if extension in {".md", ".txt"}:
        return [Passage(path.read_text(encoding="utf-8", errors="replace"))]
    raise ValueError(f"Unsupported file extension: {extension}")


def chunk_passages(passages: list[Passage], size: int, overlap: int) -> list[Passage]:
    """Character-bounded, paragraph-aware chunks preserving PDF page numbers."""
    result: list[Passage] = []
    for passage in passages:
        normalized = re.sub(r"[ \t]+", " ", passage.text).strip()
        if not normalized:
            continue
        start = 0
        while start < len(normalized):
            end = min(start + size, len(normalized))
            if end < len(normalized):
                boundary = max(normalized.rfind("\n", start + size // 2, end),
                               normalized.rfind(" ", start + size // 2, end))
                if boundary > start:
                    end = boundary
            snippet = normalized[start:end].strip()
            if snippet:
                result.append(Passage(snippet, passage.page))
            if end == len(normalized):
                break
            start = max(start + 1, end - overlap)
    return result


def stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
