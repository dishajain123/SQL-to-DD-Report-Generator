"""Loads reference documents into Chroma: the 4X function/operator library
(Platform RAG) and a banking/NPA/ECL domain glossary (Domain RAG).
"""
from __future__ import annotations

from pathlib import Path

from app.rag.chroma_store import ChromaStore, DOMAIN_COLLECTION, PLATFORM_COLLECTION


def chunk_text(text: str, max_chars: int = 800) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paragraphs:
        if len(buf) + len(p) + 2 > max_chars and buf:
            chunks.append(buf)
            buf = p
        else:
            buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        chunks.append(buf)
    return chunks


def ingest_platform_doc(store: ChromaStore, file_path: str | Path) -> int:
    text = Path(file_path).read_text(encoding="utf-8")
    chunks = chunk_text(text)
    ids = [f"platform-{Path(file_path).stem}-{i}" for i in range(len(chunks))]
    store.add_documents(PLATFORM_COLLECTION, chunks, ids)
    return len(chunks)


def ingest_domain_doc(store: ChromaStore, file_path: str | Path) -> int:
    text = Path(file_path).read_text(encoding="utf-8")
    chunks = chunk_text(text)
    ids = [f"domain-{Path(file_path).stem}-{i}" for i in range(len(chunks))]
    store.add_documents(DOMAIN_COLLECTION, chunks, ids)
    return len(chunks)
