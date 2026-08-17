"""Architecture step 10: Context Building — Chroma-backed RAG for Domain
knowledge and Company/Platform knowledge (incl. the 4X function/operator
library). Two separate collections since the two sources are retrieved
independently and conditionally.

Embedding function: Chroma's default embedding function downloads an ONNX
model from the internet on first use, which fails in network-restricted
environments (this was discovered while building this project — it failed
in the sandboxed build environment and will fail identically on any
corporate network that blocks that download). To keep this pipeline fully
functional offline and deterministic in tests, a simple hashing-based
embedding function is used by default. It requires no network access and no
extra model weights, but it is a crude bag-of-words style embedding, not a
real semantic model. For production-quality semantic retrieval, pass a real
embedding_function (e.g. sentence-transformers loaded from a local path, or
an API-based embedding service) into ChromaStore.
"""
from __future__ import annotations

import hashlib

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from app.utils.config import settings

DOMAIN_COLLECTION = "domain_rag"
PLATFORM_COLLECTION = "platform_rag"

_VECTOR_DIM = 256


class HashingEmbeddingFunction(EmbeddingFunction):
    """Deterministic, dependency-free embedding via feature hashing.
    Not a substitute for a real semantic embedding model — see module
    docstring — but keeps RAG functional with zero network dependency."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def name() -> str:
        return "hashing-embedding-v1"

    def get_config(self) -> dict:
        return {"vector_dim": _VECTOR_DIM}

    @staticmethod
    def build_from_config(config: dict) -> "HashingEmbeddingFunction":
        return HashingEmbeddingFunction()

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed(text) for text in input]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * _VECTOR_DIM
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % _VECTOR_DIM
            vector[idx] += 1.0
        norm = sum(v * v for v in vector) ** 0.5
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector


class ChromaStore:
    def __init__(self, persist_dir: str | None = None, embedding_function=None):
        self._client = chromadb.PersistentClient(path=persist_dir or settings.chroma_persist_dir)
        self._embedding_function = embedding_function or HashingEmbeddingFunction()
        self._collection_cache: dict[str, object] = {}
        # Query results are pure functions of (collection contents,
        # query_text, n_results) given the deterministic hashing embedding
        # function, so caching by that key is always correct as long as the
        # cache is cleared whenever a collection's contents change -- see
        # add_documents(). This matters because the same query_text is
        # frequently repeated: app/derivation/dd_generator.py's per-column
        # RAG lookup issues the same domain-collection query (derived from
        # the chain's business_summary, which is constant for every column
        # in that chain) once per column, so without this cache a chain
        # with N columns does N redundant embed+search round trips for a
        # single distinct query.
        self._query_cache: dict[tuple[str, str, int], list[str]] = {}

    def _collection(self, name: str):
        cached = self._collection_cache.get(name)
        if cached is not None:
            return cached
        col = self._client.get_or_create_collection(name, embedding_function=self._embedding_function)
        self._collection_cache[name] = col
        return col

    def add_documents(self, collection: str, documents: list[str], ids: list[str], metadatas: list[dict] | None = None) -> None:
        self._collection(collection).add(documents=documents, ids=ids, metadatas=metadatas)
        # Invalidate any cached query results for this collection -- its
        # contents just changed, so a previously-cached answer could now be
        # stale (e.g. an empty-result cache entry from before anything was
        # ingested).
        stale_keys = [key for key in self._query_cache if key[0] == collection]
        for key in stale_keys:
            del self._query_cache[key]

    def query(self, collection: str, query_text: str, n_results: int = 3) -> list[str]:
        cache_key = (collection, query_text, n_results)
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            return cached

        col = self._collection(collection)
        count = col.count()
        if count == 0:
            self._query_cache[cache_key] = []
            return []
        results = col.query(query_texts=[query_text], n_results=min(n_results, count))
        docs = results.get("documents", [[]])
        found = docs[0] if docs else []
        self._query_cache[cache_key] = found
        return found

    def count(self, collection: str) -> int:
        return self._collection(collection).count()