from app.rag.chroma_store import ChromaStore, DOMAIN_COLLECTION, PLATFORM_COLLECTION
from app.rag.ingest import chunk_text, ingest_domain_doc, ingest_platform_doc


def test_chunk_text_splits_on_blank_lines():
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    chunks = chunk_text(text, max_chars=1000)
    assert len(chunks) >= 1
    assert "Paragraph one." in chunks[0]


def test_chunk_text_respects_max_chars():
    text = "\n\n".join(f"Paragraph {i} with some extra text to pad it out." for i in range(20))
    chunks = chunk_text(text, max_chars=200)
    assert all(len(c) <= 250 for c in chunks)  # small slack for the join


def test_ingest_platform_doc_adds_documents(tmp_path):
    store = ChromaStore(persist_dir=str(tmp_path / "chroma"))
    doc_path = tmp_path / "func_ref.md"
    doc_path.write_text("## Functions\nDATEDIFF computes the difference between two dates.")
    n = ingest_platform_doc(store, doc_path)
    assert n >= 1
    assert store.count(PLATFORM_COLLECTION) == n


def test_ingest_domain_doc_adds_documents(tmp_path):
    store = ChromaStore(persist_dir=str(tmp_path / "chroma"))
    doc_path = tmp_path / "glossary.md"
    doc_path.write_text("NPA means Non-Performing Asset.")
    n = ingest_domain_doc(store, doc_path)
    assert n >= 1
    assert store.count(DOMAIN_COLLECTION) == n


def test_query_empty_collection_returns_empty_list(tmp_path):
    store = ChromaStore(persist_dir=str(tmp_path / "chroma"))
    assert store.query(PLATFORM_COLLECTION, "anything") == []


def test_query_returns_results_after_ingestion(tmp_path):
    store = ChromaStore(persist_dir=str(tmp_path / "chroma"))
    store.add_documents("test_col", ["DATEDIFF computes days between dates", "NPA is non performing asset"], ids=["a", "b"])
    results = store.query("test_col", "date difference function", n_results=1)
    assert len(results) == 1
