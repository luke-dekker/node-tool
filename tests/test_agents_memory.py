"""Phase B Memory — DOCUMENT / EMBEDDING / MEMORY_REF port types, plus
DocumentLoaderNode / EmbedderNode / MemoryStoreNode / RetrieverNode.

Heavy-dep parts (real Qdrant round-trip) are gated behind module
availability; the main invariant is that register + every execute path
uses deferred imports so the suite runs with neither qdrant_client nor
sentence_transformers installed.
"""
from __future__ import annotations
import importlib.util
import json
import os
import tempfile

import pytest


HAS_QDRANT = importlib.util.find_spec("qdrant_client") is not None


@pytest.fixture(autouse=True)
def _register_agent_port_types():
    from plugins.agents.port_types import register_all
    register_all()


# ── port types ─────────────────────────────────────────────────────────────

def test_memory_port_types_registered():
    from core.port_types import PortTypeRegistry
    for t in ("DOCUMENT", "EMBEDDING", "MEMORY_REF"):
        assert PortTypeRegistry.get(t) is not None


def test_memory_modules_import_without_backends():
    """Importing _memory.* must not pull qdrant_client / sentence_transformers."""
    import plugins.agents._memory.store_protocol as sp
    import plugins.agents._memory.embedder as emb
    import plugins.agents._memory.qdrant_backend as qb
    assert hasattr(sp, "Document")
    assert hasattr(sp, "Embedding")
    assert hasattr(sp, "MemoryRef")
    assert hasattr(emb, "SentenceTransformerEmbedder")
    assert hasattr(emb, "HashEmbedder")
    assert hasattr(qb, "QdrantBackend")


def test_plugin_registers_memory_nodes():
    from core.plugins import PluginContext
    import plugins.agents as agents_pkg

    ctx = PluginContext()
    agents_pkg.register(ctx)
    names = {c.type_name for c in ctx.node_classes}
    for t in ("ag_document_loader", "ag_embedder", "ag_memory_store", "ag_retriever"):
        assert t in names


# ── Document / stable_id ───────────────────────────────────────────────────

def test_document_stable_id_deterministic():
    from plugins.agents._memory.store_protocol import Document, stable_id
    a = Document(text="hello world", metadata={"source": "a.md", "chunk_idx": 0})
    b = Document(text="hello world", metadata={"source": "a.md", "chunk_idx": 0})
    assert a.id == b.id
    assert a.id == stable_id("hello world", {"source": "a.md", "chunk_idx": 0})


def test_document_different_metadata_different_id():
    from plugins.agents._memory.store_protocol import Document
    a = Document(text="x", metadata={"source": "a"})
    b = Document(text="x", metadata={"source": "b"})
    assert a.id != b.id


def test_embedding_rejects_dim_mismatch():
    from plugins.agents._memory.store_protocol import Embedding
    with pytest.raises(ValueError, match="dim mismatch"):
        Embedding(vector=[0.1, 0.2, 0.3], dim=5)


# ── DocumentLoaderNode ─────────────────────────────────────────────────────

def test_document_loader_chunks_inline_text():
    from nodes.agents.document_loader import DocumentLoaderNode
    n = DocumentLoaderNode()
    # 100 chars, chunk=30, overlap=10 → step=20 → chunks start at 0,20,40,60,80
    text = "abcdefghij" * 10
    out = n.execute({
        "path": "", "text": text, "chunk_size": 30, "chunk_overlap": 10,
    })
    docs = out["documents"]
    assert len(docs) == 5
    assert docs[0].text == text[0:30]
    assert docs[1].text == text[20:50]
    assert docs[-1].text == text[80:100]
    for i, d in enumerate(docs):
        assert d.metadata["chunk_idx"] == i
        assert d.metadata["source"] == "<inline>"


def test_document_loader_reads_file(tmp_path):
    from nodes.agents.document_loader import DocumentLoaderNode
    p = tmp_path / "sample.md"
    p.write_text("# Title\n" + ("word " * 200), encoding="utf-8")
    n = DocumentLoaderNode()
    out = n.execute({
        "path": str(p), "text": "", "chunk_size": 200, "chunk_overlap": 20,
    })
    docs = out["documents"]
    assert len(docs) >= 2
    assert all(d.metadata["source"] == str(p) for d in docs)


def test_document_loader_missing_path_raises():
    from nodes.agents.document_loader import DocumentLoaderNode
    n = DocumentLoaderNode()
    with pytest.raises(RuntimeError, match="file not found"):
        n.execute({
            "path": "C:/definitely/not/a/real/path.txt", "text": "",
            "chunk_size": 100, "chunk_overlap": 0,
        })


def test_document_loader_empty_returns_empty():
    from nodes.agents.document_loader import DocumentLoaderNode
    n = DocumentLoaderNode()
    out = n.execute({"path": "", "text": "   \n\n   ",
                     "chunk_size": 100, "chunk_overlap": 0})
    assert out["documents"] == []


def test_document_loader_ids_stable_across_runs(tmp_path):
    from nodes.agents.document_loader import DocumentLoaderNode
    p = tmp_path / "stable.txt"
    p.write_text("a" * 500, encoding="utf-8")
    n = DocumentLoaderNode()
    args = {"path": str(p), "text": "", "chunk_size": 100, "chunk_overlap": 20}
    ids1 = [d.id for d in n.execute(args)["documents"]]
    ids2 = [d.id for d in n.execute(args)["documents"]]
    assert ids1 == ids2
    assert len(set(ids1)) == len(ids1)  # no collisions


# ── EmbedderNode (via HashEmbedder — no torch dep) ────────────────────────

def test_embedder_hash_backend_produces_normalized_vectors():
    from nodes.agents.embedder import EmbedderNode
    from plugins.agents._memory.store_protocol import Document
    docs = [Document(text="hello world", metadata={"source": "t", "chunk_idx": 0}),
            Document(text="goodbye world", metadata={"source": "t", "chunk_idx": 1})]
    out = EmbedderNode().execute({
        "documents": docs, "texts": "",
        "model": "hash-64", "device": "",
    })
    embs = out["embeddings"]
    assert len(embs) == 2
    assert all(e.dim == 64 for e in embs)
    for e in embs:
        norm_sq = sum(x * x for x in e.vector)
        assert abs(norm_sq - 1.0) < 1e-6
    # Pass-through documents preserved
    assert out["documents"] == docs


def test_embedder_from_raw_text_fallback():
    from nodes.agents.embedder import EmbedderNode
    out = EmbedderNode().execute({
        "documents": None,
        "texts": "line one\nline two\n\nline three",
        "model": "hash-32", "device": "",
    })
    assert len(out["embeddings"]) == 3
    assert len(out["documents"]) == 3
    assert out["documents"][0].text == "line one"


def test_embedder_empty_input_short_circuits():
    from nodes.agents.embedder import EmbedderNode
    out = EmbedderNode().execute({
        "documents": [], "texts": "", "model": "hash-64", "device": "",
    })
    assert out["embeddings"] == []
    assert out["documents"] == []


# ── MemoryStoreNode — error paths (no backend needed) ─────────────────────

def test_memory_store_rejects_unknown_backend():
    from nodes.agents.memory_store import MemoryStoreNode
    n = MemoryStoreNode()
    with pytest.raises(RuntimeError, match="not supported"):
        n.execute({
            "documents": [], "embeddings": [],
            "backend": "pinecone", "path": ".", "collection": "x",
        })


def test_memory_store_empty_returns_ref_without_opening():
    """An empty upsert should NOT touch the backend — lets you set up refs
    in a graph that hasn't loaded docs yet without crashing on missing deps.
    """
    from nodes.agents.memory_store import MemoryStoreNode
    out = MemoryStoreNode().execute({
        "documents": [], "embeddings": [],
        "backend": "qdrant", "path": "./never_created", "collection": "empty",
    })
    ref = out["store_ref"]
    assert ref.backend == "qdrant"
    assert ref.collection == "empty"


def test_memory_store_rejects_mismatched_lengths():
    from nodes.agents.memory_store import MemoryStoreNode
    from plugins.agents._memory.store_protocol import Document, Embedding
    docs = [Document(text="a"), Document(text="b")]
    embs = [Embedding(vector=[1.0, 0.0])]
    with pytest.raises(RuntimeError, match="documents vs"):
        MemoryStoreNode().execute({
            "documents": docs, "embeddings": embs,
            "backend": "qdrant", "path": "./tmp", "collection": "c",
        })


def test_memory_store_rejects_mixed_dims():
    from nodes.agents.memory_store import MemoryStoreNode
    from plugins.agents._memory.store_protocol import Document, Embedding
    docs = [Document(text="a"), Document(text="b")]
    embs = [Embedding(vector=[1.0, 0.0]), Embedding(vector=[1.0, 0.0, 0.0])]
    with pytest.raises(RuntimeError, match="mixed embedding dims"):
        MemoryStoreNode().execute({
            "documents": docs, "embeddings": embs,
            "backend": "qdrant", "path": "./tmp", "collection": "c",
        })


# ── RetrieverNode — type validation (no backend needed) ────────────────────

def test_retriever_requires_query():
    from nodes.agents.retriever import RetrieverNode
    with pytest.raises(RuntimeError, match="query_embedding is required"):
        RetrieverNode().execute({
            "query_embedding": None, "store_ref": object(),
            "k": 3, "where": "",
        })


def test_retriever_requires_store_ref():
    from nodes.agents.retriever import RetrieverNode
    from plugins.agents._memory.store_protocol import Embedding
    with pytest.raises(RuntimeError, match="store_ref is required"):
        RetrieverNode().execute({
            "query_embedding": Embedding(vector=[1.0, 0.0]),
            "store_ref": None, "k": 3, "where": "",
        })


def test_retriever_rejects_wrong_embedding_type():
    from nodes.agents.retriever import RetrieverNode
    from plugins.agents._memory.store_protocol import MemoryRef
    with pytest.raises(RuntimeError, match="must be an Embedding"):
        RetrieverNode().execute({
            "query_embedding": [1.0, 0.0],   # plain list, not Embedding
            "store_ref": MemoryRef(),
            "k": 3, "where": "",
        })


def test_retriever_rejects_invalid_where_json():
    from nodes.agents.retriever import RetrieverNode
    from plugins.agents._memory.store_protocol import Embedding, MemoryRef
    with pytest.raises(RuntimeError, match="invalid where JSON"):
        RetrieverNode().execute({
            "query_embedding": Embedding(vector=[1.0, 0.0]),
            "store_ref": MemoryRef(), "k": 3, "where": "{bad",
        })


# ── End-to-end Qdrant round-trip (skipped if qdrant-client missing) ────────

@pytest.mark.skipif(not HAS_QDRANT, reason="qdrant-client not installed")
def test_memory_pipeline_end_to_end(tmp_path):
    """Load → embed → store → retrieve. Uses HashEmbedder so no torch needed."""
    from nodes.agents.document_loader import DocumentLoaderNode
    from nodes.agents.embedder import EmbedderNode
    from nodes.agents.memory_store import MemoryStoreNode
    from nodes.agents.retriever import RetrieverNode

    corpus = tmp_path / "corpus.md"
    corpus.write_text(
        "Capital of France is Paris. "
        "Python is a programming language. "
        "Qdrant is a vector database. "
        "Torch is a deep learning framework. ",
        encoding="utf-8",
    )
    store_dir = str(tmp_path / "qdrant_data")

    # 1. load
    loaded = DocumentLoaderNode().execute({
        "path": str(corpus), "text": "", "chunk_size": 40, "chunk_overlap": 0,
    })
    docs = loaded["documents"]
    assert len(docs) >= 3

    # 2. embed
    embed_out = EmbedderNode().execute({
        "documents": docs, "texts": "", "model": "hash-128", "device": "",
    })

    # 3. store
    ref = MemoryStoreNode().execute({
        "documents": embed_out["documents"],
        "embeddings": embed_out["embeddings"],
        "backend": "qdrant", "path": store_dir, "collection": "roundtrip",
    })["store_ref"]

    # 4. query with one of the existing chunk texts — should retrieve itself top-1
    query_text = docs[0].text
    q_emb = EmbedderNode().execute({
        "documents": None, "texts": query_text, "model": "hash-128", "device": "",
    })["embeddings"][0]

    hits = RetrieverNode().execute({
        "query_embedding": q_emb, "store_ref": ref, "k": 3, "where": "",
    })
    assert len(hits["documents"]) >= 1
    assert len(hits["scores"]) == len(hits["documents"])
    # Exact-match query: top doc text is a prefix / near-equal
    top = hits["documents"][0]
    assert top.text.strip() == query_text.strip()


@pytest.mark.skipif(not HAS_QDRANT, reason="qdrant-client not installed")
def test_memory_store_idempotent_upsert(tmp_path):
    """Re-running the same store call must not duplicate points."""
    from nodes.agents.embedder import EmbedderNode
    from nodes.agents.memory_store import MemoryStoreNode
    from plugins.agents._memory.store_protocol import MemoryRef, open_store

    store_dir = str(tmp_path / "idem_qdrant")
    texts = "alpha\nbeta\ngamma"
    emb = EmbedderNode().execute({
        "documents": None, "texts": texts, "model": "hash-64", "device": "",
    })

    args = {
        "documents": emb["documents"], "embeddings": emb["embeddings"],
        "backend": "qdrant", "path": store_dir, "collection": "idem",
    }
    MemoryStoreNode().execute(args)
    MemoryStoreNode().execute(args)  # same ids → no-op

    ref = MemoryRef(backend="qdrant", path=store_dir, collection="idem")
    store = open_store(ref)
    try:
        assert store.count() == 3
    finally:
        store.close()


@pytest.mark.skipif(not HAS_QDRANT, reason="qdrant-client not installed")
def test_memory_retriever_where_filter(tmp_path):
    """`where` filter should restrict hits to matching metadata only."""
    from nodes.agents.embedder import EmbedderNode
    from nodes.agents.memory_store import MemoryStoreNode
    from nodes.agents.retriever import RetrieverNode
    from plugins.agents._memory.store_protocol import Document

    store_dir = str(tmp_path / "filter_qdrant")
    docs = [
        Document(text="python snippet A", metadata={"source": "py.md", "chunk_idx": 0}),
        Document(text="python snippet B", metadata={"source": "py.md", "chunk_idx": 1}),
        Document(text="javascript snippet",
                 metadata={"source": "js.md", "chunk_idx": 0}),
    ]
    emb = EmbedderNode().execute({
        "documents": docs, "texts": "", "model": "hash-64", "device": "",
    })
    ref = MemoryStoreNode().execute({
        "documents": emb["documents"], "embeddings": emb["embeddings"],
        "backend": "qdrant", "path": store_dir, "collection": "filter",
    })["store_ref"]

    q = EmbedderNode().execute({
        "documents": None, "texts": "snippet",
        "model": "hash-64", "device": "",
    })["embeddings"][0]
    hits = RetrieverNode().execute({
        "query_embedding": q, "store_ref": ref, "k": 5,
        "where": json.dumps({"source": "py.md"}),
    })
    sources = {d.metadata["source"] for d in hits["documents"]}
    assert sources == {"py.md"}
