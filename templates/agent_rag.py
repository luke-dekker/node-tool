"""Agent RAG — retrieval-augmented chat.

    DocumentLoader ─docs─► Embedder ─embeddings──► MemoryStore ─ref──┐
                    │                             │                  │
                    └────docs (passthrough)───────┘                  │
                                                                     ▼
    ChatMessage(user) ─msg─► Conversation ─conv────► Agent         retrieval
                                                     ▲                │
                                    PromptTemplate ──┘                ▼
                                  (inlines retrieved                Retriever
                                   docs into system)

This template wires the full memory pipeline. By default it uses the
`hash-64` embedder — zero third-party deps (no torch) — so the template
loads even without sentence-transformers installed. Switch the Embedder
node's model to `all-MiniLM-L6-v2` once the dep is available for semantic
retrieval.

Required for actual execution (not just canvas load):
  - Ollama running on localhost:11434 (with a pulled model)
  - qdrant-client installed (`pip install qdrant-client`)
  - A source document wired into DocumentLoader's `path` input
"""
from __future__ import annotations
from core.graph import Graph
from templates._helpers import grid

LABEL = "Agent RAG (docs → embed → store → retrieve)"
DESCRIPTION = ("Retrieval-augmented chat. hash embedder by default — "
               "switch to all-MiniLM-L6-v2 for semantic retrieval.")


def build(graph: Graph) -> dict[str, tuple[int, int]]:
    from nodes.agents.ollama_client     import OllamaClientNode
    from nodes.agents.document_loader   import DocumentLoaderNode
    from nodes.agents.embedder          import EmbedderNode
    from nodes.agents.memory_store      import MemoryStoreNode
    from nodes.agents.retriever         import RetrieverNode
    from nodes.agents.chat_message      import ChatMessageNode
    from nodes.agents.conversation      import ConversationNode
    from nodes.agents.agent             import AgentNode
    from nodes.agents.prompt_template   import PromptTemplateNode

    pos = grid(step_x=240, step_y=170)
    positions: dict[str, tuple[int, int]] = {}

    # Row 0 — ingestion pipeline
    loader = DocumentLoaderNode()
    loader.inputs["chunk_size"].default_value    = 256
    loader.inputs["chunk_overlap"].default_value = 32
    graph.add_node(loader); positions[loader.id] = pos(col=0, row=0)

    ingest_embedder = EmbedderNode()
    ingest_embedder.inputs["model"].default_value = "hash-64"
    graph.add_node(ingest_embedder); positions[ingest_embedder.id] = pos(col=1, row=0)

    store = MemoryStoreNode()
    store.inputs["collection"].default_value = "agent_rag_demo"
    graph.add_node(store); positions[store.id] = pos(col=2, row=0)

    # Row 1 — query side
    user = ChatMessageNode()
    user.inputs["role"].default_value    = "user"
    user.inputs["content"].default_value = "What does the document say?"
    graph.add_node(user); positions[user.id] = pos(col=0, row=1)

    query_embedder = EmbedderNode()
    query_embedder.inputs["model"].default_value = "hash-64"
    # Bind the user's content at runtime via `texts` — single-line corpus
    graph.add_node(query_embedder); positions[query_embedder.id] = pos(col=1, row=1)

    retriever = RetrieverNode()
    retriever.inputs["k"].default_value = 3
    graph.add_node(retriever); positions[retriever.id] = pos(col=2, row=1)

    # Row 2 — prompt + agent
    tpl = PromptTemplateNode()
    tpl.inputs["template"].default_value = (
        "You are a concise assistant. Use the context to answer.\n\n"
        "Context documents will be attached to this conversation by the "
        "Retriever node."
    )
    graph.add_node(tpl); positions[tpl.id] = pos(col=0, row=2)

    conv = ConversationNode()
    graph.add_node(conv); positions[conv.id] = pos(col=1, row=2)

    cli = OllamaClientNode()
    cli.inputs["model"].default_value = "qwen2.5:0.5b"
    graph.add_node(cli); positions[cli.id] = pos(col=2, row=2)

    agent = AgentNode()
    agent.inputs["max_iterations"].default_value = 1   # no tools, just retrieval
    agent.inputs["temperature"].default_value = 0.0
    graph.add_node(agent); positions[agent.id] = pos(col=3, row=2)

    # Connections
    graph.add_connection(loader.id, "documents",  ingest_embedder.id, "documents")
    graph.add_connection(ingest_embedder.id, "embeddings", store.id, "embeddings")
    graph.add_connection(ingest_embedder.id, "documents",  store.id, "documents")
    graph.add_connection(store.id, "store_ref", retriever.id, "store_ref")
    graph.add_connection(query_embedder.id, "embeddings",
                         retriever.id, "query_embedding")
    graph.add_connection(user.id, "message", conv.id, "user")
    graph.add_connection(tpl.id, "text", agent.id, "system_prompt")
    graph.add_connection(conv.id, "conversation", agent.id, "messages")
    graph.add_connection(cli.id, "llm", agent.id, "llm")

    return positions
