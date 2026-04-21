"""Agent Research Assistant — tools + memory + prompting in one graph.

A realistic agent that composes everything the Agents plugin ships:

    DocumentLoader ─► Embedder ─► MemoryStore                (populate once)

    Ollama          ─┐
    PromptTemplate  ─┤
    ChatMessage ─► Conversation ─┤─► Agent ─► response
    Tool: search_notes  (pure)   ─┤
    Tool: calc          (pure)   ─┤
    Tool: save_note     (side)   ─┘

The agent decides when to consult memory (via `search_notes`), when to
compute (via `calc`), and when to persist a finding (`save_note`). The
system prompt comes from a PromptTemplate so the role is editable without
touching the AgentNode.

Defaults chosen so the graph is fully local and runs with zero file setup:
  - DocumentLoader uses *inline* text (a small corpus about the tool itself)
  - Embedder is `hash-64` — zero third-party deps, no torch
  - Qdrant store lives at `./qdrant_data` collection `research_notes`
  - save_note appends to `./research_notes.md` (cwd)

Required for actual execution:
  - Ollama on localhost:11434 with a pulled model (e.g. `qwen2.5:0.5b`)
  - qdrant-client installed (`pip install qdrant-client`)

Run the ingestion pass once (Run Graph), then drive the chat from the
Agents panel — the search_notes tool reads whatever the ingestion pass
wrote to `./qdrant_data/research_notes`.
"""
from __future__ import annotations
from core.graph import Graph
from templates._helpers import grid

LABEL = "Agent Research Assistant (tools + memory + prompt)"
DESCRIPTION = ("A realistic agent composing all three pillars: retrieval "
               "tool over a local Qdrant store, pure calc tool, "
               "side-effect save_note tool, + a templated role prompt.")


_SYSTEM_PROMPT = """\
You are a local-first research assistant. You have three tools:

  - search_notes(query): retrieve relevant passages from the user's
    personal note corpus (vector store). Use this BEFORE answering any
    factual question — the user prefers their own notes over your
    general knowledge.
  - calc(expression): evaluate an arithmetic expression. Always use
    this for any math; never do arithmetic in your head.
  - save_note(text): append a finding to the user's notes file. Use
    ONLY when the user explicitly asks to remember or save something.

Be concise. Cite retrieved passages by their [N] index when used.
"""


_SEED_CORPUS = """\
# Node-tool quick facts

Node-tool is a visual node-based programming environment for ML research.
The tagline is "the graph is the model": a single canvas expresses both
classical ML training pipelines (PyTorch backend) and LLM agent flows.

The default local LLM backend is Ollama on localhost:11434.
The default vector store is Qdrant in local (path) mode.
The default embedder is `hash-64` (zero-dep) or `all-MiniLM-L6-v2`
(sentence-transformers) when that dep is installed.

Training boundaries are defined by A/B markers. The subgraph between an
InputMarkerNode (A) and a TrainMarkerNode (B) sharing the same `group` is
what the Training panel actually trains. The Agent plugin's autoresearch
loop mutates nodes inside that A/B cone.

Any graph can be exported to a standalone Python script plus a pinned
requirements.txt — pure-LLM graphs omit torch entirely.
"""


_TOOL_SEARCH_CODE = '''\
# Retrieve top-k chunks from the local Qdrant store populated by the
# ingestion branch of this template. The path+collection+embedder here
# must match the MemoryStore + Embedder nodes above.
from plugins.agents._memory.store_protocol import MemoryRef, open_store
from plugins.agents._memory.embedder import build_embedder

query = str(kwargs.get("query", "")).strip()
if not query:
    return "error: empty query"

emb = build_embedder("hash-64")
qvec = emb.embed([query])[0]
ref = MemoryRef(backend="qdrant", path="./qdrant_data",
                collection="research_notes")
try:
    store = open_store(ref)
except Exception as exc:
    return f"error: could not open vector store ({exc}). Run the graph once to populate it."
try:
    hits = store.query(qvec, k=3)
finally:
    store.close()
if not hits:
    return "No matching notes found in the corpus."
return "\\n\\n".join(
    f"[{i+1}] {doc.text.strip()[:400]}" for i, (doc, _score) in enumerate(hits)
)
'''


_TOOL_CALC_CODE = '''\
# Arithmetic-only expression evaluator. AST-walk rejects anything that
# isn't a number or a basic op, so this is safe to run unsandboxed.
import ast
import operator as op

OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
       ast.Div: op.truediv, ast.Pow: op.pow, ast.Mod: op.mod,
       ast.FloorDiv: op.floordiv, ast.USub: op.neg, ast.UAdd: op.pos}

def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.operand))
    raise ValueError(f"unsupported expression: {ast.dump(node)}")

expr = str(kwargs.get("expression", "")).strip()
if not expr:
    return "error: empty expression"
try:
    return str(_eval(ast.parse(expr, mode="eval").body))
except Exception as exc:
    return f"error: {exc}"
'''


_TOOL_SAVE_CODE = '''\
# Append a timestamped finding to ./research_notes.md. side_effect=True —
# the agent must have allow_side_effect_tools=True to invoke this.
from pathlib import Path
from datetime import datetime

text = str(kwargs.get("text", "")).strip()
if not text:
    return "error: nothing to save"
fp = Path("./research_notes.md")
stamp = datetime.now().isoformat(timespec="seconds")
with fp.open("a", encoding="utf-8") as f:
    f.write(f"\\n## {stamp}\\n{text}\\n")
return f"Saved to {fp.resolve()}"
'''


def build(graph: Graph) -> dict[str, tuple[int, int]]:
    from nodes.agents.ollama_client         import OllamaClientNode
    from nodes.agents.document_loader       import DocumentLoaderNode
    from nodes.agents.embedder              import EmbedderNode
    from nodes.agents.memory_store          import MemoryStoreNode
    from nodes.agents.chat_message          import ChatMessageNode
    from nodes.agents.conversation          import ConversationNode
    from nodes.agents.prompt_template       import PromptTemplateNode
    from nodes.agents.python_function_tool  import PythonFunctionToolNode
    from nodes.agents.agent                 import AgentNode

    pos = grid(step_x=240, step_y=160)
    positions: dict[str, tuple[int, int]] = {}

    # Row 0 — ingestion (runs once; populates Qdrant at ./qdrant_data)
    loader = DocumentLoaderNode()
    loader.inputs["text"].default_value = _SEED_CORPUS
    loader.inputs["chunk_size"].default_value    = 256
    loader.inputs["chunk_overlap"].default_value = 32
    graph.add_node(loader); positions[loader.id] = pos(col=0, row=0)

    embedder = EmbedderNode()
    embedder.inputs["model"].default_value = "hash-64"
    graph.add_node(embedder); positions[embedder.id] = pos(col=1, row=0)

    store = MemoryStoreNode()
    store.inputs["collection"].default_value = "research_notes"
    graph.add_node(store); positions[store.id] = pos(col=2, row=0)

    # Row 1 — chat inputs
    user = ChatMessageNode()
    user.inputs["role"].default_value    = "user"
    user.inputs["content"].default_value = (
        "Look up what the A/B markers are used for in node-tool, then "
        "tell me how many nodes the autoresearch cone covers if A + "
        "Flatten + 2 Linears + B are the only nodes inside it."
    )
    graph.add_node(user); positions[user.id] = pos(col=0, row=1)

    conv = ConversationNode()
    graph.add_node(conv); positions[conv.id] = pos(col=1, row=1)

    cli = OllamaClientNode()
    cli.inputs["host"].default_value  = "http://localhost:11434"
    cli.inputs["model"].default_value = "qwen2.5:0.5b"
    graph.add_node(cli); positions[cli.id] = pos(col=2, row=1)

    sys_tpl = PromptTemplateNode()
    sys_tpl.inputs["template"].default_value = _SYSTEM_PROMPT
    graph.add_node(sys_tpl); positions[sys_tpl.id] = pos(col=3, row=1)

    # Row 2 — tools
    search_tool = PythonFunctionToolNode()
    search_tool.inputs["name"].default_value = "search_notes"
    search_tool.inputs["description"].default_value = (
        "Retrieve up to 3 passages from the user's personal research notes "
        "by semantic search. Call this before answering factual questions."
    )
    search_tool.inputs["input_schema"].default_value = (
        '{"type":"object","properties":{'
        '"query":{"type":"string","description":"topic / keywords"}'
        '},"required":["query"]}'
    )
    search_tool.inputs["code"].default_value = _TOOL_SEARCH_CODE
    search_tool.inputs["side_effect"].default_value = False
    graph.add_node(search_tool); positions[search_tool.id] = pos(col=0, row=2)

    calc_tool = PythonFunctionToolNode()
    calc_tool.inputs["name"].default_value = "calc"
    calc_tool.inputs["description"].default_value = (
        "Evaluate a pure arithmetic expression. Supports + - * / ** % // "
        "and parentheses. Returns the numeric result as a string."
    )
    calc_tool.inputs["input_schema"].default_value = (
        '{"type":"object","properties":{'
        '"expression":{"type":"string","description":"e.g. (1+2)*3"}'
        '},"required":["expression"]}'
    )
    calc_tool.inputs["code"].default_value = _TOOL_CALC_CODE
    calc_tool.inputs["side_effect"].default_value = False
    graph.add_node(calc_tool); positions[calc_tool.id] = pos(col=1, row=2)

    save_tool = PythonFunctionToolNode()
    save_tool.inputs["name"].default_value = "save_note"
    save_tool.inputs["description"].default_value = (
        "Append a one-line finding to the user's ./research_notes.md file. "
        "Use only when the user explicitly asks to remember or save a fact."
    )
    save_tool.inputs["input_schema"].default_value = (
        '{"type":"object","properties":{'
        '"text":{"type":"string","description":"the finding to save"}'
        '},"required":["text"]}'
    )
    save_tool.inputs["code"].default_value = _TOOL_SAVE_CODE
    save_tool.inputs["side_effect"].default_value = True
    graph.add_node(save_tool); positions[save_tool.id] = pos(col=2, row=2)

    # Row 3 — agent
    agent = AgentNode()
    agent.inputs["max_iterations"].default_value = 5
    agent.inputs["temperature"].default_value    = 0.2
    # save_note is side_effect=True — must opt in.
    agent.inputs["allow_side_effect_tools"].default_value = True
    graph.add_node(agent); positions[agent.id] = pos(col=3, row=2)

    # ── Connections ────────────────────────────────────────────────────
    # Ingestion
    graph.add_connection(loader.id,   "documents",  embedder.id, "documents")
    graph.add_connection(embedder.id, "embeddings", store.id,    "embeddings")
    graph.add_connection(embedder.id, "documents",  store.id,    "documents")

    # Chat
    graph.add_connection(user.id,    "message",      conv.id,  "user")
    graph.add_connection(conv.id,    "conversation", agent.id, "messages")
    graph.add_connection(cli.id,     "llm",          agent.id, "llm")
    graph.add_connection(sys_tpl.id, "text",         agent.id, "system_prompt")

    # Tools
    graph.add_connection(search_tool.id, "tool", agent.id, "tool_1")
    graph.add_connection(calc_tool.id,   "tool", agent.id, "tool_2")
    graph.add_connection(save_tool.id,   "tool", agent.id, "tool_3")

    return positions
