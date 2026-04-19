# `plugins/agents/` — Design Document

**Status:** design only, no code yet. Phase-A implementation begins after Luke signs off on this doc.

**Mission.** Add an agent-builder plugin to node-tool that supports building, running, and locally deploying agents. Open-source-stack-only. Default LLM backend: **Ollama**. Default vector store: **Qdrant** (local mode). Same plugin shape as `plugins/pytorch/`. Aligned with Karpathy's autoresearch as the long-arc target — an agent that mutates the current graph, re-runs it, evaluates, decides the next edit. The graph IS the program the agent edits.

**Hard rules** (carried from research prompt):

1. Local-first. No cloud LLM dependency at runtime; cloud may exist as opt-in nodes only.
2. Mirror `plugins/pytorch/` exactly — same file layout, same `register(ctx)` shape, same `<plugin>_orchestrator.handle_rpc(method, params)` shape, same `PanelSpec` style.
3. Heavy imports deferred to node-execute time. Plugin must `register()` cleanly with neither Ollama running nor sentence-transformers installed.
4. Cross-plugin only via public RPC. The agents plugin never imports from `plugins/pytorch/` or any sibling.
5. Export must produce a standalone Python script — `pip install ... && python graph.py` runs the agent on another machine with no node-tool dependency.

**Decisions (resolved 2026-04-19, Luke).** Folded into the relevant sections below; summarized here for skim:

| # | Decision | Sections affected |
|---|---|---|
| 1 | Tool sandbox: **raw `exec` behind `allow_side_effect_tools` flag**, no isolation in v1. Solo-use posture. | §B.5 |
| 2 | **MCP support promoted to Phase B** (alongside core tool nodes), not deferred to Phase D. | §D, §M |
| 3 | Autoresearch eval loop: **GUI stays responsive** — `EvaluatorNode` polls in background, can `Stop` mid-trial. | §H.3, §H.4 |
| 4 | Empty-state hint when zero Ollama models pulled: **generic message + link to `https://ollama.com/library`**, no specific model named. Picker itself is already JIT from `ollama list`. | §F (section 2) |
| 5 | Mutation region: **reuse the existing `MarkerRole.INPUT` / `MarkerRole.TRAIN_TARGET` (A/B) markers**. The mutable region is the topological subgraph between A and B for the targeted group. No new marker role. | §D, §H.1, §H.5 |
| 6 | Cross-plugin RPC: **build the generic registry** (method-prefix-based) in Phase C, not a hardcoded agents+pytorch fallback chain. | §I, §M (Phase C) |

---

## A. LLM runtime

### A.1 Default backend: Ollama

Ranked decision (full matrix in research notes — abridged here):

| Backend | Win-11 install | Process model | Streaming | Tools | Embeddings | Verdict |
|---|---|---|---|---|---|---|
| **Ollama** | One-click `.exe`; auto-service on `:11434`; bundled CUDA | Out-of-process HTTP | Native SSE; `ollama` Python lib + OpenAI-compat at `/v1` | Native `tools=` | `client.embed()` | **Default** |
| llama-cpp-python | `pip` CPU wheel; CUDA wheels fragile, MSVC build hostile | In-process (native DLL) | OpenAI-shaped chunks | `chat_format="chatml-function-calling"`; JSON-schema constrained | `Llama(embedding=True)` | **Secondary, opt-in** |
| HF Transformers | `pip` + torch (heavy import) | In-process | Manual `TextIteratorStreamer` + thread | None native | None native (use sentence-transformers) | Opt-in for HF-only models |
| vLLM | Linux-only; WSL/Docker required | Out-of-process | OpenAI SSE | Yes | Yes | Out of scope on Windows |

Reasons to make Ollama the default: Luke already runs it on his other machines; out-of-process means `import ollama` is free at register time; OpenAI-compat surface means the same `LLMClient` adapter works against Ollama, LM Studio, llama.cpp's HTTP server, vLLM-in-WSL, and OpenRouter by varying only `base_url`.

The plugin **does not spawn the Ollama server**. It assumes the user has it installed and running. Health-check via `client.ping()`; surface "Ollama is not running" in the inspector when it fails.

### A.2 LLMClient protocol

```python
class LLMClient(Protocol):
    capabilities: ClassVar[set[str]]   # {"tools", "embed", "json_schema", "stream"}
    def chat(self, messages: list[Message], *, tools: list[Tool] | None = None,
             response_format: dict | None = None, **kw) -> ChatResult: ...
    def stream(self, messages: list[Message], *, tools=None, **kw) -> Iterator[StreamChunk]: ...
    async def astream(self, messages: list[Message], *, tools=None, **kw) -> AsyncIterator[StreamChunk]: ...
    def embed(self, texts: list[str], *, model: str | None = None) -> list[list[float]]: ...
    def list_models(self) -> list[ModelInfo]: ...
    def ping(self) -> bool: ...
```

Refinements over the research-prompt's draft:

- Messages, not single prompt — every modern backend is message-oriented.
- `stream` yields incremental deltas (SSE semantics), not accumulated text.
- `astream` is required because the GUI loop is non-blocking; Ollama's `AsyncClient` and `openai.AsyncOpenAI` both support it natively.
- `embed` always takes a batch.
- `capabilities` lets node UIs grey out unsupported toggles without try/except.

Concrete adapters (Phase A): `OllamaClient(host=…)`, `OpenAICompatClient(base_url=…, api_key=…)` — the latter covers LM Studio, llama.cpp's `python -m llama_cpp.server`, vLLM endpoints. Phase-B adds `LlamaCppInProcessClient` (in-process llama-cpp-python).

### A.3 Port types

New types registered in `plugins/agents/port_types.py`:

| Type | Carries | Notes |
|---|---|---|
| `LLM` | A configured `LLMClient` instance | Singleton-per-config; not editable in inspector |
| `MESSAGE` | `{role, content, name?, tool_calls?, tool_call_id?}` | Single chat message |
| `CONVERSATION` | `list[MESSAGE]` | Runs through Agent / Chat nodes |
| `PROMPT_TEMPLATE` | `str` with `{var}` slots | Rendered by `PromptTemplateNode` |
| `TOOL` | `{name, description, input_schema, callable, side_effect: bool}` | Bound to `AgentNode` |
| `DOCUMENT` | `{id, text, metadata: dict}` | Output of loaders, input to embedder/store |
| `EMBEDDING` | `list[float]` with `dim` attribute | Distinct from `TENSOR` so `RetrieverNode` can't accept arbitrary tensors |
| `MEMORY_REF` | `{backend, path, collection}` (string handle, not live object) | Keeps graph serializable |

Convention: `EMBEDDING ↔ TENSOR` conversion via explicit adapter nodes only. `MEMORY` is **not** a port type — store handles are *resource* nodes addressed by name (mirrors the marker pattern from the recent marker refactor).

### A.4 Model file management

Delegated to the backend. Ollama owns its model dir (`%USERPROFILE%\.ollama\models`); HF backends use `~/.cache/huggingface`. The plugin only:

- Lists what's already pulled (`ollama list` → `LLMClientNode` model dropdown).
- Surfaces the install command in a hint when a model is missing (e.g., "Run `ollama pull qwen2.5:0.5b`").

No download UI in v1. Optional `pull_model` RPC in v2 if Luke wants in-app pulls.

### A.5 Streaming → GUI

Mirror the existing `drain_*` pattern from `plugins/pytorch/training_orchestrator.py`. Streaming tokens are NOT carried through the graph as a port value; they go via the orchestrator's event buffer:

```
AgentNode.execute(...)
  └─ orchestrator.start_stream(session_id, messages, tools)
       └─ background asyncio task feeds tokens into orchestrator._pending_tokens[session_id]
GUI panel polls drain_tokens(session_id, poll_ms=100)
  └─ returns {"chunks": [...], "done": bool}
  └─ rendered by a CustomSection (custom_kind="chat_stream")
```

The graph-time output of `AgentNode` is the final `CONVERSATION` (post-stream). Streaming is purely a UI-affordance.

---

## B. Tool protocol

### B.1 What is a tool

A `ToolNode` outputs a `TOOL` value: `{name, description, input_schema, callable, side_effect: bool}`. `input_schema` is JSON Schema (the OpenAI / Ollama / llama.cpp common denominator). `side_effect` flags shell exec / file write / HTTP — the GUI surfaces danger and Phase-D export gates these behind an explicit allowlist.

### B.2 AgentNode binding

`AgentNode` accepts:

- `llm: LLM` (single)
- `system_prompt: PROMPT_TEMPLATE` (optional)
- `tools: TOOL[]` (variadic; the inspector's dynamic-form lists every connected tool with its schema)
- `memory: MEMORY_REF` (optional)
- `messages: CONVERSATION` (the user input)

Output: `response: CONVERSATION` (full transcript including tool calls and their results), plus `final_message: MESSAGE` for convenience.

### B.3 GraphAsTool (composability)

Mirrors `plugins/pytorch/graph_module.py`'s `GraphAsModule` pattern. `GraphAsToolNode` wraps a sub-graph as a callable tool:

- The wrapped sub-graph declares its inputs via `INPUT` markers (reusing the existing marker role) and its output via a new `TOOL_OUTPUT` marker role.
- The sub-graph's exposed inputs become the tool's `input_schema`.
- The wrapper executes the sub-graph in topological order, returns the marked output.
- This is how a user composes "use the existing pytorch training pipeline as a tool the agent can call to retrain a small model with proposed hyperparams."

### B.4 Tool-calling protocol

Wire format: OpenAI-compatible `tools=[{"type":"function","function":{...}}]`. Ollama and llama.cpp both speak this natively; HF/local-only models that don't get a ReAct fallback (Dify-style strategy enum on the AgentNode — `strategy: ENUM(function_calling, react)`).

The agent loop runs **inside** `AgentNode.execute`. Loop budget: `max_iterations: INT` (default 5; Dify's recommendation for "complex" is 10–15 — expose, default low).

### B.5 Security

V1: **deliberately ungated** beyond a single `allow_side_effect_tools: bool` on `AgentNode` (default `False`). Tools with `side_effect=True` raise unless the flag is on. `PythonFunctionToolNode` runs raw `exec` — no subprocess isolation, no `RestrictedPython`. Posture is "solo developer running their own graphs," not "running untrusted templates from strangers."

What this does NOT defend against: a malicious shared template containing a `PythonFunctionToolNode` that deletes files, exfiltrates data, or pulls in malware. If/when Luke starts sharing graphs publicly, revisit with subprocess isolation as the v2 default. Documented at the top of `templates/agent_chat.py` so users importing third-party graphs see the warning.

---

## C. Memory

### C.1 Vector store: Qdrant local mode (default)

`qdrant-client` in local mode — `QdrantClient(path="./qdrant_data")` — runs Qdrant's actual indexing logic in-process, no separate server. The same client object reaches a remote `:6333` if Luke moves to a Qdrant server later, with **zero code change** in the node — only the `MemoryStoreNode`'s `host`/`path` panel field changes. Strongest filter DSL of the four candidates (Filter / must / should / must_not / nested keys / range / geo).

Secondary backends supported behind the same `MemoryStoreNode(backend=…)` selector: `chroma`, `lance`. FAISS not exposed as a node — too low-level.

### C.2 Embedder: sentence-transformers + `all-MiniLM-L6-v2`

Reuses the torch already installed for training. 22.7M params, 384-dim, ~80MB on disk, ~5–14k sentences/sec on CPU. Default model exposed as a panel ENUM with `bge-small-en-v1.5` (also 384-dim) as a one-click upgrade. Fallback: `fastembed` (ONNX-only, no torch) if torch is absent.

After model download, no network calls; `HF_HUB_OFFLINE=1` is set inside the embedder node's process to enforce.

### C.3 Memory model

Two layers, both optional:

- **Short-term** = the `CONVERSATION` running through the AgentNode. Bounded by a `history_window: INT` (messages) on AgentNode itself. No separate node.
- **Long-term** = `MemoryStoreNode` + `EmbedderNode` + `RetrieverNode`. Wired explicitly in the graph; the `RetrieverNode`'s output `DOCUMENT[]` is templated into the prompt by `PromptTemplateNode`. RAG is composition, not a magic flag on AgentNode.

---

## D. Node catalog (v1)

Category in palette: **Agents**. Subcategories below.

| Node | Subcategory | Inputs | Outputs | Behavior |
|---|---|---|---|---|
| `OllamaClientNode` | LLM | panel: `host: STRING`, `model: ENUM` (loaded via `list_local_models` RPC) | `llm: LLM` | Constructs an `OllamaClient`. Defers `import ollama`. |
| `OpenAICompatClientNode` | LLM | panel: `base_url`, `api_key`, `model` | `llm: LLM` | LM Studio / llama.cpp server / vLLM / OpenRouter via OpenAI SDK. |
| `LlamaCppClientNode` | LLM | panel: `model_path`, `n_ctx`, `n_gpu_layers` | `llm: LLM` | In-process via `llama-cpp-python`. Defers heavy import. |
| `PromptTemplateNode` | Prompt | `template: STRING`, `vars: DICT` | `prompt: PROMPT_TEMPLATE`, `text: STRING` | Jinja2 render with `{var}` slots. |
| `ChatMessageNode` | Prompt | `role: ENUM(system,user,assistant)`, `content: STRING` | `message: MESSAGE` | Build a single message. |
| `ConversationNode` | Prompt | `messages: MESSAGE[]` | `conversation: CONVERSATION` | Bundle messages. |
| `AgentNode` | Agent | `llm: LLM`, `system_prompt: PROMPT_TEMPLATE`, `tools: TOOL[]`, `memory: MEMORY_REF`, `messages: CONVERSATION`. Panel: `strategy`, `max_iterations`, `temperature`, `history_window`, `allow_side_effect_tools`. | `response: CONVERSATION`, `final_message: MESSAGE`, `tool_calls: DICT[]` | Runs the function-calling / ReAct loop. Streams via orchestrator. |
| `ToolNode` | Tools | `name: STRING`, `description: STRING`, `input_schema: DICT`, `python_callable: STRING` (dotted path) | `tool: TOOL` | Wraps a Python callable as a tool. |
| `PythonFunctionToolNode` | Tools | `name`, `description`, `code: STRING` (panel multi-line) | `tool: TOOL` | Inline Python; sandboxed exec; `side_effect=True`. |
| `GraphAsToolNode` | Tools | `subgraph_id: STRING` (panel picker) | `tool: TOOL` | Wraps a marked sub-graph as a callable tool. |
| `MCPToolNode` | Tools | `server_url: STRING` | `tools: TOOL[]` | Pulls tools from an MCP (Model Context Protocol) server. **Phase B core.** |
| `DocumentLoaderNode` | Memory | `path: STRING`, `chunk_size: INT`, `chunk_overlap: INT` | `documents: DOCUMENT[]` | Reads txt/md/pdf/html, splits, attaches `{source, chunk_idx}`. |
| `EmbedderNode` | Memory | `documents: DOCUMENT[]` *or* `text: STRING[]`, panel `model: ENUM` | `embeddings: EMBEDDING[]`, `documents: DOCUMENT[]` (passthrough) | Lazy-loads `SentenceTransformer`. |
| `MemoryStoreNode` | Memory | `documents: DOCUMENT[]`, `embeddings: EMBEDDING[]`. Panel: `backend: ENUM(qdrant,chroma,lance)`, `path`, `collection`. | `store_ref: MEMORY_REF` | Idempotent upsert keyed on `id`. |
| `RetrieverNode` | Memory | `query_embedding: EMBEDDING`, `store_ref: MEMORY_REF`, `k: INT`, `where: DICT` | `documents: DOCUMENT[]`, `scores: FLOAT[]` | Backend-normalized filter dict. |
| `ChainNode` | Flow | variadic `steps: AGENT_OR_LLM[]`, `input: CONVERSATION` | `output: CONVERSATION` | Sequential composition. |
| `RouterNode` | Flow | `condition: STRING` (Jinja), branches as variadic outputs | typed branch outputs | Dify-style classifier-router. |
| `IterationNode` | Flow | `items: ANY[]`, `body: SUBGRAPH_REF` | `outputs: ANY[]` | Dify-style for-each; sequential or parallel-bounded. |
| `MutatorNode` | Autoresearch | `target_graph_id: STRING`, `llm: LLM`, `program_md: STRING` | `mutation: DICT` | Asks the LLM to propose ONE typed mutation against the target graph. |
| `EvaluatorNode` | Autoresearch | `target_graph_id: STRING`, `metric: ENUM`, `budget_seconds: INT` | `score: FLOAT`, `status: ENUM(keep,discard,crash)`, `log: STRING` | Calls pytorch RPC to run training, extracts metric, returns. |
| `ExperimentLoopNode` | Autoresearch | `mutator: NODE_REF`, `evaluator: NODE_REF`, `budget: DICT` | `results: DICT[]`, `best_graph_id: STRING` | Drives the mutate→eval→keep/revert loop. |

**Marker roles** added by this plugin (in `core.node.MarkerRole` namespace, referenced from `plugins/agents/`):

- `MarkerRole.AGENT_ENTRY` — graph-level entry point for an agent invocation when exporting a standalone script.
- `MarkerRole.TOOL_OUTPUT` — designates the sub-graph output that `GraphAsToolNode` returns to the caller.

**Mutation region for autoresearch reuses existing pytorch markers** — see §H.1. The autoresearch loop reads `Graph.nodes_by_role(MarkerRole.INPUT)` and `Graph.nodes_by_role(MarkerRole.TRAIN_TARGET)` for the targeted group; the mutable region is the topological subgraph between A (INPUT) and B (TRAIN_TARGET). No new marker role for this — the markers Luke already drops to define a training pipeline ARE the mutation boundary.

---

## E. Orchestrator + RPC surface

### E.1 Class shape

`plugins/agents/agents_orchestrator.py`:

```python
class AgentOrchestrator:
    def __init__(self, graph: Graph): ...
    # ---- LLM enumeration ----
    def list_local_models(self, params) -> dict: ...     # → {"models":[{name,size,modified}, ...]}
    def ping_backend(self, params) -> dict: ...          # → {"ok": bool, "host": str}
    # ---- Conversation lifecycle ----
    def agent_start(self, params) -> dict: ...           # {agent_node_id, message} → {session_id}
    def agent_stop(self, params) -> dict: ...
    def get_agent_state(self, params) -> dict: ...       # {session_id} → {status, tokens_in, tokens_out, latency_ms, model, tool_calls}
    def drain_tokens(self, params) -> dict: ...          # {session_id} → {"chunks":[...], "done": bool}
    def drain_logs(self, params) -> dict: ...            # → {"lines":[...]}
    # ---- Autoresearch ----
    def autoresearch_start(self, params) -> dict: ...    # {graph_id, budget:{trials,wall_clock_s,loss_threshold}}
    def autoresearch_state(self, params) -> dict: ...    # → {trials_done, best_score, current_status, history:[...]}
    def autoresearch_stop(self, params) -> dict: ...
    # ---- RPC entry ----
    def handle_rpc(self, method: str, params: dict | None = None) -> Any: ...
```

`handle_rpc` dispatches via a `_METHODS` dict, raises `ValueError` for unknown methods so `gui/app.py.dispatch_rpc` can fall through to other plugins (same pattern as training).

### E.2 RPC method list

| Method | Params | Returns | Notes |
|---|---|---|---|
| `agent_list_local_models` | `{backend?}` | `{models:[…]}` | Drives `OllamaClientNode` panel dropdown |
| `agent_ping_backend` | `{backend, host?}` | `{ok, host}` | Inspector status field |
| `agent_start` | `{agent_node_id, message, session_id?}` | `{session_id}` | Starts streaming run |
| `agent_stop` | `{session_id}` | `{ok}` | |
| `get_agent_state` | `{session_id}` | `{status, tokens_in, tokens_out, latency_ms, model, n_tool_calls}` | StatusSection poll |
| `agent_drain_tokens` | `{session_id}` | `{chunks:[…], done}` | CustomSection poll @ 100ms |
| `agent_drain_logs` | `{}` | `{lines:[…]}` | Terminal feed |
| `autoresearch_start` | `{graph_id, budget:{trials, wall_clock_s, loss_threshold}, mutator_node_id, evaluator_node_id}` | `{run_id}` | |
| `autoresearch_state` | `{run_id}` | `{trials_done, current, best:{score,graph_hash}, history:[…]}` | StatusSection + PlotSection |
| `autoresearch_stop` | `{run_id}` | `{ok}` | |
| `apply_mutation` | `{graph_id, op:{kind,…}}` | `{ok, new_graph_hash}` | Internal — typed mutation primitives below |
| `revert_to` | `{graph_id, hash}` | `{ok}` | |

Streaming pattern reuses the training orchestrator's `drain_*` shape verbatim — same buffering, same poll cadence in the panel spec.

---

## F. PanelSpec

Panel label: **Agents**. File: `plugins/agents/_panel_agents.py`. Returns a `PanelSpec(label="Agents", sections=[…])` matching the section schema in `core/panel.py`.

Sections (top-to-bottom):

1. **Backend** — `FormSection`: `backend: ENUM(ollama,openai_compat,llama_cpp)`, `host`, `api_key` (hidden unless openai_compat). Status pill: live result of `agent_ping_backend`.
2. **Model picker** — `DynamicFormSection`, `source_rpc="agent_list_local_models"`, `item_label_template="{name} ({size_h})"`, fields: `default: bool`. Picker is JIT — lists whatever `ollama list` returns on the current machine, so the same graph works on any box. Empty-hint (zero models pulled): `"No local Ollama models found. Pull one with `ollama pull <model>` — see https://ollama.com/library for options."` No specific model named; the plugin makes no assumption about target hardware.
3. **Selected agent** — `DynamicFormSection`, `source_rpc="agent_list_agent_nodes"` (returns `AgentNode` instances in graph), per-item fields: `system_prompt`, `temperature`, `max_iterations`, `strategy`, `history_window`, `allow_side_effect_tools`.
4. **Tools** — `DynamicFormSection`, `source_rpc="agent_list_bound_tools"`, per-tool fields: `enabled: bool`, `confirm_before_call: bool`. Side-effect tools rendered with a warning icon (CustomSection wrapper).
5. **Chat** — `CustomSection`, `custom_kind="chat_stream"`, params `{session_rpc:"get_agent_state", drain_rpc:"agent_drain_tokens", input_field_id:"chat_input"}`. Renders a streaming transcript + send-message input box. The renderer is GUI-specific (DPG widget, React component, Godot scene); the spec is the same.
6. **Status** — `StatusSection`, `source_rpc="get_agent_state"`, fields `{model, tokens_in, tokens_out, latency_ms, n_tool_calls}`, poll 1000ms.
7. **Controls** — `ButtonsSection`: `Start` (`agent_start`, collects [Selected agent, Tools, Chat]), `Stop` (`agent_stop`).
8. **Autoresearch** — collapsible sub-panel (CustomSection wrapper around the following):
   - `FormSection`: budget — `trials: INT`, `wall_clock_s: INT` (default 300, mirrors Karpathy's 5-min cap), `loss_threshold: FLOAT?`.
   - `ButtonsSection`: `Start AR` (`autoresearch_start`), `Stop AR` (`autoresearch_stop`).
   - `StatusSection` (`autoresearch_state`, poll 1000ms): `trials_done`, `best_score`, `current_status`, `current_op_kind`.
   - `PlotSection`: source `autoresearch_state`, series `{best_so_far, current_score}`, x = `trial_idx`.

Implementation note: the renderer doesn't need to know what an "agent" is — every section is one of the six existing kinds. The only new piece of GUI work is the `chat_stream` custom renderer, scoped per-frontend (DPG: a child window with auto-scroll text + input; React: a `<ChatStream/>` component; Godot: a VBoxContainer scene).

---

## G. Deployment + export

### G.1 Per-node Jinja partials (agentok pattern)

`plugins/agents/_export/templates/`:

```
main.j2                  # orchestrator: imports + setup + topological execute + entry
ollama_client.j2
openai_compat_client.j2
llama_cpp_client.j2
prompt_template.j2
chat_message.j2
agent.j2
tool_python.j2
tool_graph.j2
embedder.j2
memory_store.j2
retriever.j2
chain.j2
router.j2
import_mapping.j2        # class_type → import statement (deduped)
```

Each node's `export(iv, ov) -> (imports, lines)` reads from these partials. `main.j2` walks `graph.topological_order()`, emits `node_<safe_id>` per node, then a `if __name__ == "__main__": main()` entry point. This mirrors `core/exporter.py`'s existing contract and `dustland/agentok`'s template layout.

### G.2 What an exported script looks like

For a minimal "Ollama + one tool" graph (the v1 demo from §J):

```python
"""Generated by node-tool plugins/agents — 2026-04-19."""
from __future__ import annotations
from ollama import Client

def web_search(query: str) -> str:
    """Stub — replace with real search backend."""
    return f"results for {query}"

TOOLS = [{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web",
        "parameters": {"type":"object","properties":{"query":{"type":"string"}},"required":["query"]},
    },
}]

def main():
    llm = Client(host="http://localhost:11434")
    messages = [
        {"role":"system","content":"You are a concise research assistant."},
        {"role":"user","content":"What's the capital of France?"},
    ]
    for _ in range(5):  # max_iterations
        resp = llm.chat(model="qwen2.5:7b", messages=messages, tools=TOOLS)
        msg = resp["message"]
        messages.append(msg)
        if not msg.get("tool_calls"):
            print(msg["content"])
            return
        for call in msg["tool_calls"]:
            result = {"web_search": web_search}[call["function"]["name"]](**call["function"]["arguments"])
            messages.append({"role":"tool","content":str(result),"tool_call_id":call.get("id","")})

if __name__ == "__main__":
    main()
```

Companion `requirements.txt` emitted alongside, pinned to currently-installed versions of only the libs the graph actually uses (`ollama`, `qdrant-client`, `sentence-transformers`, etc.). No `Dockerfile` in v1 — Luke's deployment target is "another local machine with Python," not "a kube cluster."

### G.3 Dependency minimality

Critical: the agents plugin must NOT drag torch into a pure-LLM exported script. Mechanism:

- Each node's `export()` declares ONLY its own imports.
- `EmbedderNode` is the only thing that pulls `sentence-transformers` (and via it, torch).
- A graph that has `OllamaClientNode → AgentNode` and no embedder → exported `requirements.txt` contains only `ollama`.

This is the inverse of the recently-completed pytorch work (decoupling pure-LLM graphs from torch); the agents plugin must preserve that separation.

### G.4 Talking to the LLM at deploy time

Convention: exported scripts read `OLLAMA_HOST` env var, default `http://localhost:11434`. A short paragraph in the generated script header documents:

- Install Ollama on the target box (link to docs).
- `ollama pull <model>` for each model the graph references (script prints these on first run if missing).
- Optional: run as a Windows service via `nssm install ollama "C:\…\ollama.exe" serve` (stub in `plugins/agents/_export/deploy/ollama_service.md`).

`systemd` unit and Windows-service stub both ship as static markdown — no runtime install logic in v1.

---

## H. Autoresearch integration

### H.1 The mapping

Karpathy's autoresearch mutates `train.py` (text). Node-tool's autoresearch mutates the **subgraph between the existing A/B markers** for the targeted group — `MarkerRole.INPUT` (A) and `MarkerRole.TRAIN_TARGET` (B), the same markers the user already drops to define a training pipeline. The mutator's allowed scope is exactly the topological cone between them. A and B themselves cannot be deleted (they define the boundary); everything in the cone is fair game.

Why reuse rather than introduce a new marker role: zero new UI, zero new mental model — if a graph trains, it can be autoresearched. The pytorch panel's per-group structure (`group` field on markers) gives autoresearch its targeting story for free; `autoresearch_start` takes a `group` param (default: the only group, or first if multiple).

The five primitive mutation kinds, identified from autoresearch's edit categories:

| Karpathy | Node-tool RPC |
|---|---|
| Architecture edit | `apply_mutation({op:"swap_node_class", node_id, new_class_name})`, `apply_mutation({op:"add_node", class_name, connections})`, `apply_mutation({op:"remove_node", node_id})` |
| Optimizer edit | `apply_mutation({op:"swap_node_class", node_id, new_class_name})` (Adam→Muon), `apply_mutation({op:"set_input", node_id, port, value})` |
| Hyperparam edit | `apply_mutation({op:"set_input", node_id, port, value})` |
| Model size edit | combination of `set_input` + `swap_node_class` |
| Code-deletion edit | `apply_mutation({op:"remove_node", node_id})` + `apply_mutation({op:"add_connection", …})` to bypass |

Plus framing primitives: `commit_graph()` (snapshot to hash), `revert_to(hash)`, `run_eval(budget_seconds)`. These already largely exist in `server.py`'s graph-mutation methods — the MutatorNode dispatches to them rather than reaching into `core.graph` directly.

### H.2 How the agent sees the graph

Reuse `serialize_graph` and add `serialize_graph_textual(graph_id, region)` that emits:

```
Node a3f2 (PythonNode) "preprocess"
  inputs: data=<DataLoaderNode b1c4 .out>, scale=2.0
  outputs: result -> [TrainMarkerNode d8e1 .input]
Node b1c4 (DataLoaderNode) "loader"
  inputs: path="./data.csv", batch=32
  outputs: out -> [PythonNode a3f2 .data]
…
```

Compact enough to fit into a 4–8k context for a small graph. The MutatorNode's prompt template includes this textual form, the program.md-equivalent (a `mutation_playbook: STRING` panel field on the node), and the recent results.tsv tail. The LLM is asked to emit a single JSON-typed mutation op.

### H.3 Reading training results back

Cross-plugin RPC, **non-blocking**. The `EvaluatorNode` runs the trial as a yielding poll loop so the GUI stays responsive (panels keep updating, Stop works mid-trial):

```
train_start(params)              # returns immediately with status "running"
loop {
    yield                        # cooperative yield; GUI panels poll their own RPCs
    state = get_training_state()
    if state.status in {"done", "error"}: break
    if wall_clock_elapsed > budget: train_stop(); state.status = "timeout"; break
    if user_pressed_stop: train_stop(); state.status = "aborted"; break
}
losses = get_training_losses()
```

Score = final `val_loss` (or whatever metric the EvaluatorNode is configured for). Status mapping: `done` → `keep` if score < best_so_far else `discard`; `error`/`timeout`/`aborted` → `crash` (record `score=inf`). Mirrors Karpathy's `keep|discard|crash` ledger exactly.

Implementation detail: EvaluatorNode runs in the orchestrator's autoresearch task (an `asyncio.Task` owned by `AgentOrchestrator`), not on the graph-execute thread. The autoresearch panel's poll-driven `autoresearch_state` shows live progress; the user can hit Stop to cancel without freezing the UI.

### H.4 Budget + stopping

Three stop conditions, OR'd, all editable in the panel:

- `trials >= max_trials`
- `wall_clock_elapsed_s >= max_wall_clock_s` (default **300 s** per trial — Karpathy's 5-minute cap)
- `best_score <= loss_threshold`

A history TSV is written to `./.node-tool/autoresearch/<run_id>/results.tsv` with columns `(trial_idx, graph_hash, op_kind, score, status, wall_clock_s, error?)`. Same shape as Karpathy's append-only ledger.

### H.5 What the agent CAN'T do

Hard guardrails enforced by `apply_mutation`:

- Mutations only land on nodes **inside the topological cone between A (`MarkerRole.INPUT`) and B (`MarkerRole.TRAIN_TARGET`) for the targeted group**. Computed by `Graph.subgraph_between(input_node, target_node)`. Anything outside (datasets, panels, other groups, post-train output) is read-only to the mutator.
- A and B markers themselves cannot be removed, swapped, or have their `group` changed — they define the boundary.
- Adding nodes is restricted to a configurable allowlist of node classes (default: all pytorch-plugin layer/optimizer/loss/activation nodes). No tool nodes, no agent nodes, no IO nodes — autoresearch can't bootstrap itself into a different graph.
- `set_input` cannot point at a file path outside the working directory.
- No new pip installs (mirrors Karpathy's "no installing packages" rule).

---

## I. Plugin boundary

The agents plugin touches ONLY:

- `core/node.py` — `BaseNode`, `MarkerRole`, port-type constants.
- `core/panel.py` — `PanelSpec`, sections, fields, actions.
- `core/graph.py` — `Graph`, `nodes_by_role()`, `topological_order()`.
- `core/port_types.py` — `PortTypeRegistry.register()`.
- `core/plugins.py` — `PluginContext`.
- `core/exporter.py` — `GraphExporter` (for export).

Does NOT touch:

- `gui/` — no DPG / Godot / React imports. Layout is `PanelSpec` only. The `chat_stream` custom renderer lives per-frontend (e.g., `gui/dpg/custom_renderers/chat_stream.py`), not in the plugin.
- `nodes/` — agents nodes live under `nodes/agents/`, owned by this plugin.
- `plugins/pytorch/` — never imported. The cross-plugin call (`EvaluatorNode` → training) goes through the public RPC surface (`dispatch_rpc` in-process, or `server.py` JSON-RPC for out-of-process React/Godot frontends).

**Cross-plugin RPC map** (agents → pytorch):

| Caller | RPC method | Params | Used for |
|---|---|---|---|
| `EvaluatorNode` | `train_start` | `{epochs, datasets, optimizer, loss, device}` | Run one training trial |
| `EvaluatorNode` | `get_training_state` | `{}` | Poll until `status in {done, error}` |
| `EvaluatorNode` | `get_training_losses` | `{}` | Extract final val_loss for scoring |
| `EvaluatorNode` | `train_stop` | `{}` | Enforce wall-clock budget |
| `MutatorNode` | `serialize_graph` | `{graph_id}` | Get serializable form for textual rendering |
| `MutatorNode` | `apply_mutation` | `{graph_id, op}` | (own RPC, lives on AgentOrchestrator) |

**Foundation work (Phase C prerequisites):**

1. **Add `Graph.snapshot() -> GraphHash` and `Graph.revert_to(hash)` to core**, expose via `server.py`. Today's `add_node` / `remove_node` / `add_connection` methods imply this surface but don't formalize it; autoresearch needs transactional snapshot → mutate → commit-or-rollback semantics. Manual graph edits in the GUI gain undo/redo for free as a side effect.
2. **Refactor `gui/app.py.dispatch_rpc` to a method-prefix registry.** Today it hardcodes a fall-through chain: `_training_orch.handle_rpc(...)` → fall through to `_robotics_ctrl` on `ValueError`. With agents joining as a third plugin (and more likely later), this becomes a registry: each plugin registers an orchestrator factory keyed on a method prefix (`agent_*`, `train_*`, `robotics_*`). `dispatch_rpc` becomes a one-line lookup. Plugin `register(ctx)` gains `ctx.register_orchestrator(prefix, factory)` to wire it. This is the right shape now, not later — it's a small refactor today and a much larger one once five plugins all hardcode their own fall-through path.

---

## J. File layout

```
plugins/agents/
├── __init__.py                      # register(ctx)
├── DESIGN.md                        # this file
├── port_types.py                    # LLM, MESSAGE, CONVERSATION, PROMPT_TEMPLATE, TOOL,
│                                    # DOCUMENT, EMBEDDING, MEMORY_REF
├── agents_orchestrator.py           # AgentOrchestrator (handle_rpc, drain_*, autoresearch loop)
├── _panel_agents.py                 # build_agents_panel_spec()
├── _llm/
│   ├── __init__.py
│   ├── protocol.py                  # LLMClient Protocol, Message, Tool, ChatResult, StreamChunk
│   ├── ollama_client.py             # defers import ollama
│   ├── openai_compat_client.py      # defers import openai
│   └── llama_cpp_client.py          # defers import llama_cpp
├── _memory/
│   ├── __init__.py
│   ├── store_protocol.py            # VectorStore Protocol
│   ├── qdrant_backend.py            # defers import qdrant_client
│   ├── chroma_backend.py
│   ├── lance_backend.py
│   └── embedder.py                  # defers import sentence_transformers
├── _autoresearch/
│   ├── __init__.py
│   ├── mutator.py                   # MutationOp dataclass + LLM-driven proposer
│   ├── evaluator.py                 # cross-plugin RPC caller for training eval
│   ├── loop.py                      # ExperimentLoop driver
│   └── ledger.py                    # results.tsv writer
├── _export/
│   ├── __init__.py
│   └── templates/
│       ├── main.j2
│       ├── ollama_client.j2
│       ├── openai_compat_client.j2
│       ├── llama_cpp_client.j2
│       ├── prompt_template.j2
│       ├── chat_message.j2
│       ├── agent.j2
│       ├── tool_python.j2
│       ├── tool_graph.j2
│       ├── embedder.j2
│       ├── memory_store.j2
│       ├── retriever.j2
│       ├── chain.j2
│       ├── router.j2
│       └── import_mapping.j2
└── _factories.py                    # build_llm_client(backend, **kw), build_vector_store(...)

nodes/agents/                        # one node class per file (hygiene rule)
├── __init__.py
├── ollama_client.py
├── openai_compat_client.py
├── llama_cpp_client.py
├── prompt_template.py
├── chat_message.py
├── conversation.py
├── agent.py
├── tool.py
├── python_function_tool.py
├── graph_as_tool.py
├── mcp_tool.py                      # phase B/C
├── document_loader.py
├── embedder.py
├── memory_store.py
├── retriever.py
├── chain.py
├── router.py
├── iteration.py
├── mutator.py
├── evaluator.py
└── experiment_loop.py

templates/                           # ships with shipped plugins
├── agent_chat.py                    # demo: Ollama + system prompt + one tool (J.1 below)
├── agent_rag.py                     # demo: docs → embed → store → retrieve → agent
└── agent_autoresearch_mlp.py        # demo: H.X mutate-eval loop on a tiny MLP

tests/
├── test_agents_plugin_register.py   # plugin loads with neither Ollama nor sentence-transformers installed
├── test_agents_nodes.py             # per-node smoke tests (mock LLMClient)
├── test_agents_orchestrator.py      # handle_rpc dispatch + drain_*
├── test_agents_export.py            # round-trip: build graph → export → exec the script in subprocess
└── test_agents_templates_one_step.py # each shipped template runs one full agent turn against MockLLMClient
```

Hygiene compliance:

- `nodes/agents/<name>.py` — one class per file, no helpers. Helpers live in `plugins/agents/_factories.py` and `plugins/agents/_llm/`, `_memory/`, etc.
- `_*` underscore-prefixed packages and modules are NOT auto-discovered as nodes (matches existing convention).
- All heavy imports (`ollama`, `qdrant_client`, `sentence_transformers`, `llama_cpp`, `transformers`) deferred inside methods, never at module top.

---

## K. Minimum viable demo (v1)

`templates/agent_chat.py` — graph the user can build in the GUI:

```
[OllamaClientNode] (model=qwen2.5:7b, host=localhost:11434)
       │ llm
       ▼
[ChatMessageNode] (role=system, content="You are a concise assistant.") ─┐
[ChatMessageNode] (role=user, content="{user_input}")                    │
       │ message                                                         │
       └─────────► [ConversationNode] ──conversation──┐                  │
                                                       ▼                  │
                                                  [AgentNode] ◄───────────┘
                                                       │ system_prompt
                                  [PromptTemplateNode]─┘
                                                       │ tools
       [PythonFunctionToolNode]──tool─────────────────┘
        (name="get_time", code="from datetime import datetime; return datetime.now().isoformat()")
                                                       │ response
                                                       ▼
                                                  (rendered in Chat panel section, streaming)
```

Expected user experience: open the Agents panel, click Start, type "what time is it", watch tokens stream in, see the model issue a `get_time` tool call, see the tool result fed back, see the final response. Total dependencies: `ollama` (Python client + running daemon). Nothing else.

---

## L. Autoresearch demo (v1)

`templates/agent_autoresearch_mlp.py` — a graph where MutatorNode tries 3 variants of a small MLP on a fixed dataset:

```
                 [DataLoaderNode]──out──┐
                                        ▼
[InputMarkerNode A] ──► [LinearNode] ──► [ReLUNode] ──► [LinearNode] ──► [TrainMarkerNode B]
       └─── mutable region = subgraph between A and B (LinearNode/ReLUNode/LinearNode) ───┘

[OllamaClientNode] (model=<whatever you've pulled, e.g. qwen2.5:7b>)
       │ llm
       ▼
[MutatorNode] (target_group=<group_id of A/B above>, playbook="propose ONE change per iteration:
                         - swap activation (ReLU/GELU/Tanh)
                         - change hidden width (powers of 2 between 32–512)
                         - add/remove a Linear+activation block")
       │
       ▼
[EvaluatorNode] (metric=val_loss, budget_seconds=60, calls train_start RPC on pytorch plugin)
       │
       ▼
[ExperimentLoopNode] (budget = {trials:3, wall_clock_s:300, loss_threshold:0.05})
       │
       ▼
   results.tsv  +  best graph hash  +  panel shows running plot
```

Three trials, each ≤60s training, total ≤5min wall-clock. The panel's Autoresearch sub-section shows `trials_done`, `best_score`, and a line plot of `best_so_far` vs `trial_idx`. The graph hash for the best trial is offered as "Apply best" (one-click revert-to-best).

---

## M. Staged build plan

| Phase | Scope | Shippable when |
|---|---|---|
| **A — Core LLM + Agent loop** | `OllamaClient`, `OpenAICompatClient`, port types LLM/MESSAGE/CONVERSATION/PROMPT_TEMPLATE, nodes: OllamaClientNode, OpenAICompatClientNode, PromptTemplateNode, ChatMessageNode, ConversationNode, AgentNode (no tools yet, no streaming yet — just `chat()` end-to-end). PanelSpec: Backend, Model picker, Selected agent, Chat (non-streaming), Status, Controls. AgentOrchestrator with `agent_start/stop/state/drain_logs` (drain_tokens stubbed). Tests: register, node smoke, mock-LLM integration. | User can build a graph that talks to a local Ollama model and see the response in the panel. |
| **B — Tools + Memory + Streaming** | TOOL/DOCUMENT/EMBEDDING/MEMORY_REF port types, all Tool nodes (Tool, PythonFunctionTool, GraphAsTool, **MCPToolNode**), Memory nodes (DocumentLoader, Embedder, MemoryStore, Retriever) with Qdrant default. Streaming via drain_tokens + chat_stream custom renderer (DPG only in this phase). Add LlamaCppClient. PanelSpec: Tools section + streaming chat. Side-effect-tool gate (no sandbox; `allow_side_effect_tools` flag only). | User can build a RAG-augmented agent with custom tools, MCP-server tools, and watch responses stream in. |
| **C — Autoresearch loop** | MutatorNode (reads textual graph from existing A/B markers), EvaluatorNode (yielding poll loop, cross-plugin RPC to pytorch), ExperimentLoopNode, ledger TSV. AgentOrchestrator gains autoresearch_* + an `asyncio.Task`-driven evaluator. Panel gets the Autoresearch sub-section + plot. **Foundation work:** `Graph.snapshot()` / `revert_to()` / `subgraph_between(a, b)` in core; **`dispatch_rpc` method-prefix registry refactor** with `ctx.register_orchestrator(prefix, factory)` API. Ship `templates/agent_autoresearch_mlp.py`. | User can run the demo from §L: 3 trials of a small MLP, GUI stays responsive throughout, picks best automatically. |
| **D — Deployment + export** | Per-node Jinja partials, GraphExporter integration, `requirements.txt` emission, pure-LLM-graphs-don't-require-torch enforcement (test). React/Godot chat_stream renderers. Service-deploy stubs (Windows nssm + systemd unit). | User can `Export Code` on an agent graph and `pip install … && python graph.py` it on another machine. |

Each phase ends with all tests green and at least one shipped template demonstrating the new capability.

---

## N. Decisions log (resolved 2026-04-19)

All six prior open questions resolved with Luke. Folded into the relevant sections; here's the trail:

1. **Tool sandbox** → §B.5. Raw `exec` behind `allow_side_effect_tools` flag; no isolation in v1. Solo-use posture; revisit if shared templates become a thing.
2. **MCP timing** → §D node catalog (`MCPToolNode` row), §M Phase B. MCP is a Phase-B core deliverable, not Phase D.
3. **Eval loop** → §H.3. EvaluatorNode runs as a yielding poll loop on an `asyncio.Task` owned by `AgentOrchestrator`. GUI stays responsive; Stop works mid-trial.
4. **Empty-state model hint** → §F section 2. Generic message + link to `https://ollama.com/library`. No specific model named — picker is JIT from `ollama list` and the same graph runs on any machine.
5. **Mutation region** → §D marker roles, §H.1, §H.5. **Reuse the existing A/B markers** (`MarkerRole.INPUT` / `MarkerRole.TRAIN_TARGET`). The mutable region is the topological cone between them. No new marker role; if a graph trains, it can be autoresearched. New core helper: `Graph.subgraph_between(a, b)`.
6. **Cross-plugin RPC** → §I, §M Phase C. Generic method-prefix registry, not a hardcoded fall-through chain. New plugin-context API: `ctx.register_orchestrator(prefix, factory)`. Done in Phase C alongside `Graph.snapshot()` / `revert_to()`.

---

## O. Out of scope for v1

- Multi-user / auth / sessions-as-a-service.
- Cloud LLM nodes (Anthropic, OpenAI direct). Use OpenAICompatClient against any compat endpoint.
- Agent-to-agent protocols beyond a single graph.
- Observability dashboards beyond panel + terminal.
- Full sandbox for tool execution (Decision 1: ungated in v1; revisit before sharing graphs publicly).
- `Dockerfile` emission on export.

## P. Stretch (mentioned, not designed)

- Record-and-replay harness for agent runs (every LLM call logged with seed; deterministic replay node).
- Self-evaluation loop using a cheap local model as judge (`llama-guard`-style) — wires naturally as another `AgentNode` evaluating the first agent's output.

(MCP tool compatibility was a stretch in v1; promoted to Phase B core per Decision 2.)
