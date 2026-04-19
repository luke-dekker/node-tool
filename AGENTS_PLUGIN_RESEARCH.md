# Agents Plugin — Research Prompt

**To Claude (future session):** read this end-to-end before starting. It is
self-contained; you should not need the conversation that produced it.

## Mission

Produce a complete design document for `plugins/agents/` — a node-tool
plugin for **building, running, and locally deploying agents**. The plugin
must be open-source-stack-only (no cloud services required at runtime or
authoring time) and must follow the same architectural patterns the pytorch
plugin already establishes in this repo.

Output: one design document (`plugins/agents/DESIGN.md`) covering the
sections listed under "Deliverables" below. Do not write code in this
phase — the output is the blueprint implementation will follow.

## Node-tool context (don't re-derive, use as-is)

* **Plugin architecture** — `plugins/<name>/__init__.py` defines `register(ctx)`
  which calls `ctx.register_port_type`, `ctx.discover_nodes`, and
  `ctx.register_panel_spec`. See `plugins/pytorch/` for the reference
  implementation.
* **Graph model** — `core.graph.Graph` holds nodes + connections; nodes are
  `core.node.BaseNode` subclasses with typed input/output ports. Execution is
  topological. "The graph IS the model" philosophy extends to agents: the
  graph IS the agent / agent pipeline.
* **Panel contract** — `core.panel.PanelSpec` describes a plugin's side
  panel once; every GUI (DPG, React, Godot) renders the same spec through
  its own generic renderer. Section kinds: `form`, `dynamic_form`,
  `status`, `plot`, `buttons`, `custom`. See
  `plugins/pytorch/_panel_training.py` for the reference.
* **Orchestrator pattern** — long-lived, stateful domain logic lives in a
  `plugins/<name>/<name>_orchestrator.py` class exposing `handle_rpc(method,
  params)`. Both `server.py` (for React / Godot via JSON-RPC) and
  `gui/app.py.dispatch_rpc` (for DPG in-process) route through it. See
  `plugins/pytorch/training_orchestrator.py`.
* **Marker roles** — `BaseNode.marker_role` + `Graph.nodes_by_role()` lets
  plugins flag "special" nodes without hardcoding type_name strings. Used by
  training (`INPUT`, `TRAIN_TARGET`); agents will likely need new roles.
* **Port types live with their plugin** — `plugins/pytorch/port_types.py`
  registers `TENSOR`, `DATALOADER`, etc. at `register()` time. The agents
  plugin must do the same for its types.

## Hard constraints

1. **Open-source, local-first.** Every primitive must work with an entirely
   local stack. No OpenAI / Anthropic / cloud-LLM dependency at runtime.
   Cloud providers may be supported as optional nodes, but the plugin must
   pass its smoke tests with only local tooling installed.
2. **Alignment with the pytorch plugin.** Same file layout, same naming
   patterns, same orchestrator / PanelSpec / handle_rpc shape. A developer
   who can read `plugins/pytorch/` must recognize `plugins/agents/`.
3. **Alignment with Karpathy's autoresearch** (`github.com/karpathy/autoresearch`).
   Autoresearch is a minimal three-file workflow where an agent iteratively
   modifies a `train.py`, runs a short experiment, evaluates on validation
   loss. The node-tool equivalent is: an agent node iteratively **mutates
   the current graph**, re-executes it (reusing existing training
   infrastructure), evaluates the result, and decides the next edit. The
   graph IS the program the agent is editing.
4. **Reuse existing libs.** The repo already depends on `torch`, `numpy`,
   `pandas`, `sklearn`, `scipy`, `lerobot`. Don't introduce new heavy deps
   where an existing one suffices. For LLM / embedding / vector stores,
   prefer libraries that are widely used, pure Python where possible, and
   don't require a GPU for dev-time smoke testing.
5. **Deployment target = another local machine.** "Running local agents on
   another machine" means a graph exported from node-tool can be run
   headlessly with `pip install ... && python graph.py` — no node-tool
   server required at runtime. Export must produce a standalone artifact.

## Reference projects (scan each before starting)

* **agno-agi/agno** — code-first Python agent framework with
  teams/memory/tools/knowledge abstractions. Study their agent-team
  composition model and session isolation; it's the closest thing to a
  mature taxonomy of primitives.
* **karpathy/autoresearch** — three-file self-modifying ML research loop.
  This is the alignment target for agent-driven graph mutation.
* **FlowiseAI/Flowise** — mature node-based agent builder (TypeScript). Study
  their node catalog (LLM / Prompt / Memory / Chain / Tool / Agent /
  VectorStore / Retriever) and graph JSON schema. We want feature parity on
  nodes, not on the runtime.
* **simstudioai/sim** — another visual builder; explicit Ollama + vLLM
  Docker support. Study their local-LLM story and how they handle the
  difference between "node definition" and "running block."
* **dustland/agentok** — visual→Python codegen for Microsoft AutoGen (ag2).
  Study the codegen path; our "export standalone Python" deliverable is
  closely related.
* **langgenius/dify** — workflow orchestration canvas with RAG + agents +
  monitoring. Study their workflow execution model and input/output typing.
  Flag anything that requires their cloud before using it.

## Key design questions to answer

Address every one of these in the design doc. For each, argue the decision,
name the tradeoff, and cite the reference project(s) that informed it.

### A. LLM runtime

1. What is the **default local LLM backend**? Candidates: Ollama (HTTP),
   llama.cpp (via `llama-cpp-python`), vLLM (HTTP), Hugging Face
   Transformers (direct torch). Rank them by ease-of-install, Windows
   support, streaming quality, model range, and GPU/CPU footprint.
2. What is the **abstraction**? A `LLMClient` protocol with `complete`,
   `stream`, `embed` methods; concrete implementations per backend.
3. What is the **port type**? Probably `LLM` (a handle to a configured
   client) + `MESSAGE` / `CONVERSATION` for typed I/O.
4. How are **model files** managed? Does the plugin download them, or rely
   on the backend (Ollama pull, HF cache) to manage?
5. **Streaming** — the existing port/execute model is value-in, value-out.
   How do streaming tokens reach the GUI? Proposal: a new event stream via
   the orchestrator's `drain_*` pattern, routed to a `log_tail`-like custom
   section.

### B. Tool protocol

1. What is a **tool** in node-tool terms? Proposal: a `ToolNode` exposes a
   `TOOL` output that bundles `{name, description, input_schema, callable}`.
   An `AgentNode` accepts a list of `TOOL` inputs.
2. Must tools be **pure functions** or can they have side effects? Both —
   classify them so the GUI can surface danger (shell exec, file write).
3. **Can any other node become a tool?** Proposal: a generic
   `GraphAsTool` node wraps a sub-graph as a callable tool — same pattern
   as GraphAsModule for training. Crucial for composability.
4. **Tool calling protocol** — OpenAI-compatible function calling? JSON
   mode? Native local-model support varies; decide the contract.
5. **Security** — sandboxing shell exec / file IO tools. Out of scope for
   v1? Design doc must at minimum note the risk.

### C. Memory

1. Short-term (conversation history) vs long-term (RAG / vector store).
2. Local vector store: `chromadb`, `lancedb`, `qdrant` (local mode),
   `faiss` — pick one with rationale. Pure-python preferred.
3. Embedding model: `sentence-transformers` via HF Transformers is the
   obvious local default. Verify no cloud calls.
4. Port types: `MEMORY`, `DOCUMENT`, `EMBEDDING`? Or reuse existing TENSOR
   for embeddings?

### D. Node catalog (v1)

Enumerate the minimum viable node set. Include category + subcategory
(palette placement), input/output ports with types, and a one-sentence
description. Must at minimum cover:

* LLMClientNode (per backend), PromptTemplateNode, ChatMessageNode
* AgentNode (LLM + tools + memory), ToolNode, GraphAsToolNode
* MemoryStoreNode, EmbedderNode, RetrieverNode, DocumentLoaderNode
* ChainNode (sequential), ParallelNode, RouterNode (branching)
* Autoresearcher-style: MutatorNode (proposes graph edits), EvaluatorNode
  (runs the graph, returns a score), ExperimentLoopNode (Mutator →
  Evaluator → Mutator, until stopping criterion)

### E. Orchestrator + RPC surface

Design `AgentOrchestrator.handle_rpc(method, params)`. At minimum:

* `agent_start(agent_id, message)` → starts a conversation
* `agent_stream(session_id)` → returns queued tokens (drain pattern)
* `agent_stop(session_id)`
* `get_agent_state(session_id)` → {status, tokens_in, tokens_out, cost, ...}
* `list_local_models()` → enumerate Ollama / HF cache
* `autoresearch_start(graph_id, budget)` → kick off the mutation loop
* `autoresearch_state(run_id)` → progress + best-so-far
* `autoresearch_stop(run_id)`

### F. PanelSpec

Design the Agents panel following the training panel's shape. Sections at
minimum: Model picker (DynamicForm or form with local-models enumerated),
Prompt (form), Tools (dynamic_form over selected tool nodes), Chat (custom
section — streaming transcript), Controls (Start/Stop), Status, and an
Autoresearch sub-section with experiment progress.

### G. Deployment + export

The existing `export_code()` RPC produces a Python script from an ML graph.
Extend the pattern for agent graphs:

1. What does a standalone exported agent script look like? (Entry point,
   imports, how it initializes the LLM client, how tools are wired.)
2. How do we package an agent for another machine — requirements.txt,
   `Dockerfile`, or just the .py? Argue based on the open-source-local
   constraint.
3. Can the exported script run **without torch** if the agent doesn't use
   any torch nodes? (This is the reverse of what the pytorch plugin just
   achieved — the agents plugin must not drag torch into a pure-LLM
   graph.) Verify dependency minimality.
4. How does a deployed agent talk to its LLM? Assume Ollama is running on
   the target machine; what's the hostname / port convention? Provide a
   concrete `systemd` or Windows-service stub in the doc.

### H. Autoresearch integration specifics

1. Karpathy's autoresearch mutates a `train.py` text file. Node-tool's
   equivalent: mutate the Graph object (add / remove nodes + connections).
2. What mutation operations are primitive? (ADD_NODE, REMOVE_NODE,
   ADD_CONNECTION, REMOVE_CONNECTION, SET_INPUT.) These map cleanly onto
   `server.py` RPC methods that already exist — the MutatorNode drives
   them through a dispatcher, not by hand-editing the graph.
3. How does the agent **see** the graph? Proposal: a serializer that emits
   a human-readable textual description (node types, connections, current
   input values) short enough to fit an LLM context window. Reuse
   `serialize_graph` where possible.
4. How is **training result** fed back? The pytorch orchestrator's
   `get_training_state` / `get_training_losses` already expose this. The
   MutatorNode reads those to score a generation and pick the next edit.
5. What's the **budget** and **stopping criterion**? Wall-clock, trials,
   loss threshold. Mirror autoresearch's 5-minute experiment cap.

### I. Where the plugin boundary sits

Confirm the plugin does NOT reach into `core/`, `gui/`, or `nodes/`
directly. If the agents plugin needs something from the pytorch plugin
(e.g., "run training and read loss"), it goes through the public RPC
surface of the training orchestrator — not via import. Document the
cross-plugin-dependency story. Flag if any weakness in the current core
prevents clean plugin-to-plugin contracts.

## Deliverables

`plugins/agents/DESIGN.md` with sections matching A–I above, plus:

* **File layout** — exact file tree for `plugins/agents/`.
* **Minimum viable demo** — a concrete graph a user could build in the GUI
  (using the v1 node catalog) to, say, run a local Llama-3 model through a
  prompt with one tool call and see the streaming response.
* **Autoresearch demo** — a concrete graph where the MutatorNode tries 3
  variants of a small MLP on a fixed dataset, picks the best.
* **Cross-plugin RPC map** — which pytorch orchestrator methods the agents
  orchestrator will call for the autoresearch loop.
* **Staged build plan** — phase A (core LLM / prompt / agent), phase B
  (memory + tools), phase C (autoresearch loop), phase D (deployment
  export). Each phase = shippable milestone.
* **Open questions** — list anything you could not resolve with research
  alone and flag for Luke's decision.

## Out of scope for v1

* Multi-user / auth / sessions as a service.
* Cloud LLM integration (may be a v2 optional node).
* Agent-to-agent communication protocols beyond the single-graph case.
* Observability dashboards beyond the existing terminal + status section.

## Stretch ideas to mention (not design)

* MCP (Model Context Protocol) tool compatibility for standing on top of
  Anthropic's emerging tool ecosystem without the cloud.
* A "record-and-replay" harness for debugging flaky agent runs — every LLM
  call logged, deterministic playback.
* Agent self-evaluation loops using a cheap local model as the judge
  (`llama-guard`-style).

## Pointers for the research

* Verify Windows support for each backend — Luke is on Windows 11.
* Test the "plugin loads without deps" path: every heavy import must be
  deferred to node instantiation / execution time, not module-top-level,
  so the agents plugin can register without Ollama / HF running.
* Follow the existing memory on `project_architecture_notes.md` —
  specifically "node files contain nodes, nothing else" and per-template
  smoke tests. Design the agents plugin to pass those constraints.
