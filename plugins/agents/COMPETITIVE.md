# node-tool agents plugin — Product-level comparison

A feature matrix is easy and not very useful. The questions that actually
matter when deciding what to build next are:

  1. **What is the product?** Is node-tool's agent plugin even the same
     *kind of thing* as Flowise, Dify, Langflow, Sim, or n8n?
  2. **What are the nodes?** What atoms do you get to compose with?
  3. **What is the flow?** What is actually passed between nodes — and
     how finely can a user shape that flow into the vessel they want?

This document walks those three axes concretely, then names the
product-design decision that falls out.

---

## 1. What is the product?

The single sharpest axis: **is the graph the program, or is the graph a
config for an engine?**

| System | What the graph *is* |
|---|---|
| **node-tool** | A literal program. Each node has its own `execute()`; connections pass live Python objects. There is no engine between nodes — the graph IS the computation. |
| **Langflow** | A visual DSL that compiles into LangChain / LangGraph Python code run by the LangChain runtime. |
| **Flowise AgentFlow V2** | A DAG of native nodes compiled to Flowise's TypeScript execution engine with `$flow.state` variable-passing. |
| **Dify** | A workflow config for Dify's hosted / self-hosted runtime — threaded through a shared variable pool, durable between runs. |
| **Sim** | A workflow config for Sim's Postgres+realtime runtime; deploys as API/schedule/webhook triggers. |
| **n8n** | A workflow config for n8n's Node.js engine; AI Agent is one cluster node among 400+ connector nodes. |

This matters because it tells you whether you and the competitor are
solving the same problem:

- **node-tool** is a **node-based Python IDE** that happens to ship an
  agents plugin. You can drop into Python from the canvas at any point
  (`PythonNode`, `PythonFunctionToolNode`). The same canvas also trains
  PyTorch models, runs sklearn pipelines, ingests CSVs — agents are one
  plugin among eight. The user is fundamentally *programming*.

- **Flowise / Dify / Sim / n8n** are **LLM-app builders**. The canvas
  IS the product. You compose over pre-built primitives; you don't
  usually compose primitives yourself. The user is fundamentally
  *configuring*.

- **Langflow** sits in the middle. It's pip-installable and its export
  is Python, but the code it emits is LangChain glue — you're shaping
  a LangChain chain, not writing primitives.

**So:** node-tool's agents plugin is not competing with Dify. It's
competing with "LangChain + VSCode." The nearer neighbors in spirit
are Langflow (pip-installable Python-first) and, a step further, the
ComfyUI / Blender Geometry Nodes pattern of "graph = program."

---

## 2. What are the nodes?

Atoms of agent composition, roughly.

### node-tool agents plugin (18 nodes; 7 agent-specific port types)

| Category | Nodes | What they emit |
|---|---|---|
| LLM clients | `ag_ollama_client`, `ag_openai_compat_client`, `ag_llama_cpp_client` | `LLM` (a live client object with `.chat()`/`.stream()`) |
| Messaging | `ag_chat_message`, `ag_conversation`, `ag_prompt_template` | `MESSAGE`, `CONVERSATION` (list[Message]), `STRING` |
| Agent | `ag_agent` | `CONVERSATION` + final `MESSAGE` + `STRING` (text) + tool-call log |
| Tools | `ag_tool` (dotted path), `ag_python_function_tool` (inline exec), `ag_mcp_tool` (stdio + http), `ag_graph_as_tool` (subgraph callable) | `TOOL` (ToolDef dataclass: name, description, JSON schema, Python callable, side_effect flag) |
| Memory | `ag_document_loader`, `ag_embedder`, `ag_memory_store`, `ag_retriever` | `DOCUMENT[]`, `EMBEDDING[]`, `MEMORY_REF`, (doc, score) pairs |
| Autoresearch | `ag_mutator`, `ag_evaluator`, `ag_experiment_loop` | `ANY` dicts consumed by the orchestrator's bg loop |

Port types: `LLM`, `MESSAGE`, `CONVERSATION`, `PROMPT_TEMPLATE`, `TOOL`,
`DOCUMENT`, `EMBEDDING`, `MEMORY_REF` — 7 agent-specific types, each with
its own color / pin shape, over and above core types (`STRING`, `INT`,
`FLOAT`, `BOOL`, `TENSOR`, `ANY`).

And the canvas also exposes 260+ non-agent nodes from other plugins
(pytorch, numpy, pandas, sklearn, io, robotics). Any of them can sit on
the same canvas as an agent.

### Dify (~14 node types)

Start, End / Answer, LLM, Knowledge Retrieval, Question Classifier,
IF/ELSE, Iteration, Code, Template, HTTP Request, Parameter Extractor,
Variable Aggregator, Tool, Direct Reply.

### Flowise AgentFlow V2 (~10 native node types)

Start, Agent, LLM, Condition, Loop, Iteration, Custom Function, Direct
Reply, HTTP Request, plus Tool nodes. ("Agent as Tool" for sub-agent
composition.)

### Langflow (hundreds, 1:1 with LangChain)

Model, Agent, Tool, Chain, Memory, VectorStore, Retriever, Prompt —
one node per LangChain class, so the node count is essentially
"however many LangChain classes there are." Most flows boil down to
wiring a small number of these slots.

### n8n AI Agent (1 cluster + 4 sub-socket types + 400+ connectors)

`AI Agent` root node with sub-sockets for Chat Model, Memory, Output
Parser, and Tool. Any of n8n's 400+ integration nodes can plug into the
Tool socket.

### Grain of composition

The interesting observation is *where each system draws the line between
one node and many*.

- In node-tool a "chat request" is **five nodes**: OllamaClient +
  ChatMessage + Conversation + PromptTemplate + Agent. The user wires
  the conversation together port-by-port. A message is a first-class
  wire on the canvas; so is a tool; so is the LLM backend.
- In Dify / Flowise / n8n a chat request is **one or two nodes** (LLM
  node, or AI Agent with attached sub-sockets). Messages and
  conversations are implicit inside the node's state.
- In Langflow it's somewhere in between (Prompt + ChatModel + Memory +
  Agent as separate nodes, LangChain-shaped).

**node-tool is the most granular on the data side**: more wires, more
type discipline, but more to assemble. The other systems hide more
inside a single Agent node and give you knobs on that node instead of
upstream composition.

---

## 3. What is the flow?

What is actually carried on an edge? Where does mid-run state live?

### node-tool

- **Edges carry live Python objects.** An edge from `OllamaClientNode`
  to `AgentNode.llm` carries a live `OllamaClient` instance. An edge
  from `PythonFunctionToolNode` to `AgentNode.tool_1` carries a live
  `ToolDef` with a compiled Python callable attached. An edge from
  `DocumentLoaderNode` to `EmbedderNode` carries a `list[Document]`.
- **Typed ports with runtime coercion.** Every port declares a type;
  `PortTypeRegistry.coerce_value` runs at connection time. An `EMBEDDING`
  can't go into a `TENSOR` input silently.
- **DAG, single pass per `graph.execute()`.** Topological order, each
  node's `execute(inputs: dict) → dict` runs exactly once per graph
  execution. No cycles, no back-edges, no re-entry.
- **Control flow lives inside a node, not on the canvas.** AgentNode
  runs the function-calling loop internally up to `max_iterations`.
  Autoresearch's `ExperimentLoop` runs on a background thread owned by
  the orchestrator. The graph sees one invocation and one set of
  outputs in both cases.
- **No IF/ELSE. No Iteration. No Switch. No Merge.** Branching means
  either swapping the whole graph (`load_template`), or dropping into
  Python inside a `PythonNode` / `PythonFunctionToolNode`.

### Dify

- **Edges carry references to named variables in a shared pool.** Each
  node declares inputs (`source_variable`) and outputs (`name, type`).
  Variables are JSON-serializable — strings, numbers, arrays, file
  handles. No live-object passing.
- **Graph-level control flow.** `IF/ELSE` has two output paths;
  `Iteration` re-enters its body N times; `Parameter Extractor` runs
  an LLM to pick a branch; `Variable Aggregator` merges parallel
  branches back into one.
- **State is durable.** Runs live in the DB; variables persist for
  inspection in the run-log UI.

### Flowise AgentFlow V2

- **Dependency/queue execution model** with `$flow.state` — a shared
  object any node can read/write.
- **Condition, Loop, Iteration** are first-class nodes.
- Closer to a workflow engine than a dataflow graph.

### Langflow

- **Edges carry LangChain objects at runtime** — Chain, Agent,
  VectorStore, Tool instances. Similar spirit to node-tool's live-object
  passing, but the vocabulary is LangChain's.
- **LangGraph** is the cycle/branching escape hatch.

### n8n

- **Edges carry arrays of JSON items.** Every node receives `items: list`
  and returns `items: list`. "Flow" is literally items moving through
  nodes.
- **IF / Switch / Merge / Loop Over Items / Split In Batches** are
  first-class nodes — control flow is the whole point of n8n.

### The granularity picture

Two orthogonal axes:

```
                          Control-flow granularity
                          (IF/LOOP/SWITCH on canvas)
                                   ▲
                                   │
                 n8n ●             │            ● Dify
                                   │
            Flowise V2 ●           │
                                   │
                                   │            ● Langflow (via LangGraph)
                                   │
     ──────────────────────────────┼──────────────────────────────►
                                   │            Data-flow granularity
                                   │            (typed live objects between nodes)
                                   │
                                   │            ● node-tool
                                   │
                                   ▼
```

- Low-right (node-tool): typed live objects per wire; loops & branches
  are hidden inside nodes or live in Python.
- Upper-left (n8n): JSON items between nodes; loops & branches are the
  whole vocabulary.
- Upper-right (Dify): named JSON variables + rich graph-level control.

Neither axis is "better." They answer different questions about what
the user is *shaping*.

---

## 4. The product-design question that falls out

The user who picks Dify wants to **shape the control flow** — what
happens when retrieval returns nothing, when an LLM confidence is low,
when two branches finish at different times. The canvas is where
decisions live.

The user who picks node-tool (today) wants to **shape the data** —
what a Message actually is, what a Tool is, what's in the store, how
the pytorch subgraph hooks into the LLM. The canvas is where types and
values live.

Where node-tool hits a wall is exactly where you said — *the user
wants to "guide the flow into the desired vessel" and doesn't have the
granularity at the canvas level.* A concrete example every LLM-app
developer runs into:

> "If `search_notes` returns nothing, retry with a broader query.
> Otherwise, pass results into the agent. If the agent's response
> mentions a number, route through `calc`. Otherwise finalize."

In node-tool today that's **all inside one Python function tool**, or
split across pre-scripted Python in multiple tools. None of it shows
up as wires on the canvas. In Dify it's literally IF/ELSE + Iteration
+ Parameter Extractor nodes, visible on the canvas, logged per-run.

**The real decision:** which direction does node-tool go?

1. **Add graph-level control flow.** A small set of control-flow
   primitives — `IF`, `Switch`, `ForEach`, `Retry`, `Merge` —
   operating on the typed-port model we already have. This catches up
   to the "shape the flow" axis without abandoning the typed live-object
   model. Biggest design risk: IF/ELSE on a DAG requires either running
   both branches and selecting (wasteful) or delaying execution
   (breaking the single-pass semantic that makes training graphs work).

2. **Double down on "drop to Python."** Keep the canvas as a dataflow
   graph. Make `PythonFunctionToolNode` and `PythonNode` the obvious
   escape hatch for anything branchy. Invest in editor-level ergonomics
   (decent code editor in the node panel, inline errors, debug into
   the tool code). Accept that complex control flow lives in Python,
   not in the graph.

3. **Hybrid.** Ship `IF` + `ForEach` + `Retry` as the minimum viable
   control-flow set — they cover 80% of what users actually reach for —
   and keep Python as the escape hatch for the rest. Avoid the full
   Dify/n8n control vocabulary (Switch with N branches, Merge,
   Aggregator) unless a user hits a wall.

Options 1 and 3 have a ripple effect on `graph.execute()` — it becomes
a small interpreter over the DAG rather than a single topological
pass. That's the work.

Option 2 is cheaper but locks in the positioning: "node-tool is for
data-shaping; if you need flow control, write Python." Fine as a
product statement, but it means the flow-shaping users go to Dify.

The honest read: option 3 is probably where this ends up, because
autoresearch *already* needs control flow (mutate → eval → keep/revert
is literally IF-EVAL-BETTER). Right now that logic lives inside
`ExperimentLoop` on a background thread, invisible to the canvas. If
a graph-level `IF` existed, autoresearch could BE a small subgraph
rather than an orchestrator-driven bg thread — the same primitive
would serve both cases.
