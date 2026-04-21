"""Agent Autoresearch — single-node LLM-guided search over an MLP.

    Data In (A:x) ─► Flatten ─► Linear ─► ReLU ─► Linear ─► Data Out (B)
        │                          ▲                ▲          │
        │ batch_size               out_features     activation │ lr (primary)
        │                          │                │          │
        └──── (control wires from agent) ───────────┴──────────┘

    OllamaClient ─llm─► AutoresearchAgent
                              │
                              └─ control wires define the search space:
                                  • A.batch_size       (INT)
                                  • Linear_1.out_features  (INT, width)
                                  • Linear_1.activation    (STRING choice)
                                  • B.lr               (FLOAT, primary)

Start the run from the Agents panel → Autoresearch → "Start Autoresearch".
The agent walks its own outgoing `control` connections to discover what's
in scope, then mutate→train→keep/revert per trial.

A TSV ledger lands at `./.node-tool/autoresearch/<run_id>/results.tsv`.

## Setup before clicking Start

1. Open the Training panel and click Start once. The training params
   (dataset path, batch size, val_fraction, optimizer, loss, epochs)
   get cached. Autoresearch re-uses them per trial — without this step
   it has no dataset to train on and refuses to start.
2. The Agents panel's Autoresearch section drives the loop.

Required:
  - Ollama on localhost:11434 (with a model pulled, e.g. `llama3.1:8b`)
  - MNIST (or any 28x28 grayscale → 10-class dataset) wired into the
    Training panel for group "mlp"
"""
from __future__ import annotations
from core.graph import Graph
from templates._helpers import grid

LABEL = "Agent Autoresearch (mutate → eval → keep MLP)"
DESCRIPTION = ("LLM-guided architecture search on an MNIST-shape MLP. "
               "Single AutoresearchAgent node; control wires define the "
               "search space (width, activation, learning rate, batch size).")


_PLAYBOOK = """\
Tune the wired parameters to lower val_loss on a small MLP for MNIST.
Try one cluster of changes per trial — small steps work better than big
swings.

Heuristics:
  - out_features: powers of 2 between 32 and 512.
  - activation:  prefer relu / gelu / silu / tanh.
  - lr:          log-uniform between 1e-5 and 1e-2.
  - batch_size:  powers of 2 between 32 and 256.

Avoid configurations that already appear in Recent trials.
"""


def build(graph: Graph) -> dict[str, tuple[int, int]]:
    from nodes.pytorch.input_marker          import InputMarkerNode
    from nodes.pytorch.flatten               import FlattenNode
    from nodes.pytorch.linear                import LinearNode
    from nodes.pytorch.train_marker          import TrainMarkerNode
    from nodes.agents.ollama_client          import OllamaClientNode
    from nodes.agents.autoresearch_agent     import AutoresearchAgentNode

    pos = grid(step_x=220, step_y=170)
    positions: dict[str, tuple[int, int]] = {}

    # Row 0 — the trainable pipeline
    data_in = InputMarkerNode()
    data_in.inputs["modality"].default_value = "x"
    data_in.inputs["group"].default_value    = "mlp"
    data_in.inputs["batch_size"].default_value = 128
    graph.add_node(data_in); positions[data_in.id] = pos(col=0, row=0)

    flat = FlattenNode()
    graph.add_node(flat); positions[flat.id] = pos(col=1, row=0)

    h1 = LinearNode()
    h1.inputs["out_features"].default_value = 128
    h1.inputs["activation"].default_value   = "relu"
    graph.add_node(h1); positions[h1.id] = pos(col=2, row=0)

    h2 = LinearNode()
    h2.inputs["out_features"].default_value = 10     # 10 MNIST classes
    h2.inputs["activation"].default_value   = "none"
    graph.add_node(h2); positions[h2.id] = pos(col=3, row=0)

    train_out = TrainMarkerNode()
    train_out.inputs["group"].default_value  = "mlp"
    train_out.inputs["kind"].default_value   = "logits"
    train_out.inputs["target"].default_value = "label"
    train_out.inputs["lr"].default_value     = 0.001
    train_out.inputs["optimizer"].default_value = "adam"
    train_out.inputs["loss"].default_value   = "crossentropy"
    train_out.inputs["epochs"].default_value = 3
    graph.add_node(train_out); positions[train_out.id] = pos(col=4, row=0)

    graph.add_connection(data_in.id,   "tensor",     flat.id,      "tensor_in")
    graph.add_connection(flat.id,      "tensor_out", h1.id,        "tensor_in")
    graph.add_connection(h1.id,        "tensor_out", h2.id,        "tensor_in")
    graph.add_connection(h2.id,        "tensor_out", train_out.id, "tensor_in")

    # Row 1 — the autoresearch driver
    cli = OllamaClientNode()
    cli.inputs["host"].default_value  = "http://localhost:11434"
    cli.inputs["model"].default_value = "llama3.1:8b"
    graph.add_node(cli); positions[cli.id] = pos(col=0, row=1)

    agent = AutoresearchAgentNode()
    agent.inputs["group"].default_value         = "mlp"
    agent.inputs["playbook"].default_value      = _PLAYBOOK
    agent.inputs["metric"].default_value        = "val_loss"
    agent.inputs["trials"].default_value        = 8
    agent.inputs["wall_clock_s"].default_value  = 900.0
    agent.inputs["eval_budget_s"].default_value = 60.0
    agent.inputs["temperature"].default_value   = 0.4
    graph.add_node(agent); positions[agent.id] = pos(col=2, row=1)

    # LLM into the agent
    graph.add_connection(cli.id, "llm", agent.id, "llm")

    # Control wires — these define the search space. Anything not wired
    # here is OFF-LIMITS to the agent. The canvas reads at a glance.
    # Width is safe to tune because LinearNode now infers its in_features
    # from the upstream tensor on each forward, so changing h1's
    # `out_features` automatically reshapes h2 on its next forward.
    graph.add_connection(agent.id, "control", h1.id,        "out_features")
    graph.add_connection(agent.id, "control", h1.id,        "activation")
    graph.add_connection(agent.id, "control", h1.id,        "freeze")
    graph.add_connection(agent.id, "control", train_out.id, "lr")
    graph.add_connection(agent.id, "control", data_in.id,   "batch_size")

    return positions
