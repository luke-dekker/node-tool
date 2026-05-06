"""Agent Autoresearch — single-node LLM-guided search over a 3-hidden MLP.

    Data In (A:x) ─► Flatten ─► Linear ─► Linear ─► Linear ─► Linear ─► B
                                  h1       h2       h3       h_out (fixed 10)
                                  ▲▲       ▲▲       ▲▲
                                  ││       ││       ││    (control wires)
                                  └┴───────┴┴───────┴┴──── out_features + activation

    OllamaClient ─llm─► AutoresearchAgent
                              │
                              └─ control wires (search space):
                                  • A.batch_size            (INT)
                                  • h1/h2/h3.out_features   (INT, width)
                                  • h1/h2/h3.activation     (STRING choice)
                                  • B.lr                    (FLOAT, primary)

    Output head h_out is intentionally NOT wired — class count must stay
    at 10. With three same-type hidden layers each exposing two ports,
    the LLM gets six Linear targets whose context strings are nearly
    identical ("Linear (in: Linear; out: Linear)") for h2/h3. Only the
    positional id (T1..T6) distinguishes them — this template is the
    direct test of whether the LLM reasons about position.

Start the run from the Agents panel → Autoresearch → "Start Autoresearch".
The agent walks its own outgoing `control` connections to discover what's
in scope, then mutate→train→keep/revert per trial. Training kicks off
directly — no need to click Start on the Training panel first.

A TSV ledger lands at `./.node-tool/autoresearch/<run_id>/results.tsv`.

Required:
  - Ollama on localhost:11434 (with a model pulled, e.g. `llama3.1:8b`)
  - A dataset `path` set on the A marker (or cached from a prior
    Training panel submission) so training can load data each trial.
"""
from __future__ import annotations
from core.graph import Graph
from templates._helpers import grid

LABEL = "Agent Autoresearch (3-hidden MLP — positional id test)"
DESCRIPTION = ("LLM-guided search on a 3-hidden MLP. Three same-type "
               "Linears are each wired for out_features + activation so "
               "positional ids (T1..T6) are the only way the LLM can "
               "tell them apart. Output head stays fixed at 10 classes.")


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
    from nodes.pytorch.layer                 import LayerNode
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
    data_in.inputs["path"].default_value     = "mnist"  # torchvision built-in
    graph.add_node(data_in); positions[data_in.id] = pos(col=0, row=0)

    flat = FlattenNode()
    graph.add_node(flat); positions[flat.id] = pos(col=1, row=0)

    h1 = LayerNode()
    h1.inputs["kind"].default_value         = "linear"
    h1.inputs["out_features"].default_value = 256
    h1.inputs["activation"].default_value   = "relu"
    graph.add_node(h1); positions[h1.id] = pos(col=2, row=0)

    h2 = LayerNode()
    h2.inputs["kind"].default_value         = "linear"
    h2.inputs["out_features"].default_value = 128
    h2.inputs["activation"].default_value   = "relu"
    graph.add_node(h2); positions[h2.id] = pos(col=3, row=0)

    h3 = LayerNode()
    h3.inputs["kind"].default_value         = "linear"
    h3.inputs["out_features"].default_value = 64
    h3.inputs["activation"].default_value   = "relu"
    graph.add_node(h3); positions[h3.id] = pos(col=4, row=0)

    h_out = LayerNode()
    h_out.inputs["kind"].default_value         = "linear"
    h_out.inputs["out_features"].default_value = 10     # 10 MNIST classes — fixed
    h_out.inputs["activation"].default_value   = "none"
    graph.add_node(h_out); positions[h_out.id] = pos(col=5, row=0)

    train_out = TrainMarkerNode()
    train_out.inputs["group"].default_value  = "mlp"
    train_out.inputs["kind"].default_value   = "logits"
    train_out.inputs["target"].default_value = "label"
    train_out.inputs["lr"].default_value     = 0.001
    train_out.inputs["optimizer"].default_value = "adam"
    train_out.inputs["loss"].default_value   = "crossentropy"
    train_out.inputs["epochs"].default_value = 3
    graph.add_node(train_out); positions[train_out.id] = pos(col=6, row=0)

    graph.add_connection(data_in.id,   "tensor",     flat.id,      "tensor_in")
    graph.add_connection(flat.id,      "tensor_out", h1.id,        "tensor_in")
    graph.add_connection(h1.id,        "tensor_out", h2.id,        "tensor_in")
    graph.add_connection(h2.id,        "tensor_out", h3.id,        "tensor_in")
    graph.add_connection(h3.id,        "tensor_out", h_out.id,     "tensor_in")
    graph.add_connection(h_out.id,     "tensor_out", train_out.id, "tensor_in")

    # Row 1 — the autoresearch driver
    cli = OllamaClientNode()
    cli.inputs["host"].default_value  = "http://localhost:11434"
    cli.inputs["model"].default_value = "llama3.1:8b"
    graph.add_node(cli); positions[cli.id] = pos(col=0, row=1)

    agent = AutoresearchAgentNode()
    agent.inputs["group"].default_value            = "mlp"
    agent.inputs["playbook"].default_value         = _PLAYBOOK
    agent.inputs["metric"].default_value           = "val_loss"
    agent.inputs["trials"].default_value           = 8
    # Trials are fast feedback — 1 epoch per trial is usually enough to
    # rank hyperparameter configs. Full training (10+ epochs) happens
    # later on the winning config.
    agent.inputs["epochs_per_trial"].default_value = 1
    agent.inputs["eval_budget_s"].default_value    = 120.0
    agent.inputs["wall_clock_s"].default_value     = 1800.0
    agent.inputs["temperature"].default_value      = 0.4
    graph.add_node(agent); positions[agent.id] = pos(col=2, row=1)

    # LLM into the agent
    graph.add_connection(cli.id, "llm", agent.id, "llm")

    # Control wires — these define the search space. Anything not wired
    # here is OFF-LIMITS to the agent. The canvas reads at a glance.
    # Width is safe to tune because LinearNode infers its in_features
    # from the upstream tensor on each forward, so when h1's out_features
    # changes, h2 auto-reshapes on its next forward (and so on down).
    # h_out is intentionally NOT wired — its out_features must stay 10.
    for hidden in (h1, h2, h3):
        graph.add_connection(agent.id, "control", hidden.id, "out_features")
        graph.add_connection(agent.id, "control", hidden.id, "activation")
    graph.add_connection(agent.id, "control", train_out.id, "lr")
    graph.add_connection(agent.id, "control", data_in.id,   "batch_size")

    return positions
