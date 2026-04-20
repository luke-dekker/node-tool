"""Agent Autoresearch — LLM-driven mutate → eval → keep/revert on a tiny MLP.

    Data In (A:x) ─► Flatten ─► Linear ─► ReLU ─► Linear ─► Data Out (B)
         └─── mutation region (autoresearch cone between A and B) ────┘

    OllamaClient ─llm─► Mutator ─► (proposes set_input / swap_node_class ops)
    Evaluator    ─► EvalSpec (metric=val_loss, budget=60s)
    ExperimentLoop ─► LoopSpec (trials=3, wall_clock_s=300)

Start the run from the Agents panel → Autoresearch → "Start Autoresearch".
The orchestrator:
  1. Snapshots the graph
  2. Asks the Mutator for ONE op against the A/B cone
  3. Applies it (gated by `allowlist`)
  4. Fires `train_start` on the pytorch plugin via the registry
  5. Polls until done, scores final `val_loss`
  6. Keeps the mutation if better, reverts otherwise

A TSV ledger lands at `./.node-tool/autoresearch/<run_id>/results.tsv`.

Required:
  - Ollama running on localhost:11434 (for the Mutator's LLM)
  - A training target configured in the Training panel (dataset, optimizer,
    loss) — the Evaluator delegates training to the pytorch plugin.
"""
from __future__ import annotations
from core.graph import Graph
from templates._helpers import grid

LABEL = "Agent Autoresearch (mutate → eval → keep MLP)"
DESCRIPTION = ("LLM-driven architecture search on a tiny MLP. Mutates the "
               "A/B cone, evaluates via the training panel, keeps the winner.")


def build(graph: Graph) -> dict[str, tuple[int, int]]:
    from nodes.pytorch.input_marker   import InputMarkerNode
    from nodes.pytorch.flatten        import FlattenNode
    from nodes.pytorch.linear         import LinearNode
    from nodes.pytorch.train_marker   import TrainMarkerNode
    from nodes.agents.ollama_client       import OllamaClientNode
    from nodes.agents.mutator             import MutatorNode
    from nodes.agents.evaluator           import EvaluatorNode
    from nodes.agents.experiment_loop     import ExperimentLoopNode

    pos = grid(step_x=220, step_y=170)
    positions: dict[str, tuple[int, int]] = {}

    # Row 0 — the trainable pipeline (mutation cone)
    data_in = InputMarkerNode()
    data_in.inputs["modality"].default_value = "x"
    data_in.inputs["group"].default_value    = "mlp"
    graph.add_node(data_in); positions[data_in.id] = pos(col=0, row=0)

    flat = FlattenNode()
    graph.add_node(flat); positions[flat.id] = pos(col=1, row=0)

    h1 = LinearNode()
    h1.inputs["in_features"].default_value  = 784
    h1.inputs["out_features"].default_value = 64
    h1.inputs["activation"].default_value   = "relu"
    graph.add_node(h1); positions[h1.id] = pos(col=2, row=0)

    h2 = LinearNode()
    h2.inputs["in_features"].default_value  = 64
    h2.inputs["out_features"].default_value = 10
    h2.inputs["activation"].default_value   = "none"
    graph.add_node(h2); positions[h2.id] = pos(col=3, row=0)

    train_out = TrainMarkerNode()
    train_out.inputs["group"].default_value = "mlp"
    train_out.inputs["kind"].default_value  = "logits"
    graph.add_node(train_out); positions[train_out.id] = pos(col=4, row=0)

    graph.add_connection(data_in.id, "tensor",     flat.id, "tensor_in")
    graph.add_connection(flat.id,    "tensor_out", h1.id,   "tensor_in")
    graph.add_connection(h1.id,      "tensor_out", h2.id,   "tensor_in")
    graph.add_connection(h2.id,      "tensor_out", train_out.id, "tensor_in")

    # Row 1 — the autoresearch driver
    cli = OllamaClientNode()
    cli.inputs["model"].default_value = "qwen2.5:0.5b"
    graph.add_node(cli); positions[cli.id] = pos(col=0, row=1)

    mut = MutatorNode()
    mut.inputs["group"].default_value = "mlp"
    mut.inputs["playbook"].default_value = (
        "Propose ONE change per iteration: swap a Linear's activation "
        "between relu/gelu/tanh, or adjust its out_features to a power of 2 "
        "between 32 and 512. Do not add or remove nodes. Emit ONLY the JSON "
        "mutation op, no prose."
    )
    mut.inputs["temperature"].default_value = 0.4
    graph.add_node(mut); positions[mut.id] = pos(col=1, row=1)

    evl = EvaluatorNode()
    evl.inputs["metric"].default_value         = "val_loss"
    evl.inputs["budget_seconds"].default_value = 60.0
    evl.inputs["epochs"].default_value         = 3
    evl.inputs["group"].default_value          = "mlp"
    graph.add_node(evl); positions[evl.id] = pos(col=2, row=1)

    loop = ExperimentLoopNode()
    loop.inputs["trials"].default_value         = 3
    loop.inputs["wall_clock_s"].default_value   = 300.0
    loop.inputs["loss_threshold"].default_value = 0.0
    loop.inputs["allowlist"].default_value      = "pt_linear,pt_flatten"
    graph.add_node(loop); positions[loop.id] = pos(col=3, row=1)

    # Mutator needs an LLM to propose ops
    graph.add_connection(cli.id, "llm", mut.id, "llm")

    return positions
