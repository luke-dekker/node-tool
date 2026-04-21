"""AutoresearchAgentNode — single-node driver for LLM-guided architecture
search.

Replaces the old MutatorNode + EvaluatorNode + ExperimentLoopNode trio.
The user wires `llm` to an LLM client and wires `control` (an output
port) into any input port the agent may tune — `linear.out_features`,
`dropout.p`, `B_marker.lr`, `A_marker.batch_size`, etc. The set of wires
out of `control` IS the search space; nothing else is mutable.

Triggered from the Agents panel's Autoresearch section. The orchestrator
runs the loop on a daemon thread; the node's outputs (`best_score`,
`best_hash`, `history`) reflect the latest state — `control` itself
emits None during normal `graph.execute()` because it's a relationship
marker, not a data wire.

Design philosophy: every wire on the canvas means something. No floating
config nodes; no hidden orchestrator scans by `type_name`. Scope of
agent control is visible at a glance.
"""
from __future__ import annotations
from typing import Any

from core.node import BaseNode, PortType


_DEFAULT_PLAYBOOK = """\
Propose ONE trial's worth of changes per call. You may modify any subset
of the controllable parameters listed below — small steps work better
than big swings.

Heuristics:
  - For width/hidden-size INT params, try powers of 2 between 32 and 512.
  - For learning-rate FLOAT params, try log-uniform between 1e-5 and 1e-2.
  - For activations (STRING choices), prefer relu/gelu/silu/tanh.
  - For dropout/regularization rates, try 0.0 to 0.5.

Output JSON only — no prose, no code fences:
  {"changes": [{"target": "<target_id>", "value": <new value>}, ...]}
"""


class AutoresearchAgentNode(BaseNode):
    type_name   = "ag_autoresearch"
    label       = "Autoresearch Agent"
    category    = "Agents"
    subcategory = "Autoresearch"
    description = (
        "LLM-guided architecture search. Wire `llm` to a backend and wire "
        "`control` into any port the agent may tune. The set of `control` "
        "wires defines the search space. Trigger from the Agents panel."
    )

    def __init__(self):
        # Live state mirrored to output ports — written by the orchestrator's
        # control loop on each trial completion. Defaults so a fresh node
        # reads as 'idle / no result yet'.
        self._best_score: float = float("inf")
        self._best_hash:  str   = ""
        self._history:    list  = []
        super().__init__()

    def _setup_ports(self) -> None:
        self.add_input("llm", "LLM", default=None,
                       description="LLM client (Ollama / OpenAI-compat / llama.cpp)")
        self.add_input("playbook", PortType.STRING, default=_DEFAULT_PLAYBOOK,
                       description="Search strategy / heuristics for the LLM")
        self.add_input("group", PortType.STRING, default="task_1",
                       description="A/B group whose training the agent evaluates")
        self.add_input("metric", PortType.STRING, default="val_loss",
                       choices=["val_loss", "train_loss", "accuracy"],
                       description="Metric to minimize (loss) / maximize (accuracy)")
        self.add_input("trials", PortType.INT, default=8,
                       description="Maximum mutate→eval cycles")
        self.add_input("wall_clock_s", PortType.FLOAT, default=900.0,
                       description="Total wall-clock budget across all trials")
        self.add_input("eval_budget_s", PortType.FLOAT, default=60.0,
                       description="Per-trial training budget")
        self.add_input("loss_threshold", PortType.FLOAT, default=0.0,
                       description="Stop early if best score ≤ this. 0 = disabled.")
        self.add_input("temperature", PortType.FLOAT, default=0.4,
                       description="LLM sampling temperature for proposals")

        # `control` is a relationship marker — wire it INTO any input port
        # the agent may tune. During normal graph execute it emits None,
        # so downstream targets fall back to their own default_value
        # (which is what the agent has been writing into).
        self.add_output("control", PortType.ANY,
                        description="Wire into any port the agent may tune. "
                                    "Each connection adds that port to the "
                                    "search space.")
        self.add_output("best_score", PortType.FLOAT,
                        description="Best metric value seen so far across trials")
        self.add_output("best_hash", PortType.STRING,
                        description="Graph hash of the best-scoring trial")
        self.add_output("history", PortType.ANY,
                        description="List of trial summaries (id, op, score, kept/discard)")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {
            "control":    None,
            "best_score": self._best_score,
            "best_hash":  self._best_hash,
            "history":    list(self._history),
        }

    def export(self, iv, ov):
        # Autoresearch is an interactive driver — it doesn't make sense
        # in a one-shot exported script. Emit a stub that explains why.
        return [], [
            "# AutoresearchAgentNode is a runtime-only driver and is not "
            "exported.",
            "# Train the winning configuration directly instead.",
        ]
