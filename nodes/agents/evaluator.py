"""EvaluatorNode — config carrier for the autoresearch trial scoring policy.

Emits an `EvalSpec` dict (metric name, per-trial wall-clock budget, training
start params). The actual poll loop lives in
`plugins.agents._autoresearch.evaluator.run_eval`, driven by the orchestrator
— not by this node's execute(). Keeping the node as a config emitter lets
Graph.execute() run in any context without pulling training dependencies.
"""
from __future__ import annotations
from typing import Any

from core.node import BaseNode, PortType


class EvaluatorNode(BaseNode):
    type_name   = "ag_evaluator"
    label       = "Evaluator"
    category    = "Agents"
    subcategory = "Autoresearch"
    description = ("Scoring policy for one autoresearch trial. Emits an "
                   "EvalSpec dict the orchestrator consumes.")

    def _setup_ports(self) -> None:
        self.add_input("metric", PortType.STRING, default="val_loss",
                       choices=["val_loss", "train_loss", "accuracy"],
                       description="Metric to minimize (for loss) / maximize (for accuracy)")
        self.add_input("budget_seconds", PortType.FLOAT, default=60.0,
                       description="Per-trial wall-clock cap in seconds")
        self.add_input("epochs", PortType.INT, default=5,
                       description="Training epochs passed to train_start")
        self.add_input("group", PortType.STRING, default="task_1",
                       description="Which A/B group to train (multi-task graphs)")
        self.add_output("eval_spec", PortType.ANY,
                        description="{metric, budget_seconds, epochs, group}")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {
            "eval_spec": {
                "metric":          (inputs.get("metric") or "val_loss").strip()
                                    or "val_loss",
                "budget_seconds":  float(inputs.get("budget_seconds") or 60.0),
                "epochs":          int(inputs.get("epochs") or 5),
                "group":           (inputs.get("group") or "task_1").strip()
                                    or "task_1",
            },
        }

    def export(self, iv, ov):
        return [], [f"# EvaluatorNode export pending Phase D"]
