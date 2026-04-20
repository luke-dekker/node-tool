"""ExperimentLoopNode — budget config for the autoresearch driver.

Emits a `LoopSpec` dict. The orchestrator's `autoresearch_start` RPC reads
this spec (plus EvaluatorSpec + MutatorNode config) to drive the loop on a
background thread.
"""
from __future__ import annotations
from typing import Any

from core.node import BaseNode, PortType


class ExperimentLoopNode(BaseNode):
    type_name   = "ag_experiment_loop"
    label       = "Experiment Loop"
    category    = "Agents"
    subcategory = "Autoresearch"
    description = ("Budget for the autoresearch driver. The orchestrator "
                   "runs mutate→eval→keep/revert until one of the stop "
                   "conditions is hit.")

    def _setup_ports(self) -> None:
        self.add_input("trials", PortType.INT, default=5,
                       description="Max mutate→eval cycles")
        self.add_input("wall_clock_s", PortType.FLOAT, default=300.0,
                       description="Total wall-clock budget (all trials combined)")
        self.add_input("loss_threshold", PortType.FLOAT, default=0.0,
                       description=("Stop early if best_score ≤ threshold. "
                                    "0 = disabled."))
        self.add_input("allowlist", PortType.STRING, default="",
                       description=("Comma-separated type_names the mutator "
                                    "may add/swap to. Blank = all registered."))
        self.add_output("loop_spec", PortType.ANY,
                        description="{trials, wall_clock_s, loss_threshold, allowlist}")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        raw_al = (inputs.get("allowlist") or "").strip()
        allowlist = [x.strip() for x in raw_al.split(",") if x.strip()] if raw_al else []
        lt = float(inputs.get("loss_threshold") or 0.0)
        return {
            "loop_spec": {
                "trials":          max(1, int(inputs.get("trials") or 5)),
                "wall_clock_s":    float(inputs.get("wall_clock_s") or 300.0),
                "loss_threshold":  (lt if lt > 0 else None),
                "allowlist":       allowlist,
            },
        }

    def export(self, iv, ov):
        return [], [f"# ExperimentLoopNode export pending Phase D"]
