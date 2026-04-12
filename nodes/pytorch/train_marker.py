"""TrainMarkerNode — "B" marker for training graphs.

Half of the A/B marker pair that replaces DatasetNode + TrainOutputNode. B
markers are dumb — they just label a tensor as the target of a training
section. The training panel discovers them, reads the `kind` (logits or
pre-computed loss), and runs the optimizer step accordingly.

Multiple B markers with different `group` values = multi-task training.
Each group pairs with same-group A markers to form one trainable section.

Design mirror of TrainOutputNode (which this replaces), minus all the
legacy fallback plumbing.
"""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


class TrainMarkerNode(BaseNode):
    type_name   = "pt_train_marker"
    label       = "Data Out (B)"
    category    = "Training"
    subcategory = "Markers"
    description = (
        "Training target marker. Pairs with Data In (A) markers of the same "
        "group. Wire the graph section's output tensor into tensor_in — "
        "that tensor becomes the optimization target. Set kind='loss' if the "
        "tensor is already a pre-computed scalar loss (VAE-style graphs); "
        "otherwise kind='logits' and the panel applies the configured loss "
        "function against the batch's label tensor."
    )

    def _setup_ports(self) -> None:
        self.add_input("tensor_in", PortType.TENSOR, default=None,
                       description="Graph section output (logits or scalar loss)")
        self.add_input("group",     PortType.STRING, default="task_1",
                       description="Pairs with Data In (A) markers of the same group")
        self.add_input("kind",      PortType.STRING, default="logits",
                       choices=["logits", "loss"],
                       description="'logits' = panel computes loss vs target column; "
                                   "'loss' = tensor_in is already a scalar loss")
        self.add_input("target",    PortType.STRING, default="label",
                       description="Batch column name for the ground-truth tensor "
                                   "(used when kind='logits'; ignored when kind='loss')")
        self.add_input("task_name", PortType.STRING, default="",
                       description="Optional display label; defaults to group")
        self.add_output("config",   PortType.ANY,
                        description="Pass-through for GraphAsModule output targeting")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {
            "config": {
                "tensor_in": inputs.get("tensor_in"),
                "kind":      str(inputs.get("kind") or "logits"),
                "group":     str(inputs.get("group") or "task_1"),
                "target":    str(inputs.get("target") or "label"),
                "task_name": str(inputs.get("task_name") or
                                 inputs.get("group") or "task_1"),
            },
        }

    def export(self, iv, ov):
        tin   = iv.get("tensor_in") or "None  # TODO: connect model output or loss"
        kind  = self.inputs["kind"].default_value or "logits"
        group = self.inputs["group"].default_value or "task_1"
        out   = ov.get("config", "_train_marker")
        return [], [
            f"# Data Out marker: group={group!r} kind={kind!r}",
            f"{out} = {tin}  # training target",
        ]
