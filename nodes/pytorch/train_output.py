"""TrainOutputNode — lightweight marker for the training target.

The graph-side half of the training system. Marks a tensor as the training
target (either model logits or a pre-computed scalar loss). The Training Panel
discovers TrainOutput nodes by scanning the graph, reads the tensor during
training, and computes the optimizer step.

Multiple TrainOutput nodes in one graph = multi-task training. The panel maps
each one to a dataset and alternates/combines as configured.

This replaces TrainingConfigNode: hyperparameters (epochs, lr, optimizer,
device) move to the Training Panel as persistent UI widgets. The graph only
declares WHAT to optimize; the panel declares HOW.
"""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


class TrainOutputNode(BaseNode):
    type_name   = "pt_train_output"
    label       = "Train Output"
    category    = "Training"
    subcategory = "Config"
    description = (
        "Mark a tensor as the training target. Wire model logits or a "
        "pre-computed loss here. The Training Panel discovers this node and "
        "uses its input as the optimization target. For multi-task: add "
        "one Train Output per task; the panel maps each to a dataset."
    )

    def _setup_ports(self) -> None:
        self.add_input("tensor_in", PortType.TENSOR, default=None,
                       description="Model output (logits) or scalar loss tensor")
        self.add_input("loss_is_output", PortType.BOOL, default=False,
                       description="If True, tensor_in is already a scalar loss; "
                                   "skip loss_fn in the training loop")
        self.add_input("task_name", PortType.STRING, default="",
                       description="Optional label shown in the panel's task table "
                                   "(auto-generated from the node label if empty)")
        self.add_output("config", PortType.ANY,
                        description="Pass-through for backward compat with "
                                    "GraphAsModule output targeting")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        # Pass through the tensor + flags so GraphAsModule can read them
        return {
            "config": {
                "tensor_in":      inputs.get("tensor_in"),
                "loss_is_output": bool(inputs.get("loss_is_output", False)),
                "task_name":      str(inputs.get("task_name") or self.label),
            },
        }

    def export(self, iv, ov):
        tin = iv.get("tensor_in") or "None  # TODO: connect model output or loss"
        lio = bool(self.inputs["loss_is_output"].default_value)
        name = self.inputs["task_name"].default_value or "default"
        out = ov.get("config", "_train_output")
        return [], [
            f"# Train Output: {name!r} (loss_is_output={lio})",
            f"{out} = {tin}  # training target",
        ]
