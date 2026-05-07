"""Loss Function factory node — emits an nn.* loss callable selected by dropdown.

Supports CTC alongside the basic supervised losses. CTC is special: its
constructor takes `blank` (the blank token index, conventionally 0) and
`zero_infinity` (whether to mask infinite losses from impossible alignments)
instead of `reduction` alone. The inspector hides those fields for the other
loss types via `relevant_inputs`.
"""
from __future__ import annotations
from core.node import BaseNode, PortType


# (display_class_name, factory(nn, reduction))
# CTC handled separately because its constructor signature differs.
_LOSS_MAP = {
    "mse":           ("MSELoss",            lambda nn, r: nn.MSELoss(reduction=r)),
    "cross_entropy": ("CrossEntropyLoss",   lambda nn, r: nn.CrossEntropyLoss()),
    "bce":           ("BCELoss",            lambda nn, r: nn.BCELoss(reduction=r)),
    "bce_logits":    ("BCEWithLogitsLoss",  lambda nn, r: nn.BCEWithLogitsLoss(reduction=r)),
    "l1":            ("L1Loss",             lambda nn, r: nn.L1Loss(reduction=r)),
}


class LossFnNode(BaseNode):
    type_name   = "pt_loss_fn"
    label       = "Loss Function"
    category    = "Training"
    subcategory = "Loss"
    description = (
        "Construct a PyTorch loss function. Pick `loss_type`:\n"
        "  mse / l1                     — regression\n"
        "  bce / bce_logits             — binary classification\n"
        "  cross_entropy                — multi-class classification\n"
        "  ctc                          — CTC for sequence labeling (ASR, OCR);\n"
        "                                 forward needs log_probs + targets +\n"
        "                                 input_lengths + target_lengths"
    )

    def relevant_inputs(self, values):
        loss_type = (values.get("loss_type") or "mse").strip().lower()
        if loss_type == "ctc":
            return ["loss_type", "blank", "zero_infinity", "reduction"]
        return ["loss_type", "reduction"]

    def _setup_ports(self):
        self.add_input("loss_type",  PortType.STRING, default="mse",
                       choices=["mse", "cross_entropy", "bce", "bce_logits", "l1", "ctc"])
        self.add_input("reduction",  PortType.STRING, default="mean",
                       choices=["mean", "sum", "none"])
        # CTC-only
        self.add_input("blank",          PortType.INT,  default=0, optional=True,
                       description="CTC: index of the blank token in your vocabulary.")
        self.add_input("zero_infinity",  PortType.BOOL, default=True, optional=True,
                       description="CTC: zero out losses on impossible alignments "
                                   "(target longer than input). Recommended True.")
        self.add_output("loss_fn", PortType.LOSS_FN)

    def execute(self, inputs):
        try:
            import torch.nn as nn
            loss_type = str(inputs.get("loss_type", "mse") or "mse")
            reduction = str(inputs.get("reduction", "mean") or "mean")
            if loss_type == "ctc":
                return {"loss_fn": nn.CTCLoss(
                    blank=int(inputs.get("blank") or 0),
                    reduction=reduction,
                    zero_infinity=bool(inputs.get("zero_infinity", True)),
                )}
            entry = _LOSS_MAP.get(loss_type)
            if entry is None:
                return {"loss_fn": None}
            _, factory = entry
            return {"loss_fn": factory(nn, reduction)}
        except Exception:
            return {"loss_fn": None}

    def export(self, iv, ov):
        loss_type = (self.inputs["loss_type"].default_value or "mse")
        reduction = (self.inputs["reduction"].default_value or "mean")
        if loss_type == "ctc":
            blank = int(self.inputs["blank"].default_value or 0)
            zi    = bool(self.inputs["zero_infinity"].default_value)
            return ["import torch.nn as nn"], [
                f"{ov['loss_fn']} = nn.CTCLoss(blank={blank}, "
                f"reduction={reduction!r}, zero_infinity={zi})"
            ]
        entry = _LOSS_MAP.get(loss_type)
        cls_name = entry[0] if entry else "MSELoss"
        no_reduction = loss_type == "cross_entropy"
        args = "" if no_reduction else f"reduction={repr(reduction)}"
        return ["import torch.nn as nn"], [
            f"{ov['loss_fn']} = nn.{cls_name}({args})"
        ]
