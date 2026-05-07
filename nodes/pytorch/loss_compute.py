"""LossComputeNode — compute a loss tensor inline as a graph operation.

The bridge that lets you compute losses INSIDE the graph instead of relying
on TrainingConfigNode's string-based loss menu. Useful for:

  - VAE-style training where the loss is `recon_loss + beta * KL` and needs
    multiple model outputs to assemble
  - Multi-task learning where each task computes its own loss tensor and a
    Math node sums them with optional weights
  - Custom losses with extra arguments
  - Anything where you want to inspect the loss as a real graph value

Used together with `TrainingConfigNode(loss_is_output=True)`: the graph's
leaf is the scalar loss tensor; the training loop just calls
`loss = model(batch); loss.backward()` with no separate loss_fn.

Supported loss types:
    mse           — F.mse_loss(pred, target)
    bce           — F.binary_cross_entropy(pred, target) — pred must be in [0,1]
    bce_logits    — F.binary_cross_entropy_with_logits(pred, target)
    cross_entropy — F.cross_entropy(pred, target) — pred is logits, target is class indices
    l1            — F.l1_loss(pred, target)
    smooth_l1     — F.smooth_l1_loss(pred, target) (Huber)
    ctc           — F.ctc_loss(log_probs, targets, input_lengths, target_lengths)
                    pred is treated as log_probs of shape (T, N, C); the node
                    auto-applies log_softmax over the C dim if the input doesn't
                    look log-normalized, and auto-transposes (N, T, C) → (T, N, C)
                    when input_lengths makes the layout unambiguous. Targets are
                    integer indices, padded (N, S) or 1-D concatenated; lengths
                    come from the `input_lengths` and `target_lengths` ports.
"""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


_LOSS_TYPES = ["mse", "bce", "bce_logits", "cross_entropy", "l1", "smooth_l1", "ctc"]


def _apply_loss(loss_type: str, pred, target):
    """Dispatch to the right F.* loss function (non-CTC paths only)."""
    import torch.nn.functional as F
    key = (loss_type or "mse").strip().lower().replace("-", "_")
    if key == "mse":
        return F.mse_loss(pred, target)
    if key == "bce":
        return F.binary_cross_entropy(pred, target)
    if key in ("bce_logits", "bcewithlogits"):
        return F.binary_cross_entropy_with_logits(pred, target)
    if key in ("cross_entropy", "crossentropy", "ce"):
        return F.cross_entropy(pred, target)
    if key == "l1":
        return F.l1_loss(pred, target)
    if key in ("smooth_l1", "huber"):
        return F.smooth_l1_loss(pred, target)
    return F.mse_loss(pred, target)


def _apply_ctc(pred, target, input_lengths, target_lengths,
               blank: int, zero_infinity: bool):
    """Apply CTC. Auto-transposes (N, T, C) → (T, N, C) and log-softmaxes pred
    if it doesn't already look log-normalized."""
    import torch.nn.functional as F

    # If pred came in (N, T, C), permute to (T, N, C). Heuristic: dim 0 equals
    # batch size (matches input_lengths.shape[0]) AND dim 1 doesn't.
    if pred.dim() == 3 and input_lengths is not None:
        N = int(input_lengths.shape[0])
        if pred.shape[0] == N and pred.shape[1] != N:
            pred = pred.transpose(0, 1).contiguous()

    # Auto log-softmax if pred doesn't already look log-normalized
    # (max > 0 along C means raw logits).
    if pred.max().item() > 0:
        pred = F.log_softmax(pred, dim=-1)

    # Clamp input_lengths to pred's actual T axis. Mel hop framing rarely
    # matches an upstream lengths-from-samples calculation exactly, and
    # F.ctc_loss raises if any input_length > T.
    import torch
    T_actual = pred.shape[0]
    if input_lengths is not None:
        input_lengths = torch.clamp(input_lengths.long(), max=T_actual)

    return F.ctc_loss(
        pred, target, input_lengths, target_lengths,
        blank=blank, zero_infinity=zero_infinity, reduction="mean",
    )


class LossComputeNode(BaseNode):
    type_name   = "pt_loss_compute"
    label       = "Loss Compute"
    category    = "Training"
    subcategory = "Loss"
    description = (
        "Compute a loss tensor as a first-class graph operation. Wire pred + "
        "target tensors and pick a loss type. Output is a scalar loss tensor "
        "that can be summed with other losses (multi-task) or fed directly "
        "into TrainingConfig with loss_is_output=True."
    )

    def relevant_inputs(self, values):
        loss_type = (values.get("loss_type") or "mse").strip().lower()
        if loss_type == "ctc":
            # CTC additionally exposes blank/zero_infinity (the lengths inputs
            # are wired data ports, always shown).
            return ["loss_type", "weight", "blank", "zero_infinity"]
        return ["loss_type", "weight"]

    def _setup_ports(self) -> None:
        self.add_input("pred",      PortType.TENSOR, default=None,
                       description="Model prediction tensor (CTC: log_probs or logits, "
                                   "shape (T,N,C) or (N,T,C); auto log-softmax + transpose).")
        self.add_input("target",    PortType.TENSOR, default=None,
                       description="Ground truth tensor (CTC: integer targets, padded (N,S) "
                                   "or 1-D concatenated).")
        self.add_input("input_lengths",  PortType.TENSOR, default=None, optional=True,
                       description="CTC only: LongTensor (N,) of pred sequence lengths.")
        self.add_input("target_lengths", PortType.TENSOR, default=None, optional=True,
                       description="CTC only: LongTensor (N,) of target sequence lengths.")
        self.add_input("loss_type", PortType.STRING, default="mse",
                       choices=_LOSS_TYPES)
        self.add_input("weight",    PortType.FLOAT,  default=1.0,
                       description="Multiplier on the loss before output — useful for multi-task weighting")
        self.add_input("blank",          PortType.INT,  default=0, optional=True,
                       description="CTC: index of the blank token (typically 0).")
        self.add_input("zero_infinity",  PortType.BOOL, default=True, optional=True,
                       description="CTC: mask infinite losses from impossible alignments.")
        self.add_output("loss", PortType.TENSOR,
                        description="Scalar loss tensor")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        pred   = inputs.get("pred")
        target = inputs.get("target")
        loss_type = (inputs.get("loss_type") or "mse").strip().lower()
        weight = float(inputs.get("weight") or 1.0)
        if pred is None or target is None:
            return {"loss": None}
        try:
            if loss_type == "ctc":
                in_lens = inputs.get("input_lengths")
                tg_lens = inputs.get("target_lengths")
                if in_lens is None or tg_lens is None:
                    return {"loss": None}
                loss = _apply_ctc(pred, target, in_lens, tg_lens,
                                  blank=int(inputs.get("blank") or 0),
                                  zero_infinity=bool(inputs.get("zero_infinity", True)))
            else:
                loss = _apply_loss(loss_type, pred, target)
            if weight != 1.0:
                loss = loss * weight
            return {"loss": loss}
        except Exception:
            return {"loss": None}

    def export(self, iv, ov):
        pred   = iv.get("pred")   or "None  # TODO: connect pred"
        target = iv.get("target") or "None  # TODO: connect target"
        loss_type = (self.inputs["loss_type"].default_value or "mse").strip().lower().replace("-", "_")
        weight = float(self.inputs["weight"].default_value or 1.0)
        out = ov.get("loss", "_loss")

        if loss_type == "ctc":
            in_lens = iv.get("input_lengths")  or "None  # TODO: connect input_lengths"
            tg_lens = iv.get("target_lengths") or "None  # TODO: connect target_lengths"
            blank   = int(self.inputs["blank"].default_value or 0)
            zi      = bool(self.inputs["zero_infinity"].default_value)
            lines = [
                # Auto-transpose if pred came in (N, T, C); auto log-softmax if needed.
                f"_p = {pred}",
                f"if _p.dim() == 3 and {in_lens} is not None and "
                f"_p.shape[0] == {in_lens}.shape[0] and _p.shape[1] != {in_lens}.shape[0]:",
                f"    _p = _p.transpose(0, 1).contiguous()",
                f"if _p.max().item() > 0:",
                f"    _p = F.log_softmax(_p, dim=-1)",
                f"{out} = F.ctc_loss(_p, {target}, {in_lens}, {tg_lens}, "
                f"blank={blank}, zero_infinity={zi}, reduction='mean')",
            ]
            if weight != 1.0:
                lines.append(f"{out} = {out} * {weight}")
            return ["import torch", "import torch.nn.functional as F"], lines

        loss_call_map = {
            "mse":           f"F.mse_loss({pred}, {target})",
            "bce":           f"F.binary_cross_entropy({pred}, {target})",
            "bce_logits":    f"F.binary_cross_entropy_with_logits({pred}, {target})",
            "cross_entropy": f"F.cross_entropy({pred}, {target})",
            "l1":            f"F.l1_loss({pred}, {target})",
            "smooth_l1":     f"F.smooth_l1_loss({pred}, {target})",
        }
        call = loss_call_map.get(loss_type, loss_call_map["mse"])
        if weight != 1.0:
            return ["import torch", "import torch.nn.functional as F"], [
                f"{out} = {call} * {weight}",
            ]
        return ["import torch", "import torch.nn.functional as F"], [
            f"{out} = {call}",
        ]
