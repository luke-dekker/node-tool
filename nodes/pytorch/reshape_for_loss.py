"""ReshapeForLossNode — flatten sequence dim into batch for token-level loss.

The bridge between sequence models (LSTM, GRU, Transformer encoder) and
classification losses (CrossEntropyLoss, BCEWithLogitsLoss) when you want a
loss per timestep.

Why this exists alongside TensorReshapeNode: TensorReshape is a generic
single-tensor shape mutation (you tell it the new shape). ReshapeForLoss
has SEMANTIC meaning — it knows the (B, T, V) → (B*T, V) and (B, T) →
(B*T,) contract for sequence loss prep, takes both logits and labels in
one node, and labels the outputs accordingly. For everything else (general
reshapes, broadcasting prep, etc.) use TensorReshape.

Sequence model outputs are typically:
    logits: (B, T, V)        — B batches, T timesteps, V vocabulary size
    labels: (B, T)           — same B and T, integer class indices per timestep

But CrossEntropyLoss expects:
    logits: (N, V)           — N samples, V classes
    labels: (N,)             — N integer class indices

This node reshapes (B, T, V) -> (B*T, V) and (B, T) -> (B*T,) so the loss can
be computed cleanly. It is the canonical "flatten before loss" step in any
language model or sequence labeling setup.

If you only have logits and no labels (inference mode), only the logits port
gets reshaped. If you only have labels (testing the reshape itself), only the
labels port gets reshaped.
"""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


class ReshapeForLossNode(BaseNode):
    type_name   = "pt_reshape_for_loss"
    label       = "Reshape For Loss"
    category    = "Layers"
    subcategory = "Recurrent"
    description = (
        "Flatten the sequence dim into the batch dim so token-level CrossEntropy "
        "works on sequence model outputs. logits (B, T, V) -> (B*T, V) and "
        "labels (B, T) -> (B*T,). Use it between an LSTM/GRU forward and a "
        "loss for char-level / word-level language models."
    )

    def _setup_ports(self) -> None:
        self.add_input("logits", PortType.TENSOR, default=None,
                       description="Shape (B, T, V) — sequence model output")
        self.add_input("labels", PortType.TENSOR, default=None,
                       description="Shape (B, T) — integer class indices")
        self.add_output("logits_flat", PortType.TENSOR,
                        description="Shape (B*T, V) — ready for CrossEntropy")
        self.add_output("labels_flat", PortType.TENSOR,
                        description="Shape (B*T,) — ready for CrossEntropy")
        self.add_output("info",        PortType.STRING)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        empty = {"logits_flat": None, "labels_flat": None, "info": ""}
        logits = inputs.get("logits")
        labels = inputs.get("labels")

        out = dict(empty)
        notes: list[str] = []
        try:
            if logits is not None:
                # (B, T, V) -> (B*T, V). Use reshape so non-contiguous inputs
                # (e.g. from a transposed RNN output) still work.
                v = logits.shape[-1]
                out["logits_flat"] = logits.reshape(-1, v)
                notes.append(f"logits {tuple(logits.shape)} -> ({out['logits_flat'].shape[0]}, {v})")
            if labels is not None:
                # (B, T) -> (B*T,). If labels are already 1D we leave them alone.
                if labels.dim() >= 2:
                    out["labels_flat"] = labels.reshape(-1)
                else:
                    out["labels_flat"] = labels
                notes.append(f"labels {tuple(labels.shape)} -> ({out['labels_flat'].shape[0]},)")
            out["info"] = "; ".join(notes) if notes else "no input connected"
            return out
        except Exception as exc:
            return {**empty, "info": f"reshape failed: {exc}"}

    def export(self, iv, ov):
        logits      = iv.get("logits") or "None  # TODO: connect logits tensor"
        labels      = iv.get("labels")
        logits_var  = ov.get("logits_flat", "_logits_flat")
        labels_var  = ov.get("labels_flat", "_labels_flat")
        info_var    = ov.get("info",        "_reshape_info")
        lines = [
            f"# Flatten sequence dim into batch for token-level loss",
            f"{logits_var} = {logits}.reshape(-1, {logits}.shape[-1])",
        ]
        if labels:
            lines.append(
                f"{labels_var} = {labels}.reshape(-1) if {labels}.dim() >= 2 else {labels}"
            )
        else:
            lines.append(f"{labels_var} = None  # no labels connected")
        lines.append(
            f"{info_var} = f'flattened to ({{{logits_var}.shape[0]}}, {{{logits_var}.shape[-1]}})'"
        )
        return ["import torch"], lines
