"""Freeze Layers node — granular freeze controls for any nn.Module.

Replaces the old "freeze everything or nothing" behavior with explicit modes:

  - "all":     freeze every parameter (feature extractor mode)
  - "none":    unfreeze every parameter (full fine-tuning)
  - "first_n": freeze the first N top-level children, leave the rest trainable
               (the canonical 'freeze early layers, fine-tune later layers' for
               transfer learning — though for ResNet etc. you usually want
               FreezeNamedLayersNode with the explicit name list)
  - "last_n":  freeze the last N top-level children, leave earlier ones trainable

The legacy `freeze_all` BOOL is still accepted for backward compat with saved
graphs and tests, but new graphs should use `mode` instead. The class was
originally named FreezeBackboneNode but never actually did anything backbone-
specific — it was just "freeze all parameters". The new name reflects what
it actually does, and the alias keeps existing imports working.
"""
from __future__ import annotations
from core.node import BaseNode, PortType


class FreezeLayersNode(BaseNode):
    type_name   = "pt_freeze_backbone"   # kept for backward-compat with saves
    label       = "Freeze Layers"
    category    = "Models"
    subcategory = "Pretrained"
    description = (
        "Set requires_grad on a model's parameters. "
        "mode='all' freezes everything (feature extractor); "
        "'none' unfreezes everything; "
        "'first_n'/'last_n' freeze the first or last N top-level children. "
        "For named-prefix freezing (e.g. ResNet's 'conv1,bn1,layer1..layer4') "
        "use Freeze Named Layers instead."
    )

    def _setup_ports(self):
        self.add_input("model", PortType.MODULE, default=None)
        self.add_input("mode",  PortType.STRING, default="all",
                       choices=["all", "none", "first_n", "last_n"])
        self.add_input("n",     PortType.INT,    default=0,
                       description="Count for first_n / last_n modes")
        # Legacy boolean — still accepted for backward compat with old graphs
        # and tests. If `mode` is the default "all" and freeze_all is explicitly
        # set, we honor freeze_all. New graphs should leave this and use mode.
        self.add_input("freeze_all", PortType.BOOL, default=True,
                       description="DEPRECATED: use `mode` instead.")
        self.add_output("model", PortType.MODULE)
        self.add_output("info",  PortType.STRING)

    def execute(self, inputs):
        try:
            model = inputs.get("model")
            if model is None:
                return {"model": None, "info": "No model"}

            mode = str(inputs.get("mode") or "all").strip().lower()
            n    = max(0, int(inputs.get("n") or 0))
            children = list(model.children())

            if mode == "all":
                # Honor legacy freeze_all bool here so old test/save defaults work
                freeze = bool(inputs.get("freeze_all", True))
                for p in model.parameters():
                    p.requires_grad = not freeze
                action = "frozen all" if freeze else "unfrozen all"
            elif mode == "none":
                for p in model.parameters():
                    p.requires_grad = True
                action = "unfrozen all"
            elif mode == "first_n":
                for p in model.parameters():
                    p.requires_grad = True
                for child in children[:n]:
                    for p in child.parameters():
                        p.requires_grad = False
                action = f"frozen first {min(n, len(children))} of {len(children)} children"
            elif mode == "last_n":
                for p in model.parameters():
                    p.requires_grad = True
                for child in children[-n:] if n > 0 else []:
                    for p in child.parameters():
                        p.requires_grad = False
                action = f"frozen last {min(n, len(children))} of {len(children)} children"
            else:
                return {"model": model, "info": f"Unknown mode: {mode}"}

            frozen    = sum(1 for p in model.parameters() if not p.requires_grad)
            trainable = sum(1 for p in model.parameters() if p.requires_grad)
            total     = frozen + trainable
            info = (f"{action} — frozen={frozen:,}/{total:,} param tensors "
                    f"(trainable={trainable:,})")
            return {"model": model, "info": info}
        except Exception as exc:
            return {"model": None, "info": f"error: {exc}"}

    def export(self, iv, ov):
        in_model = iv.get("model") or "None  # TODO: connect a model"
        mv      = ov.get("model", "_frozen_model")
        iv_var  = ov.get("info",  "_frozen_model_info")
        mode    = str(self.inputs["mode"].default_value or "all").strip().lower()
        n       = max(0, int(self.inputs["n"].default_value or 0))
        freeze_all_legacy = bool(self.inputs["freeze_all"].default_value)

        lines = [f"{mv} = {in_model}", f"_children = list({mv}.children())"]

        if mode == "all":
            lines.append(
                f"for _p in {mv}.parameters(): _p.requires_grad = {not freeze_all_legacy}"
            )
        elif mode == "none":
            lines.append(f"for _p in {mv}.parameters(): _p.requires_grad = True")
        elif mode == "first_n":
            lines += [
                f"for _p in {mv}.parameters(): _p.requires_grad = True",
                f"for _child in _children[:{n}]:",
                f"    for _p in _child.parameters(): _p.requires_grad = False",
            ]
        elif mode == "last_n":
            lines += [
                f"for _p in {mv}.parameters(): _p.requires_grad = True",
                f"for _child in _children[-{n}:] if {n} > 0 else []:",
                f"    for _p in _child.parameters(): _p.requires_grad = False",
            ]

        lines += [
            f"_n_frozen = sum(1 for _p in {mv}.parameters() if not _p.requires_grad)",
            f"_n_train  = sum(1 for _p in {mv}.parameters() if _p.requires_grad)",
            f"{iv_var} = f'{{_n_frozen:,}} frozen / {{_n_train:,}} trainable'",
        ]
        return ["import torch"], lines


# Backward-compat alias — old code paths importing FreezeBackboneNode still work
FreezeBackboneNode = FreezeLayersNode
