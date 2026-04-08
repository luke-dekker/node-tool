"""Load Model node."""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


class LoadModelNode(BaseNode):
    """Load a saved model into the tensor-flow path. Supports training and freeze."""
    type_name   = "pt_load_model"
    label       = "Load Model"
    category    = "Models"
    subcategory = "Pretrained"
    description = (
        "Load a full model saved with Save Model. Sits in the tensor-flow path "
        "(tensor_in -> tensor_out) and participates in training via get_layers(). "
        "Use freeze to lock weights for feature extraction, or trainable_layers to "
        "fine-tune only the last N layers."
    )

    def __init__(self):
        self._model = None
        self._loaded_path = ""
        super().__init__()

    def _setup_ports(self) -> None:
        self.add_input("tensor_in",        PortType.TENSOR, default=None)
        self.add_input("path",             PortType.STRING, default="model.pt",
                       description="Path to load from")
        self.add_input("device",           PortType.STRING, default="cpu")
        self.add_input("freeze",           PortType.BOOL,   default=False,
                       description="Freeze all parameters (feature extractor mode)")
        self.add_input("trainable_layers", PortType.INT,    default=0,
                       description="Unfreeze this many layers from the end (0 = respect freeze)")
        self.add_input("save_path",        PortType.STRING, default="",
                       description="If set, save model here on every graph run")
        self.add_input("save_mode",        PortType.STRING, default="overwrite",
                       choices=["overwrite", "increment"],
                       description="overwrite: replace same file  |  increment: model_1.pt, model_2.pt ...")
        self.add_output("tensor_out", PortType.TENSOR)
        self.add_output("info",       PortType.STRING,
                        description="Model class, param count, frozen/trainable summary")

    def get_layers(self) -> list:
        """Return the loaded model as a single layer for training assembly."""
        if self._model is None:
            return []
        return [self._model]

    def _apply_freeze(self, freeze: bool, trainable_layers: int) -> None:
        if freeze:
            for p in self._model.parameters():
                p.requires_grad = False
        else:
            for p in self._model.parameters():
                p.requires_grad = True
        if trainable_layers > 0:
            children = list(self._model.children())
            for child in children[-trainable_layers:]:
                for p in child.parameters():
                    p.requires_grad = True

    def _resolve_save_path(self, save_path: str, save_mode: str) -> str:
        """Return the actual path to save to, incrementing if needed."""
        if save_mode != "increment":
            return save_path
        from pathlib import Path
        p = Path(save_path)
        if not p.exists():
            return save_path
        stem, suffix = p.stem, p.suffix
        parent = p.parent
        i = 1
        while True:
            candidate = parent / f"{stem}_{i}{suffix}"
            if not candidate.exists():
                return str(candidate)
            i += 1

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import torch
        path             = inputs.get("path") or "model.pt"
        device           = inputs.get("device") or "cpu"
        freeze           = bool(inputs.get("freeze", False))
        trainable_layers = int(inputs.get("trainable_layers") or 0)
        save_path        = (inputs.get("save_path") or "").strip()
        save_mode        = (inputs.get("save_mode") or "overwrite").strip()
        tensor           = inputs.get("tensor_in")

        # Reload only when path changes
        if self._model is None or path != self._loaded_path:
            try:
                self._model = torch.load(path, map_location=device, weights_only=False)
                self._model.to(device)
                self._loaded_path = path
            except Exception as exc:
                self._model = None
                self._loaded_path = ""
                return {"tensor_out": None, "info": f"Load failed: {exc}"}

        self._apply_freeze(freeze, trainable_layers)

        total     = sum(p.numel() for p in self._model.parameters())
        trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        info = (
            f"{self._model.__class__.__name__} | {total:,} params | "
            f"trainable: {trainable:,} | frozen: {total - trainable:,}"
        )

        # Save if save_path is set
        if save_path:
            try:
                actual_path = self._resolve_save_path(save_path, save_mode)
                torch.save(self._model, actual_path)
                info += f"\nSaved -> {actual_path}"
            except Exception as exc:
                info += f"\nSave failed: {exc}"

        if tensor is None:
            return {"tensor_out": None, "info": info}

        try:
            if freeze and trainable_layers == 0:
                with torch.no_grad():
                    out = self._model(tensor.to(device))
            else:
                out = self._model(tensor.to(device))
            return {"tensor_out": out, "info": info}
        except Exception as exc:
            return {"tensor_out": None, "info": f"Forward failed: {exc}"}

    def export(self, iv, ov):
        path   = self._val(iv, "path")
        device = self._val(iv, "device")
        freeze = bool(self.inputs["freeze"].default_value)
        trainable_layers = int(self.inputs["trainable_layers"].default_value or 0)
        save_path = (self.inputs["save_path"].default_value or "").strip()
        tin    = iv.get("tensor_in")
        m_var  = f"_loaded_{self.id[:6]}"
        out_var  = ov.get("tensor_out", "_lm_out")
        info_var = ov.get("info",       "_lm_info")
        lines = [
            f"{m_var} = torch.load({path}, map_location={device}, weights_only=False)",
            f"{m_var}.to({device})",
        ]
        if freeze:
            lines.append(f"for _p in {m_var}.parameters(): _p.requires_grad = False")
        else:
            lines.append(f"for _p in {m_var}.parameters(): _p.requires_grad = True")
        if trainable_layers > 0:
            lines += [
                f"for _child in list({m_var}.children())[-{trainable_layers}:]:",
                f"    for _p in _child.parameters(): _p.requires_grad = True",
            ]
        lines += [
            f"_total     = sum(p.numel() for p in {m_var}.parameters())",
            f"_trainable = sum(p.numel() for p in {m_var}.parameters() if p.requires_grad)",
            f"{info_var} = f'{{{m_var}.__class__.__name__}} | {{_total:,}} params | "
            f"trainable: {{_trainable:,}} | frozen: {{_total - _trainable:,}}'",
        ]
        if save_path:
            lines += [
                f"torch.save({m_var}, {save_path!r})",
                f"{info_var} += '\\nSaved -> ' + {save_path!r}",
            ]
        if tin:
            lines.append(f"{out_var} = {m_var}({tin}.to({device}))")
        else:
            lines.append(f"{out_var} = None  # no tensor_in connected")
        return ["import torch"], lines
