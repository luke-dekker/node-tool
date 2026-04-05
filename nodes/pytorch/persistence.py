"""Save / load model nodes for PyTorch."""

from __future__ import annotations
from typing import Any
from core.node import BaseNode
from core.node import PortType


class SaveWeightsNode(BaseNode):
    """Save model state_dict (weights only) to a .pt file."""
    type_name   = "pt_save_weights"
    label       = "Save Weights"
    category    = "Models"
    subcategory = "Save & Load"
    description = "Save model.state_dict() to a file. Portable across architectures."

    def _setup_ports(self) -> None:
        self.add_input("model",  PortType.MODULE, default=None)
        self.add_input("path",   PortType.STRING, default="model_weights.pt")
        self.add_output("model", PortType.MODULE, description="Pass-through model")
        self.add_output("path",  PortType.STRING, description="Path that was saved to")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import torch
        model = inputs.get("model")
        path  = inputs.get("path") or "model_weights.pt"
        if model is not None:
            torch.save(model.state_dict(), path)
        return {"model": model, "path": path}


class LoadWeightsNode(BaseNode):
    """Load state_dict into an existing model architecture."""
    type_name   = "pt_load_weights"
    label       = "Load Weights"
    category    = "Models"
    subcategory = "Save & Load"
    description = (
        "Load saved weights into a model. "
        "The architecture must match the saved file."
    )

    def _setup_ports(self) -> None:
        self.add_input("model",  PortType.MODULE, default=None,
                       description="Model with matching architecture")
        self.add_input("path",   PortType.STRING, default="model_weights.pt")
        self.add_input("device", PortType.STRING, default="cpu",
                       description="Device to map weights to (cpu / cuda)")
        self.add_output("model", PortType.MODULE, description="Model with loaded weights")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import torch
        model  = inputs.get("model")
        path   = inputs.get("path") or "model_weights.pt"
        device = inputs.get("device") or "cpu"
        if model is not None:
            state = torch.load(path, map_location=device, weights_only=True)
            model.load_state_dict(state)
            model.to(device)
        return {"model": model}


class SaveCheckpointNode(BaseNode):
    """Save a full training checkpoint (model + optimizer + epoch)."""
    type_name   = "pt_save_checkpoint"
    label       = "Save Checkpoint"
    category    = "Models"
    subcategory = "Save & Load"
    description = (
        "Save model weights, optimizer state, and current epoch. "
        "Use LoadCheckpointNode to resume training exactly."
    )

    def _setup_ports(self) -> None:
        self.add_input("model",     PortType.MODULE,    default=None)
        self.add_input("optimizer", PortType.OPTIMIZER, default=None)
        self.add_input("epoch",     PortType.INT,       default=0)
        self.add_input("loss",      PortType.FLOAT,     default=0.0)
        self.add_input("path",      PortType.STRING,    default="checkpoint.pt")
        self.add_output("path",     PortType.STRING,    description="Path that was saved to")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import torch
        model     = inputs.get("model")
        optimizer = inputs.get("optimizer")
        path      = inputs.get("path") or "checkpoint.pt"
        ckpt: dict[str, Any] = {
            "epoch": inputs.get("epoch", 0),
            "loss":  inputs.get("loss",  0.0),
            "model_state_dict":     model.state_dict()     if model     is not None else None,
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        }
        torch.save(ckpt, path)
        return {"path": path}


class LoadCheckpointNode(BaseNode):
    """Load a training checkpoint — restore model, optimizer, and epoch."""
    type_name   = "pt_load_checkpoint"
    label       = "Load Checkpoint"
    category    = "Models"
    subcategory = "Save & Load"
    description = (
        "Restore model and optimizer from a checkpoint file. "
        "Resume training from the saved epoch."
    )

    def _setup_ports(self) -> None:
        self.add_input("model",      PortType.MODULE,    default=None,
                       description="Model with matching architecture")
        self.add_input("optimizer",  PortType.OPTIMIZER, default=None)
        self.add_input("path",       PortType.STRING,    default="checkpoint.pt")
        self.add_input("device",     PortType.STRING,    default="cpu")
        self.add_output("model",     PortType.MODULE,    description="Model with restored weights")
        self.add_output("optimizer", PortType.OPTIMIZER, description="Optimizer with restored state")
        self.add_output("epoch",     PortType.INT,       description="Epoch checkpoint was saved at")
        self.add_output("loss",      PortType.FLOAT,     description="Loss at checkpoint time")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import torch
        model     = inputs.get("model")
        optimizer = inputs.get("optimizer")
        path      = inputs.get("path") or "checkpoint.pt"
        device    = inputs.get("device") or "cpu"
        ckpt = torch.load(path, map_location=device, weights_only=False)
        if model is not None and "model_state_dict" in ckpt and ckpt["model_state_dict"] is not None:
            model.load_state_dict(ckpt["model_state_dict"])
            model.to(device)
        if optimizer is not None and "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return {
            "model":     model,
            "optimizer": optimizer,
            "epoch":     ckpt.get("epoch", 0),
            "loss":      ckpt.get("loss",  0.0),
        }


class ExportONNXNode(BaseNode):
    """Export a model to ONNX format for deployment."""
    type_name   = "pt_export_onnx"
    label       = "Export ONNX"
    category    = "Models"
    subcategory = "Save & Load"
    description = (
        "Export model to ONNX for deployment on hardware / inference engines. "
        "Provide a dummy input matching your model's expected input shape."
    )

    def _setup_ports(self) -> None:
        self.add_input("model",       PortType.MODULE, default=None)
        self.add_input("input_shape", PortType.STRING, default="1,784",
                       description="Dummy input shape, e.g. 1,3,224,224")
        self.add_input("path",        PortType.STRING, default="model.onnx")
        self.add_input("opset",       PortType.INT,    default=17,
                       description="ONNX opset version")
        self.add_output("path",       PortType.STRING, description="Path exported to")
        self.add_output("info",       PortType.STRING, description="Export summary")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import torch
        model     = inputs.get("model")
        path      = inputs.get("path") or "model.onnx"
        opset     = int(inputs.get("opset") or 17)
        shape_str = inputs.get("input_shape") or "1,784"
        if model is None:
            return {"path": path, "info": "No model connected."}
        try:
            shape = tuple(int(s.strip()) for s in shape_str.split(","))
            dummy = torch.zeros(*shape)
            model.eval()
            torch.onnx.export(
                model, dummy, path,
                opset_version=opset,
                input_names=["input"],
                output_names=["output"],
            )
            info = f"Exported to {path} (opset {opset}, input {shape})"
        except Exception as exc:
            info = f"Export failed: {exc}"
        return {"path": path, "info": info}


class SaveFullModelNode(BaseNode):
    """Save the entire model object (architecture + weights) to a .pt file."""
    type_name   = "pt_save_full_model"
    label       = "Save Full Model"
    category    = "Models"
    subcategory = "Save & Load"
    description = (
        "torch.save(model, path) — saves architecture AND weights together. "
        "Load back with Pretrained Block on any canvas, no graph rebuilding needed."
    )

    def _setup_ports(self) -> None:
        self.add_input("model",  PortType.MODULE, default=None)
        self.add_input("path",   PortType.STRING, default="model_full.pt")
        self.add_output("model", PortType.MODULE, description="Pass-through model")
        self.add_output("info",  PortType.STRING, description="Save confirmation")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import torch
        model = inputs.get("model")
        path  = inputs.get("path") or "model_full.pt"
        if model is None:
            return {"model": None, "info": "No model connected."}
        try:
            torch.save(model, path)
            total = sum(p.numel() for p in model.parameters())
            info  = f"Saved {model.__class__.__name__} ({total:,} params) → {path}"
        except Exception as exc:
            info = f"Save failed: {exc}"
        return {"model": model, "info": info}


class PretrainedBlockNode(BaseNode):
    """Load a full saved model as a drop-in block with freeze controls."""
    type_name   = "pt_pretrained_block"
    label       = "Pretrained Block"
    category    = "Models"
    subcategory = "Pretrained"
    description = (
        "Load a model saved with Save Full Model. "
        "Freeze all or only the first N layers for transfer learning / fine-tuning."
    )

    def _setup_ports(self) -> None:
        self.add_input("path",             PortType.STRING, default="model_full.pt",
                       description="Path to a .pt file saved with Save Full Model")
        self.add_input("device",           PortType.STRING, default="cpu")
        self.add_input("freeze_all",       PortType.BOOL,   default=False,
                       description="Freeze every parameter (feature extractor mode)")
        self.add_input("trainable_layers", PortType.INT,    default=0,
                       description="Unfreeze this many layers from the end (0 = respect freeze_all)")
        self.add_input("eval_mode",        PortType.BOOL,   default=False,
                       description="Force model.eval() — disables dropout / batchnorm training noise")
        self.add_output("model", PortType.MODULE)
        self.add_output("info",  PortType.STRING,
                        description="Class, param count, frozen/trainable summary")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import torch
        path             = inputs.get("path") or "model_full.pt"
        device           = inputs.get("device") or "cpu"
        freeze_all       = bool(inputs.get("freeze_all", False))
        trainable_layers = int(inputs.get("trainable_layers") or 0)
        eval_mode        = bool(inputs.get("eval_mode", False))

        try:
            model = torch.load(path, map_location=device, weights_only=False)
            model.to(device)
        except Exception as exc:
            return {"model": None, "info": f"Load failed: {exc}"}

        # Apply freezing
        if freeze_all:
            for p in model.parameters():
                p.requires_grad = False

        # Unfreeze the last N layers (by named children, then parameters)
        if trainable_layers > 0:
            children = list(model.children())
            for child in children[-trainable_layers:]:
                for p in child.parameters():
                    p.requires_grad = True

        if eval_mode:
            model.eval()

        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen    = total - trainable
        mode_str  = "eval" if eval_mode else "train"
        info = (
            f"{model.__class__.__name__} loaded from {path}\n"
            f"Device: {device} | Mode: {mode_str}\n"
            f"Total:     {total:,} params\n"
            f"Trainable: {trainable:,} params\n"
            f"Frozen:    {frozen:,} params"
        )
        return {"model": model, "info": info}


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


class ModelInfoNode(BaseNode):
    """Display model architecture summary and parameter count."""
    type_name   = "pt_model_info_persist"
    label       = "Model Info"
    category    = "Analyze"
    subcategory = "Models"
    description = "Print model architecture and total parameter count."

    def _setup_ports(self) -> None:
        self.add_input("model",  PortType.MODULE, default=None)
        self.add_output("model", PortType.MODULE, description="Pass-through model")
        self.add_output("info",  PortType.STRING, description="Architecture summary")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        model = inputs.get("model")
        if model is None:
            return {"model": None, "info": "No model connected."}
        total     = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        lines = [
            str(model),
            f"\nTotal params:     {total:,}",
            f"Trainable params: {trainable:,}",
            f"Frozen params:    {total - trainable:,}",
        ]
        return {"model": model, "info": "\n".join(lines)}
