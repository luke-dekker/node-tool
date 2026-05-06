"""Consolidated model save / load / export node — replaces SaveWeightsNode,
LoadWeightsNode, LoadModelNode, ExportONNXNode.

Pick `mode`:
  save_weights     — torch.save(model.state_dict(), path)
  save_checkpoint  — torch.save({model + optimizer + epoch + loss}, path)
  save_full        — torch.save(model, path)
  export_onnx      — torch.onnx.export(model, dummy_input, path)
  load_into        — load state_dict into a pre-built model
  load_checkpoint  — load checkpoint into pre-built model + optimizer
  load_full        — torch.load(path) → model (with optional surgery / freeze)

Outputs (each mode populates the relevant subset; others stay None / 0 / ""):
  model        — module passthrough (save) or loaded module (load)
  optimizer    — optimizer passthrough (save_checkpoint) or restored (load_checkpoint)
  path         — path that was written / loaded
  info         — human-readable summary
  epoch        — int (load_checkpoint)
  loss         — float (load_checkpoint)
  param_count  — int (load_full)
  trainable_count — int (load_full)
  layer_names  — comma-separated child names (load_full)
"""
from __future__ import annotations
import re
from typing import Any
from core.node import BaseNode, PortType


_SAVE_MODES = ["save_weights", "save_checkpoint", "save_full", "export_onnx"]
_LOAD_MODES = ["load_into", "load_checkpoint", "load_full"]
_MODES = _SAVE_MODES + _LOAD_MODES


def _parse_module_spec(spec: str):
    """One-line nn.Module spec like 'Linear(512, 10)' → instantiated module, or None."""
    import torch.nn as nn
    spec = (spec or "").strip()
    if not spec: return None
    m = re.match(r"^\s*(\w+)\s*\((.*)\)\s*$", spec)
    if not m: return None
    cls_name, args_str = m.group(1), m.group(2)
    cls = getattr(nn, cls_name, None)
    if cls is None: return None
    args: list = []
    for tok in args_str.split(","):
        tok = tok.strip()
        if not tok: continue
        try: args.append(int(tok)); continue
        except ValueError: pass
        try: args.append(float(tok)); continue
        except ValueError: pass
        args.append(tok.strip("'\""))
    try: return cls(*args)
    except Exception: return None


class ModelIONode(BaseNode):
    type_name   = "pt_model_io"
    label       = "Model I/O"
    category    = "Models"
    subcategory = "Save & Load"
    description = (
        "Save / load / export a model. Pick `mode`:\n"
        "  save_weights / save_checkpoint / save_full / export_onnx — write to path\n"
        "  load_into / load_checkpoint — load state into a pre-built model\n"
        "  load_full — torch.load full model (+ optional surgery, freeze)"
    )

    def __init__(self):
        self._loaded: Any = None
        self._loaded_cfg: tuple | None = None
        super().__init__()

    def get_layers(self) -> list:
        # When mode=load_full, the loaded model is the trainable layer set.
        return [self._loaded] if self._loaded is not None else []

    def relevant_inputs(self, values):
        mode = (values.get("mode") or "save_weights").strip()
        if mode == "save_weights":     return ["mode", "path"]                       # model wired
        if mode == "save_checkpoint":  return ["mode", "path", "epoch", "loss"]      # model+optim wired
        if mode == "save_full":        return ["mode", "path"]
        if mode == "export_onnx":      return ["mode", "path", "input_shape", "opset"]
        if mode == "load_into":        return ["mode", "path", "device"]             # model wired
        if mode == "load_checkpoint":  return ["mode", "path", "device"]
        if mode == "load_full":
            return ["mode", "path", "device", "freeze", "trainable_layers",
                    "remove_last_n", "replace_head", "eval_mode"]
        return ["mode"]

    def _setup_ports(self):
        self.add_input("mode",             PortType.STRING, "save_weights", choices=_MODES)
        self.add_input("model",            PortType.MODULE, default=None, optional=True)
        self.add_input("optimizer",        PortType.OPTIMIZER, default=None, optional=True)
        self.add_input("path",             PortType.STRING, "model.pt", optional=True)
        self.add_input("device",           PortType.STRING, "cpu",      optional=True)
        self.add_input("epoch",            PortType.INT,    0,          optional=True)
        self.add_input("loss",             PortType.FLOAT,  0.0,        optional=True)
        self.add_input("input_shape",      PortType.STRING, "1,784",    optional=True)
        self.add_input("opset",            PortType.INT,    17,         optional=True)
        self.add_input("freeze",           PortType.BOOL,   False,      optional=True)
        self.add_input("trainable_layers", PortType.INT,    0,          optional=True)
        self.add_input("remove_last_n",    PortType.INT,    0,          optional=True)
        self.add_input("replace_head",     PortType.STRING, "",         optional=True)
        self.add_input("eval_mode",        PortType.BOOL,   False,      optional=True)
        # outputs — each mode populates a subset
        self.add_output("model",           PortType.MODULE)
        self.add_output("optimizer",       PortType.OPTIMIZER)
        self.add_output("path",            PortType.STRING)
        self.add_output("info",            PortType.STRING)
        self.add_output("epoch",           PortType.INT)
        self.add_output("loss",            PortType.FLOAT)
        self.add_output("param_count",     PortType.INT)
        self.add_output("trainable_count", PortType.INT)
        self.add_output("layer_names",     PortType.STRING)

    def _empty(self):
        return {"model": None, "optimizer": None, "path": "", "info": "",
                "epoch": 0, "loss": 0.0, "param_count": 0, "trainable_count": 0,
                "layer_names": ""}

    def execute(self, inputs):
        import torch
        out  = self._empty()
        mode = (inputs.get("mode") or "save_weights").strip()
        path = inputs.get("path") or "model.pt"
        out["path"] = path

        try:
            if mode in _SAVE_MODES:
                model = inputs.get("model")
                if model is None:
                    out["info"] = "No model connected."
                    return out
                if mode == "save_weights":
                    torch.save(model.state_dict(), path)
                    out["model"] = model
                    out["info"] = f"Saved weights ({model.__class__.__name__}) -> {path}"
                elif mode == "save_checkpoint":
                    optimizer = inputs.get("optimizer")
                    ckpt = {
                        "epoch": inputs.get("epoch", 0),
                        "loss":  inputs.get("loss",  0.0),
                        "model_state_dict":     model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
                    }
                    torch.save(ckpt, path)
                    out["model"] = model; out["optimizer"] = optimizer
                    out["info"] = f"Saved checkpoint ({model.__class__.__name__}, epoch={ckpt['epoch']}) -> {path}"
                elif mode == "save_full":
                    torch.save(model, path)
                    total = sum(p.numel() for p in model.parameters())
                    out["model"] = model
                    out["info"] = f"Saved {model.__class__.__name__} ({total:,} params) -> {path}"
                elif mode == "export_onnx":
                    shape_str = inputs.get("input_shape") or "1,784"
                    opset = int(inputs.get("opset") or 17)
                    shape = tuple(int(s.strip()) for s in shape_str.split(","))
                    dummy = torch.zeros(*shape)
                    model.eval()
                    torch.onnx.export(model, dummy, path, opset_version=opset,
                                      input_names=["input"], output_names=["output"])
                    out["info"] = f"Exported ONNX (opset {opset}, input {shape}) -> {path}"
                return out

            # ── load modes ──────────────────────────────────────────────────
            device = inputs.get("device") or "cpu"

            if mode == "load_into":
                model = inputs.get("model")
                if model is None:
                    out["info"] = "load_into needs a pre-built model"; return out
                state = torch.load(path, map_location=device, weights_only=True)
                model.load_state_dict(state); model.to(device)
                out["model"] = model
                out["info"] = f"Loaded weights into {model.__class__.__name__}"
                return out

            if mode == "load_checkpoint":
                model = inputs.get("model"); optimizer = inputs.get("optimizer")
                ckpt = torch.load(path, map_location=device, weights_only=False)
                if model is not None and ckpt.get("model_state_dict") is not None:
                    model.load_state_dict(ckpt["model_state_dict"]); model.to(device)
                if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                out["model"] = model; out["optimizer"] = optimizer
                out["epoch"] = int(ckpt.get("epoch", 0)); out["loss"] = float(ckpt.get("loss", 0.0))
                out["info"] = f"Loaded checkpoint epoch={out['epoch']} loss={out['loss']:.4f}"
                return out

            if mode == "load_full":
                freeze        = bool(inputs.get("freeze", False))
                trainable_n   = int(inputs.get("trainable_layers") or 0)
                remove_last_n = int(inputs.get("remove_last_n") or 0)
                replace_head  = (inputs.get("replace_head") or "").strip()
                eval_mode     = bool(inputs.get("eval_mode", False))
                cfg = (path, remove_last_n, replace_head)
                if self._loaded is None or self._loaded_cfg != cfg:
                    self._loaded = torch.load(path, map_location=device, weights_only=False)
                    self._loaded.to(device)
                    self._loaded_cfg = cfg
                    # surgery
                    import torch.nn as nn
                    if remove_last_n > 0:
                        if isinstance(self._loaded, nn.Sequential):
                            self._loaded = nn.Sequential(*list(self._loaded.children())[:-remove_last_n])
                        else:
                            children = list(self._loaded.named_children())
                            for name, _ in children[-remove_last_n:]:
                                setattr(self._loaded, name, nn.Identity())
                    if replace_head:
                        new_head = _parse_module_spec(replace_head)
                        if new_head is not None:
                            children = list(self._loaded.named_children())
                            if children:
                                setattr(self._loaded, children[-1][0], new_head)
                # freeze
                for p in self._loaded.parameters():
                    p.requires_grad = not freeze
                if trainable_n > 0:
                    for child in list(self._loaded.children())[-trainable_n:]:
                        for p in child.parameters():
                            p.requires_grad = True
                if eval_mode: self._loaded.eval()
                total = sum(p.numel() for p in self._loaded.parameters())
                trainable = sum(p.numel() for p in self._loaded.parameters() if p.requires_grad)
                names = ", ".join(name for name, _ in self._loaded.named_children())
                out["model"] = self._loaded; out["param_count"] = total
                out["trainable_count"] = trainable; out["layer_names"] = names
                out["info"] = (f"{self._loaded.__class__.__name__} | {total:,} params | "
                               f"trainable: {trainable:,} | frozen: {total - trainable:,}")
                return out
            return out
        except Exception as exc:
            out["info"] = f"{mode} failed: {exc}"
            return out

    def export(self, iv, ov):
        # `iv` is a dict mapping port name → upstream variable name (or None
        # when unwired). `mode` is almost always a fixed config value rather
        # than a connected port, so read it from the instance default.
        mode = (self.inputs["mode"].default_value or "save_weights").strip()
        path = self._val(iv, "path")
        m_in = iv.get("model", "None")
        m_out = ov.get("model", "_mio_model")
        info  = ov.get("info",  "_mio_info")
        imp = ["import torch"]

        if mode == "save_weights":
            return imp, [f"torch.save({m_in}.state_dict(), {path})",
                         f"{m_out} = {m_in}", f"{info} = 'saved weights -> ' + {path}"]
        if mode == "save_full":
            return imp, [f"torch.save({m_in}, {path})",
                         f"{m_out} = {m_in}", f"{info} = 'saved model -> ' + {path}"]
        if mode == "save_checkpoint":
            opt   = iv.get("optimizer", "None")
            epoch = self._val(iv, "epoch"); loss = self._val(iv, "loss")
            return imp, [
                f"torch.save({{",
                f'    "epoch": {epoch}, "loss": {loss},',
                f'    "model_state_dict": {m_in}.state_dict(),',
                f'    "optimizer_state_dict": {opt}.state_dict() if {opt} is not None else None,',
                f"}}, {path})",
                f"{m_out} = {m_in}",
                f"{info} = 'saved checkpoint -> ' + {path}",
            ]
        if mode == "export_onnx":
            shape = self._val(iv, "input_shape"); opset = self._val(iv, "opset")
            return imp, [
                f"_shape = tuple(int(s.strip()) for s in {shape}.split(','))",
                f"_dummy = torch.zeros(*_shape)", f"{m_in}.eval()",
                f"torch.onnx.export({m_in}, _dummy, {path}, opset_version={opset}, "
                f"input_names=['input'], output_names=['output'])",
                f"{info} = 'exported onnx -> ' + {path}",
            ]
        if mode == "load_into":
            device = self._val(iv, "device")
            return imp, [
                f"_state = torch.load({path}, map_location={device}, weights_only=True)",
                f"{m_in}.load_state_dict(_state); {m_in}.to({device})",
                f"{m_out} = {m_in}", f"{info} = 'loaded weights into ' + {m_in}.__class__.__name__",
            ]
        if mode == "load_checkpoint":
            device = self._val(iv, "device"); opt = iv.get("optimizer", "None")
            return imp, [
                f"_ckpt = torch.load({path}, map_location={device}, weights_only=False)",
                f"if 'model_state_dict' in _ckpt: {m_in}.load_state_dict(_ckpt['model_state_dict'])",
                f"if {opt} is not None and 'optimizer_state_dict' in _ckpt:",
                f"    {opt}.load_state_dict(_ckpt['optimizer_state_dict'])",
                f"{m_out} = {m_in}; {info} = f\"loaded checkpoint epoch={{_ckpt.get('epoch', 0)}}\"",
            ]
        if mode == "load_full":
            device = self._val(iv, "device")
            return imp, [
                f"{m_out} = torch.load({path}, map_location={device}, weights_only=False).to({device})",
                f"{info} = '{m_out} loaded'",
            ]
        return [], [f"# unknown ModelIO mode {mode!r}"]
