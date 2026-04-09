"""Load Model node."""
from __future__ import annotations
import re
from typing import Any
from core.node import BaseNode, PortType


def _parse_module_spec(spec: str):
    """Parse a one-line module spec like 'Linear(512, 10)' into the actual nn.Module.

    Supported arg types: int, float, bare strings (treated as positional kwargs are not supported).
    Returns None if the spec can't be parsed — caller should treat that as 'no replacement'.
    """
    import torch.nn as nn
    spec = (spec or "").strip()
    if not spec:
        return None
    m = re.match(r"^\s*(\w+)\s*\((.*)\)\s*$", spec)
    if not m:
        return None
    cls_name, args_str = m.group(1), m.group(2)
    cls = getattr(nn, cls_name, None)
    if cls is None:
        return None
    args: list = []
    for tok in args_str.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            args.append(int(tok))
            continue
        except ValueError:
            pass
        try:
            args.append(float(tok))
            continue
        except ValueError:
            pass
        args.append(tok.strip("'\""))
    try:
        return cls(*args)
    except Exception:
        return None


def _shape_str(t) -> str:
    """Return a tensor shape as a tidy '(B, 3, 224, 224)' style string."""
    try:
        return f"({', '.join(str(d) for d in t.shape)})"
    except Exception:
        return ""


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
        # NEW: surgery options
        self.add_input("remove_last_n",    PortType.INT,    default=0,
                       description="Remove the last N top-level children (decapitate). "
                                   "Works best on Sequential-shaped models.")
        self.add_input("replace_head",     PortType.STRING, default="",
                       description="Replace the last child with a new module spec. "
                                   "Examples: 'Linear(512, 10)', 'Conv2d(64, 1, 1)'.")
        # Save side
        self.add_input("save_path",        PortType.STRING, default="",
                       description="If set, save model here on every graph run")
        self.add_input("save_mode",        PortType.STRING, default="overwrite",
                       choices=["overwrite", "increment"],
                       description="overwrite: replace same file  |  increment: model_1.pt, model_2.pt ...")
        # Outputs
        self.add_output("tensor_out",      PortType.TENSOR)
        self.add_output("model",           PortType.MODULE,
                        description="The loaded (and possibly modified) module — wire into Adam etc.")
        self.add_output("info",            PortType.STRING,
                        description="Model class, param count, frozen/trainable summary")
        # NEW: introspection ports
        self.add_output("input_shape",     PortType.STRING,
                        description="Shape of last tensor_in seen (after first forward pass)")
        self.add_output("output_shape",    PortType.STRING,
                        description="Shape of last tensor_out (after first forward pass)")
        self.add_output("param_count",     PortType.INT,
                        description="Total parameter count")
        self.add_output("trainable_count", PortType.INT,
                        description="Trainable parameter count")
        self.add_output("layer_names",     PortType.STRING,
                        description="Comma-separated list of top-level child layer names")

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

    def _apply_surgery(self, remove_last_n: int, replace_head_spec: str) -> str:
        """Apply structural edits (remove_last_n, replace_head) to self._model.

        Returns a short status string for the info line. Both ops are best-effort:
        if they can't be applied (model isn't Sequential-shaped, spec is junk),
        the model is left as-is and the status line says so.
        """
        import torch.nn as nn
        notes: list[str] = []

        if remove_last_n > 0:
            try:
                # Sequential models support clean slicing
                if isinstance(self._model, nn.Sequential):
                    self._model = nn.Sequential(*list(self._model.children())[:-remove_last_n])
                    notes.append(f"removed last {remove_last_n}")
                else:
                    # For non-Sequential models, drop the last N top-level children by name
                    children = list(self._model.named_children())
                    if remove_last_n >= len(children):
                        notes.append(f"remove_last_n={remove_last_n} >= num children, skipped")
                    else:
                        for name, _ in children[-remove_last_n:]:
                            setattr(self._model, name, nn.Identity())
                        notes.append(f"replaced last {remove_last_n} with Identity")
            except Exception as exc:
                notes.append(f"remove_last_n failed: {exc}")

        if replace_head_spec:
            new_head = _parse_module_spec(replace_head_spec)
            if new_head is None:
                notes.append(f"replace_head spec not parseable: {replace_head_spec!r}")
            else:
                try:
                    children = list(self._model.named_children())
                    if not children:
                        notes.append("replace_head: model has no children")
                    else:
                        last_name = children[-1][0]
                        setattr(self._model, last_name, new_head)
                        notes.append(f"replaced {last_name} -> {replace_head_spec}")
                except Exception as exc:
                    notes.append(f"replace_head failed: {exc}")

        return "; ".join(notes)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import torch
        path             = inputs.get("path") or "model.pt"
        device           = inputs.get("device") or "cpu"
        freeze           = bool(inputs.get("freeze", False))
        trainable_layers = int(inputs.get("trainable_layers") or 0)
        remove_last_n    = int(inputs.get("remove_last_n") or 0)
        replace_head     = (inputs.get("replace_head") or "").strip()
        save_path        = (inputs.get("save_path") or "").strip()
        save_mode        = (inputs.get("save_mode") or "overwrite").strip()
        tensor           = inputs.get("tensor_in")

        empty = {
            "tensor_out": None, "model": None, "info": "",
            "input_shape": "", "output_shape": "",
            "param_count": 0, "trainable_count": 0, "layer_names": "",
        }

        # Reload only when path or surgery options change
        cfg_signature = (path, remove_last_n, replace_head)
        if self._model is None or cfg_signature != getattr(self, "_loaded_cfg", None):
            try:
                self._model = torch.load(path, map_location=device, weights_only=False)
                self._model.to(device)
                self._loaded_path = path
                self._loaded_cfg = cfg_signature
                # Apply structural edits BEFORE collecting layer info
                self._surgery_notes = self._apply_surgery(remove_last_n, replace_head)
            except Exception as exc:
                self._model = None
                self._loaded_path = ""
                self._loaded_cfg = None
                return {**empty, "info": f"Load failed: {exc}"}

        self._apply_freeze(freeze, trainable_layers)

        # Collect introspection data
        total       = sum(p.numel() for p in self._model.parameters())
        trainable   = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        layer_names = ", ".join(name for name, _ in self._model.named_children())

        info = (
            f"{self._model.__class__.__name__} | {total:,} params | "
            f"trainable: {trainable:,} | frozen: {total - trainable:,}"
        )
        if getattr(self, "_surgery_notes", ""):
            info += f"\nSurgery: {self._surgery_notes}"

        # Save if save_path is set
        if save_path:
            try:
                actual_path = self._resolve_save_path(save_path, save_mode)
                torch.save(self._model, actual_path)
                info += f"\nSaved -> {actual_path}"
            except Exception as exc:
                info += f"\nSave failed: {exc}"

        out_dict = {
            **empty,
            "model": self._model,
            "info": info,
            "param_count": total,
            "trainable_count": trainable,
            "layer_names": layer_names,
        }

        if tensor is None:
            return out_dict

        try:
            tensor = tensor.to(device)
            if freeze and trainable_layers == 0:
                with torch.no_grad():
                    out = self._model(tensor)
            else:
                out = self._model(tensor)
            out_dict["tensor_out"]   = out
            out_dict["input_shape"]  = _shape_str(tensor)
            out_dict["output_shape"] = _shape_str(out)
            return out_dict
        except Exception as exc:
            out_dict["info"] = f"{info}\nForward failed: {exc}"
            return out_dict

    def export(self, iv, ov):
        path   = self._val(iv, "path")
        device = self._val(iv, "device")
        freeze = bool(self.inputs["freeze"].default_value)
        trainable_layers = int(self.inputs["trainable_layers"].default_value or 0)
        remove_last_n    = int(self.inputs["remove_last_n"].default_value or 0)
        replace_head     = (self.inputs["replace_head"].default_value or "").strip()
        save_path = (self.inputs["save_path"].default_value or "").strip()
        tin    = iv.get("tensor_in")
        m_var  = f"_loaded_{self.safe_id}"
        out_var      = ov.get("tensor_out",      "_lm_out")
        model_var    = ov.get("model",           "_lm_model")
        info_var     = ov.get("info",            "_lm_info")
        in_shape_var = ov.get("input_shape",     "_lm_in_shape")
        out_shape_var= ov.get("output_shape",    "_lm_out_shape")
        param_var    = ov.get("param_count",     "_lm_params")
        train_var    = ov.get("trainable_count", "_lm_trainable")
        names_var    = ov.get("layer_names",     "_lm_names")

        lines = [
            f"{model_var} = torch.load({path}, map_location={device}, weights_only=False)",
            f"{model_var}.to({device})",
        ]
        # Surgery: remove_last_n
        if remove_last_n > 0:
            lines += [
                f"# Remove last {remove_last_n} layers (works best on Sequential models)",
                f"if isinstance({model_var}, nn.Sequential):",
                f"    {model_var} = nn.Sequential(*list({model_var}.children())[:-{remove_last_n}])",
                f"else:",
                f"    for _name, _ in list({model_var}.named_children())[-{remove_last_n}:]:",
                f"        setattr({model_var}, _name, nn.Identity())",
            ]
        # Surgery: replace_head
        if replace_head:
            new_head = _parse_module_spec(replace_head)
            if new_head is not None:
                lines += [
                    f"# Replace last child with {replace_head}",
                    f"_children = list({model_var}.named_children())",
                    f"if _children:",
                    f"    setattr({model_var}, _children[-1][0], nn.{replace_head})",
                ]
        # Freeze
        if freeze:
            lines.append(f"for _p in {model_var}.parameters(): _p.requires_grad = False")
        else:
            lines.append(f"for _p in {model_var}.parameters(): _p.requires_grad = True")
        if trainable_layers > 0:
            lines += [
                f"for _child in list({model_var}.children())[-{trainable_layers}:]:",
                f"    for _p in _child.parameters(): _p.requires_grad = True",
            ]
        # Introspection
        lines += [
            f"{param_var}  = sum(p.numel() for p in {model_var}.parameters())",
            f"{train_var}  = sum(p.numel() for p in {model_var}.parameters() if p.requires_grad)",
            f"{names_var}  = ', '.join(name for name, _ in {model_var}.named_children())",
            f"{info_var}   = f'{{{model_var}.__class__.__name__}} | {{{param_var}:,}} params | "
            f"trainable: {{{train_var}:,}} | frozen: {{{param_var} - {train_var}:,}}'",
        ]
        if save_path:
            lines += [
                f"torch.save({model_var}, {save_path!r})",
                f"{info_var} += '\\nSaved -> ' + {save_path!r}",
            ]
        if tin:
            lines += [
                f"{in_shape_var}  = f'({{\", \".join(str(d) for d in {tin}.shape)}})'",
                f"{out_var}       = {model_var}({tin}.to({device}))",
                f"{out_shape_var} = f'({{\", \".join(str(d) for d in {out_var}.shape)}})'",
            ]
        else:
            lines += [
                f"{out_var}       = None  # no tensor_in connected",
                f"{in_shape_var}  = ''",
                f"{out_shape_var} = ''",
            ]
        return ["import torch", "import torch.nn as nn"], lines
