"""Shared helpers for PyTorch layer node export."""

from __future__ import annotations
import torch
import torch.nn as nn


def _make_activation(name: str) -> nn.Module | None:
    key = (name or "").strip().lower().replace(" ", "").replace("_", "")
    return {
        "relu":      nn.ReLU(),
        "leakyrelu": nn.LeakyReLU(0.01),
        "sigmoid":   nn.Sigmoid(),
        "tanh":      nn.Tanh(),
        "gelu":      nn.GELU(),
        "elu":       nn.ELU(),
        "silu":      nn.SiLU(),
        "swish":     nn.SiLU(),
        "softmax":   nn.Softmax(dim=1),
    }.get(key)


def _forward(layer: nn.Module, act: nn.Module | None, tensor: torch.Tensor | None):
    """Run tensor through layer (+ optional activation). Returns None if no input.

    Honors the surrounding grad context: live preview callers wrap in torch.no_grad(),
    training callers (GraphAsModule) leave grad enabled.
    """
    if tensor is None:
        return None
    try:
        out = layer(tensor)
        if act is not None:
            out = act(out)
        return out
    except Exception:
        return None


def _act_expr(act_name: str) -> str | None:
    """Returns an nn.Module instance expression for use inside the live runtime."""
    key = (act_name or "").strip().lower().replace(" ", "").replace("_", "")
    return {
        "relu":      "nn.ReLU()",
        "leakyrelu": "nn.LeakyReLU(0.01)",
        "sigmoid":   "nn.Sigmoid()",
        "tanh":      "nn.Tanh()",
        "gelu":      "nn.GELU()",
        "elu":       "nn.ELU()",
        "silu":      "nn.SiLU()",
        "swish":     "nn.SiLU()",
        "softmax":   "nn.Softmax(dim=1)",
    }.get(key)


def _act_func(act_name: str) -> str | None:
    """Returns the functional form of an activation for clean exported code.

    Functional form keeps the exported script gradient-safe (no nn.Module
    instantiation per forward) and matches the user's "graph IS the model"
    principle: each line is one explicit operation.
    """
    key = (act_name or "").strip().lower().replace(" ", "").replace("_", "")
    return {
        "relu":      "F.relu",
        "leakyrelu": "F.leaky_relu",
        "sigmoid":   "torch.sigmoid",
        "tanh":      "torch.tanh",
        "gelu":      "F.gelu",
        "elu":       "F.elu",
        "silu":      "F.silu",
        "swish":     "F.silu",
        "softmax":   "lambda __x: F.softmax(__x, dim=1)",
    }.get(key)


_BASE_LAYER_IMPORTS = ["import torch", "import torch.nn as nn"]
_F_IMPORT = "import torch.nn.functional as F"


def _layer_fwd(ov, iv, layer_var: str, layer_expr: str,
               act_name: str = "") -> tuple[list[str], list[str]]:
    """Generate: instantiate layer, apply to tensor_in, store in tensor_out.

    Returns (imports, lines). Output is a flat sequence of explicit lines —
    no Sequential, no no_grad, no helper functions. Mirrors how the live graph
    executes node-by-node, so the exported script is faithful to the visual flow.
    """
    tin  = iv.get("tensor_in") or "_x"
    tout = ov.get("tensor_out", "_out")
    func = _act_func(act_name)
    lines = [
        f"{layer_var} = {layer_expr}",
        f"{tout} = {layer_var}({tin})",
    ]
    imports = list(_BASE_LAYER_IMPORTS)
    if func:
        lines.append(f"{tout} = {func}({tout})")
        imports.append(_F_IMPORT)
    return imports, lines
