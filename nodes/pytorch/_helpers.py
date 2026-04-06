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
    """Run tensor through layer (+ optional activation). Returns None if no input."""
    if tensor is None:
        return None
    try:
        with torch.no_grad():
            out = layer(tensor)
            if act is not None:
                out = act(out)
        return out
    except Exception:
        return None


def _act_expr(act_name: str) -> str | None:
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


def _layer_fwd(ov, iv, layer_var: str, layer_expr: str, act_name: str = "") -> list[str]:
    """Generate: instantiate layer, apply to tensor_in, store in tensor_out."""
    tin  = iv.get("tensor_in") or "_x"
    tout = ov.get("tensor_out", "_out")
    act  = _act_expr(act_name)
    lines = [f"{layer_var} = {layer_expr}"]
    if act:
        lines += [
            f"with torch.no_grad():",
            f"    _tmp = {layer_var}({tin})",
            f"    {tout} = {act}(_tmp)",
        ]
    else:
        lines.append(f"{tout} = {layer_var}({tin})")
    return lines
