"""PreviewNode — display any value in the inspector + terminal.

Pass-through node for inspecting intermediate values without rewiring.
Differs from PrintNode in that PreviewNode produces a richer summary string
including type and shape info for tensors / arrays / dataframes, while
PrintNode just calls str(). Useful as a "tap" you can drop on any wire to
see what's flowing without breaking the chain.

Output is the input verbatim, so you can wire PreviewNode in the middle of
a chain without changing behavior.
"""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


def _summary(v: Any) -> str:
    if v is None:
        return "None"
    if isinstance(v, bool):
        return f"bool {v}"
    if isinstance(v, int):
        return f"int {v}"
    if isinstance(v, float):
        return f"float {v:.6g}"
    if isinstance(v, str):
        return f"str[{len(v)}] {v[:60]!r}{'...' if len(v) > 60 else ''}"
    # Tensors / arrays / dataframes — duck-type by attribute
    shape = getattr(v, "shape", None)
    if shape is not None:
        dtype = getattr(v, "dtype", "")
        return f"{type(v).__name__} shape={tuple(shape)} dtype={dtype}"
    if isinstance(v, (list, tuple)):
        return f"{type(v).__name__}[{len(v)}]"
    if isinstance(v, dict):
        keys = ", ".join(repr(k) for k in list(v.keys())[:6])
        suffix = "..." if len(v) > 6 else ""
        return f"dict[{len(v)}] {{{keys}{suffix}}}"
    return f"{type(v).__name__}"


class PreviewNode(BaseNode):
    type_name   = "preview"
    label       = "Preview"
    category    = "Python"
    subcategory = "Data"
    description = (
        "Tap any wire to see a rich summary of the value flowing through. "
        "Output is the input verbatim — wire it inline anywhere without "
        "breaking the chain."
    )

    def _setup_ports(self) -> None:
        self.add_input("Value", PortType.ANY, default=None)
        self.add_input("Label", PortType.STRING, default="")
        self.add_output("Value",   PortType.ANY)
        self.add_output("Summary", PortType.STRING)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        v = inputs.get("Value")
        label = str(inputs.get("Label") or "")
        summary = _summary(v)
        terminal_line = f"{label}: {summary}" if label else summary
        return {"Value": v, "Summary": summary, "__terminal__": terminal_line}

    def export(self, iv, ov):
        v   = iv.get("Value") or "None"
        lab = self._val(iv, "Label")
        out_v = ov.get("Value",   "_preview_v")
        out_s = ov.get("Summary", "_preview_s")
        return [], [
            f"{out_v} = {v}",
            f"{out_s} = repr({out_v})  # PreviewNode summary",
            f"print(f'{{{lab}}}: {{{out_s}}}' if {lab} else {out_s})",
        ]
