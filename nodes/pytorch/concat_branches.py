"""Concat Branches node."""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


class ConcatBranchesNode(BaseNode):
    type_name   = "pt_concat_branches"
    label       = "Concat Branches"
    category    = "Models"
    subcategory = "Architecture"
    description = (
        "torch.cat([branch_1(x), branch_2(x), ...], dim=dim). "
        "Useful for Inception-style parallel paths."
    )

    def _setup_ports(self) -> None:
        self.add_input("branch_1", PortType.MODULE, default=None)
        self.add_input("branch_2", PortType.MODULE, default=None)
        self.add_input("branch_3", PortType.MODULE, default=None)
        self.add_input("branch_4", PortType.MODULE, default=None)
        self.add_input("dim",      PortType.INT,    default=1,
                       description="Concat dim (1 = channel axis for images)")
        self.add_output("model",   PortType.MODULE)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        branches = [inputs.get(f"branch_{i}") for i in range(1, 5)
                    if inputs.get(f"branch_{i}") is not None]
        if not branches:
            return {"model": None}
        try:
            import torch
            import torch.nn as nn
            _branches = branches
            _dim = int(inputs.get("dim") or 1)

            class _ConcatMod(nn.Module):
                def __init__(self):
                    super().__init__()
                    for i, b in enumerate(_branches):
                        setattr(self, f"branch_{i}", b)
                    self._n   = len(_branches)
                    self._dim = _dim

                def forward(self, x):
                    return torch.cat(
                        [getattr(self, f"branch_{i}")(x) for i in range(self._n)],
                        dim=self._dim,
                    )

            return {"model": _ConcatMod()}
        except Exception:
            return {"model": None}

    def export(self, iv, ov):
        branches = [iv.get(f"branch_{i}") for i in range(1, 5) if iv.get(f"branch_{i}")]
        dim  = self._val(iv, "dim")
        out  = ov.get("model", "_concat_mod")
        branch_attrs = "\n".join(f"        self.branch_{i} = {b}" for i, b in enumerate(branches))
        branch_calls = ", ".join(f"self.branch_{i}(x)" for i in range(len(branches)))
        lines = [
            f"class _ConcatMod_{out}(nn.Module):",
            f"    def __init__(self):",
            f"        super().__init__()",
            branch_attrs,
            f"    def forward(self, x):",
            f"        return torch.cat([{branch_calls}], dim={dim})",
            f"{out} = _ConcatMod_{out}()",
        ]
        return ["import torch", "import torch.nn as nn"], lines
