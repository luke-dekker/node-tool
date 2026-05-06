"""Consolidated branch combiner — replaces AddBranchesNode + ConcatBranchesNode.

Pick `op`:
  add    — branch_1(x) + branch_2(x)  (Inception-style residual / parallel paths)
  concat — torch.cat([branch_i(x), ...], dim) (channel-axis concat by default)
"""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


_OPS = ["add", "concat"]


class BranchOpNode(BaseNode):
    type_name   = "pt_branch_op"
    label       = "Branch Op"
    category    = "Models"
    subcategory = "Architecture"
    description = (
        "Combine the outputs of up to 4 parallel branches into a single module.\n"
        "  add    — branch_1(x) + branch_2(x) (sum)\n"
        "  concat — torch.cat([branch_i(x) for ...], dim)"
    )

    def relevant_inputs(self, values):
        op = (values.get("op") or "add").strip()
        if op == "concat": return ["op", "dim"]   # branches are wired
        return ["op"]

    def _setup_ports(self) -> None:
        self.add_input("branch_1", PortType.MODULE, default=None)
        self.add_input("branch_2", PortType.MODULE, default=None)
        self.add_input("branch_3", PortType.MODULE, default=None, optional=True)
        self.add_input("branch_4", PortType.MODULE, default=None, optional=True)
        self.add_input("op",       PortType.STRING, "add", choices=_OPS)
        self.add_input("dim",      PortType.INT,    1, optional=True,
                       description="concat: dim to concat over (1 = channel for images)")
        self.add_output("model", PortType.MODULE)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        try:
            import torch
            import torch.nn as nn
            op = (inputs.get("op") or "add").strip()
            branches = [inputs.get(f"branch_{i}") for i in range(1, 5)
                        if inputs.get(f"branch_{i}") is not None]
            if not branches:
                return {"model": None}

            if op == "add":
                if len(branches) < 2:
                    return {"model": None}
                _b1, _b2 = branches[0], branches[1]
                class _AddMod(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.branch_1 = _b1; self.branch_2 = _b2
                    def forward(self, x):
                        return self.branch_1(x) + self.branch_2(x)
                return {"model": _AddMod()}

            if op == "concat":
                _bs  = branches
                _dim = int(inputs.get("dim") or 1)
                class _ConcatMod(nn.Module):
                    def __init__(self):
                        super().__init__()
                        for i, b in enumerate(_bs):
                            setattr(self, f"branch_{i}", b)
                        self._n = len(_bs); self._dim = _dim
                    def forward(self, x):
                        return torch.cat(
                            [getattr(self, f"branch_{i}")(x) for i in range(self._n)],
                            dim=self._dim,
                        )
                return {"model": _ConcatMod()}
            return {"model": None}
        except Exception:
            return {"model": None}

    def export(self, iv, ov):
        op = (self.inputs["op"].default_value or "add")
        branches = [iv.get(f"branch_{i}") for i in range(1, 5) if iv.get(f"branch_{i}")]
        out = ov.get("model", "_branch_mod")
        if op == "add":
            if len(branches) < 2:
                return ["import torch.nn as nn"], [f"{out} = None  # add needs >=2 branches"]
            b1, b2 = branches[0], branches[1]
            return ["import torch.nn as nn"], [
                f"class _AddMod_{out}(nn.Module):",
                f"    def __init__(self):",
                f"        super().__init__()",
                f"        self.branch_1 = {b1}", f"        self.branch_2 = {b2}",
                f"    def forward(self, x):",
                f"        return self.branch_1(x) + self.branch_2(x)",
                f"{out} = _AddMod_{out}()",
            ]
        if op == "concat":
            dim = self._val(iv, "dim")
            attrs = "\n".join(f"        self.branch_{i} = {b}" for i, b in enumerate(branches))
            calls = ", ".join(f"self.branch_{i}(x)" for i in range(len(branches)))
            return ["import torch", "import torch.nn as nn"], [
                f"class _ConcatMod_{out}(nn.Module):",
                f"    def __init__(self):",
                f"        super().__init__()",
                attrs,
                f"    def forward(self, x):",
                f"        return torch.cat([{calls}], dim={dim})",
                f"{out} = _ConcatMod_{out}()",
            ]
        return [], [f"# unknown branch op {op!r}"]
