"""Add Branches node."""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


class AddBranchesNode(BaseNode):
    type_name   = "pt_add_branches"
    label       = "Add Branches"
    category    = "Models"
    subcategory = "Architecture"
    description = "branch_1(x) + branch_2(x). For parallel paths summed together."

    def _setup_ports(self) -> None:
        self.add_input("branch_1", PortType.MODULE, default=None)
        self.add_input("branch_2", PortType.MODULE, default=None)
        self.add_output("model",   PortType.MODULE)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        b1 = inputs.get("branch_1")
        b2 = inputs.get("branch_2")
        if b1 is None or b2 is None:
            return {"model": None}
        try:
            import torch.nn as nn
            _b1, _b2 = b1, b2

            class _AddMod(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.branch_1 = _b1
                    self.branch_2 = _b2

                def forward(self, x):
                    return self.branch_1(x) + self.branch_2(x)

            return {"model": _AddMod()}
        except Exception:
            return {"model": None}

    def export(self, iv, ov):
        b1  = iv.get("branch_1", "None")
        b2  = iv.get("branch_2", "None")
        out = ov.get("model", "_add_mod")
        lines = [
            f"class _AddMod_{out}(nn.Module):",
            f"    def __init__(self):",
            f"        super().__init__()",
            f"        self.branch_1 = {b1}",
            f"        self.branch_2 = {b2}",
            f"    def forward(self, x):",
            f"        return self.branch_1(x) + self.branch_2(x)",
            f"{out} = _AddMod_{out}()",
        ]
        return ["import torch.nn as nn"], lines
