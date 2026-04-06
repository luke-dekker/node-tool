"""Residual Block node."""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


class ResidualBlockNode(BaseNode):
    type_name   = "pt_residual_block"
    label       = "Residual Block"
    category    = "Models"
    subcategory = "Architecture"
    description = (
        "x + block(x). Wire an optional projection module to match dimensions "
        "(e.g. a 1x1 Conv when channels change)."
    )

    def _setup_ports(self) -> None:
        self.add_input("block",      PortType.MODULE, default=None,
                       description="F(x) — the residual path")
        self.add_input("projection", PortType.MODULE, default=None,
                       description="Optional shortcut projection (1x1 conv / linear)")
        self.add_output("model", PortType.MODULE)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        block = inputs.get("block")
        proj  = inputs.get("projection")
        if block is None:
            return {"model": None}
        try:
            import torch.nn as nn
            _block, _proj = block, proj

            class _ResBlock(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.block = _block
                    self.projection = _proj

                def forward(self, x):
                    identity = x if self.projection is None else self.projection(x)
                    return identity + self.block(x)

            return {"model": _ResBlock()}
        except Exception:
            return {"model": None}

    def export(self, iv, ov):
        block = iv.get("block", "None")
        proj  = iv.get("projection", "None")
        out   = ov.get("model", "_res_block")
        lines = [
            f"class _ResBlock_{out}(nn.Module):",
            f"    def __init__(self):",
            f"        super().__init__()",
            f"        self.block = {block}",
            f"        self.projection = {proj}",
            f"    def forward(self, x):",
            f"        identity = x if self.projection is None else self.projection(x)",
            f"        return identity + self.block(x)",
            f"{out} = _ResBlock_{out}()",
        ]
        return ["import torch.nn as nn"], lines
