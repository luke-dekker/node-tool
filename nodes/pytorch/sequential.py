"""nn.Sequential (star topology) node."""
from __future__ import annotations
from typing import Any
import torch.nn as nn
from core.node import BaseNode, PortType


class SequentialNode(BaseNode):
    type_name   = "pt_sequential"
    label       = "Sequential"
    category    = "Models"
    subcategory = "Build"
    description = (
        "nn.Sequential(*layers) - star topology. "
        "Wire up to 8 layer nodes (via their tensor_out) into this node."
    )

    def _setup_ports(self) -> None:
        for i in range(1, 9):
            self.add_input(f"layer_{i}", PortType.MODULE, default=None)
        self.add_output("model", PortType.MODULE)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        try:
            layers = [inputs[f"layer_{i}"] for i in range(1, 9)
                      if inputs.get(f"layer_{i}") is not None]
            return {"model": nn.Sequential(*layers) if layers else None}
        except Exception:
            return {"model": None}

    def export(self, iv, ov):
        layers = [iv.get(f"layer_{i}") for i in range(1, 9) if iv.get(f"layer_{i}") is not None]
        if not layers:
            return ["import torch.nn as nn"], [f"{ov['model']} = nn.Sequential()  # no layers connected"]
        return ["import torch.nn as nn"], [f"{ov['model']} = nn.Sequential({', '.join(layers)})"]
