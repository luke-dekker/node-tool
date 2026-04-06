"""Tensor Split node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class TensorSplitNode(BaseNode):
    type_name   = "pt_tensor_split"
    label       = "Split"
    category    = "Analyze"
    subcategory = "Tensors"
    description = "torch.split(tensor, split_size, dim). Returns first two chunks (chunk_0, chunk_1)."

    def _setup_ports(self):
        self.add_input("tensor",     PortType.TENSOR, default=None)
        self.add_input("split_size", PortType.INT,    default=1)
        self.add_input("dim",        PortType.INT,    default=0)
        self.add_output("chunk_0", PortType.TENSOR)
        self.add_output("chunk_1", PortType.TENSOR)

    def execute(self, inputs):
        try:
            import torch
            t = inputs.get("tensor")
            if t is None:
                return {"chunk_0": None, "chunk_1": None}
            chunks = torch.split(t, int(inputs.get("split_size") or 1),
                                 dim=int(inputs.get("dim") or 0))
            return {
                "chunk_0": chunks[0] if len(chunks) > 0 else None,
                "chunk_1": chunks[1] if len(chunks) > 1 else None,
            }
        except Exception:
            return {"chunk_0": None, "chunk_1": None}
