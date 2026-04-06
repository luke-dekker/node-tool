"""Sample Batch node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class SampleBatchNode(BaseNode):
    type_name   = "pt_sample_batch"
    label       = "Sample Batch"
    category    = "Datasets"
    subcategory = "Loader"
    description = "Peek at the first batch of a DataLoader and output x (and optionally y) as tensors"

    def _setup_ports(self):
        self.add_input("dataloader", PortType.DATALOADER, default=None)
        self.add_output("x", PortType.TENSOR)
        self.add_output("y", PortType.TENSOR)

    def execute(self, inputs):
        try:
            dl = inputs.get("dataloader")
            if dl is None:
                return {"x": None, "y": None}
            batch = next(iter(dl))
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            y = batch[1] if isinstance(batch, (list, tuple)) and len(batch) > 1 else None
            return {"x": x, "y": y}
        except Exception:
            return {"x": None, "y": None}
