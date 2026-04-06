"""DataLoader Info node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class DataLoaderInfoNode(BaseNode):
    type_name   = "pt_dataloader_info"
    label       = "DataLoader Info"
    category    = "Datasets"
    subcategory = "Loader"
    description = "Return DataLoader info string"

    def _setup_ports(self):
        self.add_input("dataloader", PortType.DATALOADER, default=None)
        self.add_output("info", PortType.STRING)

    def execute(self, inputs):
        try:
            dl = inputs.get("dataloader")
            if dl is None:
                return {"info": "None"}
            return {"info": f"batches={len(dl)} batch_size={dl.batch_size}"}
        except Exception:
            return {"info": "error"}

    def export(self, iv, ov):
        dl = self._val(iv, 'dataloader')
        return [], [f"{ov['info']} = f\"batches={{len({dl})}} batch_size={{{dl}.batch_size}}\""]
