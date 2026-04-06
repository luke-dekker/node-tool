"""DataLoader node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class DataLoaderNode(BaseNode):
    type_name   = "pt_dataloader"
    label       = "DataLoader"
    category    = "Datasets"
    subcategory = "Loader"
    description = "Wrap a Dataset into a DataLoader ready for training."

    def _setup_ports(self):
        self.add_input("dataset",     PortType.DATASET, default=None)
        self.add_input("batch_size",  PortType.INT,     default=32)
        self.add_input("shuffle",     PortType.BOOL,    default=True)
        self.add_input("num_workers", PortType.INT,     default=0)
        self.add_input("pin_memory",  PortType.BOOL,    default=False)
        self.add_input("drop_last",   PortType.BOOL,    default=False)
        self.add_output("dataloader", PortType.DATALOADER)
        self.add_output("info",       PortType.STRING)

    def execute(self, inputs):
        try:
            from torch.utils.data import DataLoader
            ds = inputs.get("dataset")
            if ds is None:
                return {"dataloader": None, "info": "No dataset connected"}
            bs  = int(inputs.get("batch_size")  or 32)
            shuf = bool(inputs.get("shuffle", True))
            nw   = int(inputs.get("num_workers") or 0)
            pm   = bool(inputs.get("pin_memory", False))
            dl_  = bool(inputs.get("drop_last", False))
            dl = DataLoader(ds, batch_size=bs, shuffle=shuf,
                            num_workers=nw, pin_memory=pm, drop_last=dl_)
            info = f"DataLoader: {len(dl)} batches x {bs}"
            return {"dataloader": dl, "info": info}
        except Exception:
            import traceback
            return {"dataloader": None, "info": traceback.format_exc().split("\n")[-2]}

    def export(self, iv, ov):
        ds = self._val(iv, 'dataset'); bs = self._val(iv, 'batch_size'); sh = self._val(iv, 'shuffle')
        nw = self._val(iv, 'num_workers'); pm = self._val(iv, 'pin_memory'); dl = self._val(iv, 'drop_last')
        dlv = ov['dataloader']; infov = ov['info']
        return ["from torch.utils.data import DataLoader"], [
            f"{dlv} = DataLoader({ds}, batch_size={bs}, shuffle={sh}, num_workers={nw}, pin_memory={pm}, drop_last={dl})",
            f"{infov} = f'DataLoader: {{len({dlv})}} batches'",
        ]
