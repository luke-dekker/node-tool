"""Train / Val Split node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class TrainValSplitNode(BaseNode):
    type_name   = "pt_train_val_split"
    label       = "Train / Val Split"
    category    = "Data"
    subcategory = "Splits"
    description = "Randomly split a Dataset into train and validation subsets."

    def _setup_ports(self):
        self.add_input("dataset",   PortType.DATASET, default=None)
        self.add_input("val_ratio", PortType.FLOAT,   default=0.2)
        self.add_input("seed",      PortType.INT,     default=42)
        self.add_output("train_dataset", PortType.DATASET)
        self.add_output("val_dataset",   PortType.DATASET)
        self.add_output("info",          PortType.STRING)

    def execute(self, inputs):
        try:
            import torch
            from torch.utils.data import random_split
            ds = inputs.get("dataset")
            if ds is None:
                return {"train_dataset": None, "val_dataset": None, "info": "No dataset"}
            val_ratio = float(inputs.get("val_ratio") or 0.2)
            seed = int(inputs.get("seed") or 42)
            n = len(ds)
            n_val = max(1, int(n * val_ratio))
            n_train = n - n_val
            gen = torch.Generator().manual_seed(seed)
            train_ds, val_ds = random_split(ds, [n_train, n_val], generator=gen)
            info = f"train={n_train}  val={n_val}"
            return {"train_dataset": train_ds, "val_dataset": val_ds, "info": info}
        except Exception:
            import traceback
            return {"train_dataset": None, "val_dataset": None, "info": traceback.format_exc().split("\n")[-2]}

    def export(self, iv, ov):
        ds = self._val(iv, 'dataset'); vr = self._val(iv, 'val_ratio'); seed = self._val(iv, 'seed')
        return ["import torch", "from torch.utils.data import random_split"], [
            f"_n_{ds} = len({ds})",
            f"_nv_{ds} = max(1, int(_n_{ds} * {vr}))",
            f"_nt_{ds} = _n_{ds} - _nv_{ds}",
            f"{ov['train_dataset']}, {ov['val_dataset']} = random_split({ds}, [_nt_{ds}, _nv_{ds}], generator=torch.Generator().manual_seed({seed}))",
        ]
