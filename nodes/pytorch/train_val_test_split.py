"""Train / Val / Test Split node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class TrainValTestSplitNode(BaseNode):
    type_name   = "pt_train_val_test_split"
    label       = "Train / Val / Test Split"
    category    = "Datasets"
    subcategory = "Loader"
    description = "Split a Dataset into train, validation, and test subsets."

    def _setup_ports(self):
        self.add_input("dataset",    PortType.DATASET, default=None)
        self.add_input("val_ratio",  PortType.FLOAT,   default=0.1)
        self.add_input("test_ratio", PortType.FLOAT,   default=0.1)
        self.add_input("seed",       PortType.INT,     default=42)
        self.add_output("train_dataset", PortType.DATASET)
        self.add_output("val_dataset",   PortType.DATASET)
        self.add_output("test_dataset",  PortType.DATASET)
        self.add_output("info",          PortType.STRING)

    def execute(self, inputs):
        try:
            import torch
            from torch.utils.data import random_split
            ds = inputs.get("dataset")
            if ds is None:
                return {"train_dataset": None, "val_dataset": None, "test_dataset": None, "info": "No dataset"}
            val_r  = float(inputs.get("val_ratio")  or 0.1)
            test_r = float(inputs.get("test_ratio") or 0.1)
            seed   = int(inputs.get("seed") or 42)
            n = len(ds)
            n_val  = max(1, int(n * val_r))
            n_test = max(1, int(n * test_r))
            n_train = n - n_val - n_test
            gen = torch.Generator().manual_seed(seed)
            train_ds, val_ds, test_ds = random_split(ds, [n_train, n_val, n_test], generator=gen)
            info = f"train={n_train}  val={n_val}  test={n_test}"
            return {"train_dataset": train_ds, "val_dataset": val_ds, "test_dataset": test_ds, "info": info}
        except Exception:
            import traceback
            return {"train_dataset": None, "val_dataset": None, "test_dataset": None, "info": traceback.format_exc().split("\n")[-2]}

    def export(self, iv, ov):
        ds = self._val(iv, 'dataset'); vr = self._val(iv, 'val_ratio')
        tr = self._val(iv, 'test_ratio'); seed = self._val(iv, 'seed')
        return ["import torch", "from torch.utils.data import random_split"], [
            f"_n_{ds} = len({ds})",
            f"_nv_{ds} = max(1, int(_n_{ds} * {vr}))",
            f"_nte_{ds} = max(1, int(_n_{ds} * {tr}))",
            f"_ntr_{ds} = _n_{ds} - _nv_{ds} - _nte_{ds}",
            f"{ov['train_dataset']}, {ov['val_dataset']}, {ov['test_dataset']} = random_split({ds}, [_ntr_{ds}, _nv_{ds}, _nte_{ds}], generator=torch.Generator().manual_seed({seed}))",
        ]
