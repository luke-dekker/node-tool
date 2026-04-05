"""Dataset operation nodes — split, wrap, load, inspect."""
from __future__ import annotations
from core.node import BaseNode
from core.node import PortType

CATEGORY = "Datasets"


class ApplyTransformNode(BaseNode):
    type_name = "pt_apply_transform"
    label = "Apply Transform"
    category = CATEGORY
    description = "Wrap a Dataset so that transform is applied to every sample's first element (e.g. image)."

    def _setup_ports(self):
        self.add_input("dataset",   PortType.DATASET,    default=None)
        self.add_input("transform", PortType.TRANSFORM,  default=None)
        self.add_output("dataset", PortType.DATASET)

    def execute(self, inputs):
        try:
            from torch.utils.data import Dataset as TorchDataset
            ds = inputs.get("dataset")
            tf = inputs.get("transform")
            if ds is None:
                return {"dataset": None}
            if tf is None:
                return {"dataset": ds}

            class TransformedDataset(TorchDataset):
                def __init__(self, base, transform):
                    self.base = base; self.transform = transform
                def __len__(self): return len(self.base)
                def __getitem__(self, i):
                    sample = self.base[i]
                    if isinstance(sample, (list, tuple)):
                        return (self.transform(sample[0]),) + tuple(sample[1:])
                    return self.transform(sample)

            return {"dataset": TransformedDataset(ds, tf)}
        except Exception:
            return {"dataset": None}


class TrainValSplitNode(BaseNode):
    type_name = "pt_train_val_split"
    label = "Train / Val Split"
    category = CATEGORY
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


class TrainValTestSplitNode(BaseNode):
    type_name = "pt_train_val_test_split"
    label = "Train / Val / Test Split"
    category = CATEGORY
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


class DataLoaderNode(BaseNode):
    type_name = "pt_dataloader"
    label = "DataLoader"
    category = CATEGORY
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


class DatasetInfoNode(BaseNode):
    type_name = "pt_dataset_info"
    label = "Dataset Info"
    category = CATEGORY
    description = "Inspect a Dataset — size, sample shape, and class distribution if available."

    def _setup_ports(self):
        self.add_input("dataset", PortType.DATASET, default=None)
        self.add_output("info", PortType.STRING)

    def execute(self, inputs):
        try:
            import torch
            ds = inputs.get("dataset")
            if ds is None:
                return {"info": "No dataset"}
            n = len(ds)
            # Sample first item to get shape
            try:
                sample = ds[0]
                if isinstance(sample, (list, tuple)):
                    x_shape = list(sample[0].shape) if hasattr(sample[0], "shape") else type(sample[0]).__name__
                    shape_str = f"x={x_shape}"
                else:
                    shape_str = str(type(sample).__name__)
            except Exception:
                shape_str = "?"

            # Class distribution if labels accessible
            dist_str = ""
            try:
                if hasattr(ds, "targets"):
                    import collections
                    counts = collections.Counter(ds.targets if not hasattr(ds.targets, "tolist") else ds.targets.tolist())
                    dist_str = "  classes: " + " ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
            except Exception:
                pass

            info = f"{n} samples  {shape_str}{dist_str}"
            return {"info": info}
        except Exception:
            import traceback
            return {"info": traceback.format_exc().split("\n")[-2]}


# Subcategory stamp
_SC = "Loader"
from core.node import BaseNode as _BN
for _n, _c in list(globals().items()):
    if isinstance(_c, type) and issubclass(_c, _BN) and _c is not _BN:
        _c.subcategory = _SC
