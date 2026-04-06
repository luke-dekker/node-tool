"""Dataset Info node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class DatasetInfoNode(BaseNode):
    type_name   = "pt_dataset_info"
    label       = "Dataset Info"
    category    = "Datasets"
    subcategory = "Loader"
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

    def export(self, iv, ov):
        ds = self._val(iv, 'dataset')
        return [], [f"{ov['info']} = f'Dataset: {{len({ds})}} samples'"]
