"""Apply Transform node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class ApplyTransformNode(BaseNode):
    type_name   = "pt_apply_transform"
    label       = "Apply Transform"
    category    = "Datasets"
    subcategory = "Loader"
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

    def export(self, iv, ov):
        ds = self._val(iv, 'dataset'); tf = self._val(iv, 'transform')
        dsv = ov['dataset']
        return ["from torch.utils.data import Dataset as _TDS"], [
            f"class _TD_{dsv}(_TDS):",
            f"    def __init__(self,b,t): self.b=b; self.t=t",
            f"    def __len__(self): return len(self.b)",
            f"    def __getitem__(self,i):",
            f"        s=self.b[i]; return (self.t(s[0]),)+tuple(s[1:]) if isinstance(s,(list,tuple)) else self.t(s)",
            f"{dsv} = _TD_{dsv}({ds}, {tf})",
        ]
