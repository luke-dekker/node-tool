"""HuggingFace Dataset node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class HuggingFaceDatasetNode(BaseNode):
    type_name   = "pt_hf_dataset"
    label       = "HuggingFace Dataset"
    category    = "Datasets"
    subcategory = "Sources"
    description = "Load a dataset from HuggingFace Hub via datasets.load_dataset(). Outputs a TensorDataset for text classification."

    def _setup_ports(self):
        self.add_input("dataset_name", PortType.STRING, default="imdb")
        self.add_input("split",        PortType.STRING, default="train")
        self.add_input("text_col",     PortType.STRING, default="text")
        self.add_input("label_col",    PortType.STRING, default="label")
        self.add_input("max_samples",  PortType.INT,    default=0)
        self.add_output("dataset", PortType.DATASET)
        self.add_output("info",    PortType.STRING)

    def execute(self, inputs):
        try:
            from datasets import load_dataset
            import torch
            from torch.utils.data import Dataset as TorchDataset

            name   = str(inputs.get("dataset_name") or "imdb")
            split  = str(inputs.get("split") or "train")
            text_col  = str(inputs.get("text_col") or "text")
            label_col = str(inputs.get("label_col") or "label")
            max_n = int(inputs.get("max_samples") or 0)

            hf_ds = load_dataset(name, split=split)
            if max_n > 0:
                hf_ds = hf_ds.select(range(min(max_n, len(hf_ds))))

            class HFWrapper(TorchDataset):
                def __init__(self, ds, tc, lc):
                    self.ds = ds; self.tc = tc; self.lc = lc
                def __len__(self): return len(self.ds)
                def __getitem__(self, i):
                    item = self.ds[i]
                    return item[self.tc], item[self.lc]

            dataset = HFWrapper(hf_ds, text_col, label_col)
            info = f"HFDataset '{name}' [{split}]: {len(dataset)} samples"
            return {"dataset": dataset, "info": info}
        except Exception:
            import traceback
            return {"dataset": None, "info": traceback.format_exc().split("\n")[-2]}

    def export(self, iv, ov):
        name = self._val(iv, 'dataset_name'); split = self._val(iv, 'split')
        dsv = ov['dataset']; infov = ov['info']
        return ["from datasets import load_dataset"], [
            f"{dsv} = load_dataset({name}, split={split})",
            f"{infov} = f'HFDataset: {{len({dsv})}} samples'",
        ]
