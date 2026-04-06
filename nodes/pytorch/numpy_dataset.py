"""Numpy Dataset node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class NumpyDatasetNode(BaseNode):
    type_name   = "pt_numpy_dataset"
    label       = "Numpy Dataset"
    category    = "Datasets"
    subcategory = "Sources"
    description = "Wrap X (ndarray) and y (ndarray) into a TensorDataset."

    def _setup_ports(self):
        self.add_input("X", PortType.NDARRAY, default=None)
        self.add_input("y", PortType.NDARRAY, default=None)
        self.add_output("dataset", PortType.DATASET)
        self.add_output("info",    PortType.STRING)

    def execute(self, inputs):
        try:
            import torch
            from torch.utils.data import TensorDataset
            X = inputs.get("X")
            y = inputs.get("y")
            if X is None or y is None:
                return {"dataset": None, "info": "X and y required"}
            x_t = torch.tensor(X.astype("float32"))
            try:
                y_t = torch.tensor(y.astype("int64"), dtype=torch.long)
            except (ValueError, TypeError):
                y_t = torch.tensor(y.astype("float32"), dtype=torch.float32)
            dataset = TensorDataset(x_t, y_t)
            return {"dataset": dataset, "info": f"NumpyDataset: {len(dataset)} samples, shape {list(x_t.shape)}"}
        except Exception:
            import traceback
            return {"dataset": None, "info": traceback.format_exc().split("\n")[-2]}

    def export(self, iv, ov):
        X = self._val(iv, 'X'); y = self._val(iv, 'y')
        dsv = ov['dataset']; infov = ov['info']
        return ["import torch", "from torch.utils.data import TensorDataset"], [
            f"{dsv} = TensorDataset(torch.tensor({X}.astype('float32')), torch.tensor({y}.astype('int64')))",
            f"{infov} = f'NumpyDataset: {{len({dsv})}} samples'",
        ]
