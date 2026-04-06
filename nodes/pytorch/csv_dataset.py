"""CSV Dataset node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class CSVDatasetNode(BaseNode):
    type_name   = "pt_csv_dataset"
    label       = "CSV Dataset"
    category    = "Datasets"
    subcategory = "Sources"
    description = "Load a CSV file into a TensorDataset. feature_cols: comma-separated column names (blank = all except target). target_col: label column name."

    def _setup_ports(self):
        self.add_input("file_path",    PortType.STRING, default="data.csv")
        self.add_input("target_col",   PortType.STRING, default="label")
        self.add_input("feature_cols", PortType.STRING, default="")
        self.add_input("normalize",    PortType.BOOL,   default=False)
        self.add_output("dataset", PortType.DATASET)
        self.add_output("info",    PortType.STRING)

    def execute(self, inputs):
        try:
            import pandas as pd
            import torch
            from torch.utils.data import TensorDataset

            path = str(inputs.get("file_path") or "")
            target_col = str(inputs.get("target_col") or "label")
            feature_cols_raw = str(inputs.get("feature_cols") or "").strip()
            normalize = bool(inputs.get("normalize", False))

            df = pd.read_csv(path)

            if feature_cols_raw:
                feat_cols = [c.strip() for c in feature_cols_raw.split(",") if c.strip()]
            else:
                feat_cols = [c for c in df.columns if c != target_col]

            X = df[feat_cols].values.astype("float32")
            y = df[target_col].values

            if normalize:
                mean = X.mean(axis=0)
                std = X.std(axis=0) + 1e-8
                X = (X - mean) / std

            # Try int labels first, fallback to float
            try:
                y = y.astype("int64")
                y_tensor = torch.tensor(y, dtype=torch.long)
            except (ValueError, TypeError):
                y = y.astype("float32")
                y_tensor = torch.tensor(y, dtype=torch.float32)

            x_tensor = torch.tensor(X, dtype=torch.float32)
            dataset = TensorDataset(x_tensor, y_tensor)
            info = f"CSVDataset: {len(dataset)} samples, {len(feat_cols)} features"
            return {"dataset": dataset, "info": info}
        except Exception:
            import traceback
            return {"dataset": None, "info": traceback.format_exc().split("\n")[-2]}

    def export(self, iv, ov):
        path = self._val(iv, 'file_path'); target = self._val(iv, 'target_col')
        feat = self._val(iv, 'feature_cols'); norm = self._val(iv, 'normalize')
        dsv = ov['dataset']; infov = ov['info']
        return ["import pandas as pd", "import torch", "from torch.utils.data import TensorDataset"], [
            f"_df_{dsv} = pd.read_csv({path})",
            f"_feat_cols = [c for c in _df_{dsv}.columns if c != {target}] if not {feat} else [{feat}]",
            f"_X = _df_{dsv}[_feat_cols].values.astype('float32')",
            f"_y = _df_{dsv}[{target}].values",
            f"{dsv} = TensorDataset(torch.tensor(_X), torch.tensor(_y.astype('int64')))",
            f"{infov} = f'CSVDataset: {{len({dsv})}} samples'",
        ]
