"""Pandas DataFrame creation nodes."""
import json
from core.node import BaseNode, PortType

C = "Pandas"


class PdFromCsvNode(BaseNode):
    type_name = "pd_from_csv"
    label = "From CSV"
    category = C
    description = "pd.read_csv(path, sep=sep)"

    def _setup_ports(self):
        self.add_input("path", PortType.STRING, "data.csv")
        self.add_input("sep",  PortType.STRING, ",")
        self.add_output("df",  PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            import pandas as pd
            return {"df": pd.read_csv(inputs.get("path") or "data.csv",
                                      sep=inputs.get("sep") or ",")}
        except Exception:
            return {"df": None}


class PdFromNumpyNode(BaseNode):
    type_name = "pd_from_numpy"
    label = "From NumPy"
    category = C
    description = "pd.DataFrame(array) — optional comma-separated column names"

    def _setup_ports(self):
        self.add_input("array",   PortType.NDARRAY)
        self.add_input("columns", PortType.STRING, "")
        self.add_output("df",     PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            import pandas as pd
            array   = inputs.get("array")
            col_str = inputs.get("columns") or ""
            if array is None:
                return {"df": None}
            cols = [c.strip() for c in col_str.split(",") if c.strip()] or None
            return {"df": pd.DataFrame(array, columns=cols)}
        except Exception:
            return {"df": None}


class PdFromDictNode(BaseNode):
    type_name = "pd_from_dict"
    label = "From JSON Dict"
    category = C
    description = "pd.DataFrame(json.loads(json_str))"

    def _setup_ports(self):
        self.add_input("json_str", PortType.STRING, "{}")
        self.add_output("df",      PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            import pandas as pd
            return {"df": pd.DataFrame(json.loads(inputs.get("json_str") or "{}"))}
        except Exception:
            return {"df": None}


class PdMakeSampleNode(BaseNode):
    type_name = "pd_make_sample"
    label = "Make Sample"
    category = C
    description = "Random float DataFrame with col_0..col_n + binary label column"

    def _setup_ports(self):
        self.add_input("rows", PortType.INT, 100)
        self.add_input("cols", PortType.INT, 4)
        self.add_input("seed", PortType.INT, 42)
        self.add_output("df",  PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            import numpy as np
            import pandas as pd
            rows = int(inputs.get("rows") or 100)
            cols = int(inputs.get("cols") or 4)
            seed = int(inputs.get("seed") if inputs.get("seed") is not None else 42)
            rng  = np.random.default_rng(seed)
            data = {f"col_{i}": rng.random(rows) for i in range(cols)}
            data["label"] = rng.integers(0, 2, rows)
            return {"df": pd.DataFrame(data)}
        except Exception:
            return {"df": None}
