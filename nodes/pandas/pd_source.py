"""Consolidated DataFrame source node.

Replaces PdFromCsvNode, PdFromDictNode, PdFromNumpyNode, PdMakeSampleNode.
Pick a `kind` from the dropdown; only the inputs that kind reads are
consulted, the rest are ignored. Output is always a DataFrame on `df`.
"""
from __future__ import annotations
import json as _json
from core.node import BaseNode, PortType


_KINDS = ["csv", "json", "numpy", "sample"]


class PdSourceNode(BaseNode):
    type_name   = "pd_source"
    label       = "DataFrame Source"
    category    = "Pandas"
    description = (
        "Build a DataFrame. Pick `kind`:\n"
        "  csv    — read path with sep\n"
        "  json   — parse json_str\n"
        "  numpy  — wrap array with optional comma-separated columns\n"
        "  sample — random demo data: rows × cols + binary 'label' column"
    )

    def relevant_inputs(self, values):
        kind = (values.get("kind") or "csv").strip()
        if kind == "csv":    return ["kind", "path", "sep"]
        if kind == "json":   return ["kind", "json_str"]
        if kind == "numpy":  return ["kind", "columns"]   # array wired
        if kind == "sample": return ["kind", "rows", "cols", "seed"]
        return ["kind"]

    def _setup_ports(self):
        self.add_input("kind",     PortType.STRING, "csv", choices=_KINDS)
        # csv
        self.add_input("path",     PortType.STRING, "data.csv", optional=True)
        self.add_input("sep",      PortType.STRING, ",", optional=True)
        # json
        self.add_input("json_str", PortType.STRING, "{}", optional=True)
        # numpy
        self.add_input("array",    PortType.NDARRAY, optional=True)
        self.add_input("columns",  PortType.STRING, "", optional=True)
        # sample
        self.add_input("rows",     PortType.INT,    100, optional=True)
        self.add_input("cols",     PortType.INT,    4,   optional=True)
        self.add_input("seed",     PortType.INT,    42,  optional=True)
        self.add_output("df", PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            import pandas as pd
            kind = (inputs.get("kind") or "csv").strip()
            if kind == "csv":
                return {"df": pd.read_csv(inputs.get("path") or "data.csv",
                                          sep=inputs.get("sep") or ",")}
            if kind == "json":
                return {"df": pd.DataFrame(_json.loads(inputs.get("json_str") or "{}"))}
            if kind == "numpy":
                arr = inputs.get("array")
                if arr is None:
                    return {"df": None}
                col_str = inputs.get("columns") or ""
                cols = [c.strip() for c in col_str.split(",") if c.strip()] or None
                return {"df": pd.DataFrame(arr, columns=cols)}
            if kind == "sample":
                import numpy as np
                rows = int(inputs.get("rows") or 100)
                cols = int(inputs.get("cols") or 4)
                seed = int(inputs.get("seed") if inputs.get("seed") is not None else 42)
                rng  = np.random.default_rng(seed)
                data = {f"col_{i}": rng.random(rows) for i in range(cols)}
                data["label"] = rng.integers(0, 2, rows)
                return {"df": pd.DataFrame(data)}
            return {"df": None}
        except Exception:
            return {"df": None}

    def export(self, iv, ov):
        kind = (self.inputs["kind"].default_value or "csv")
        if kind == "csv":
            return (["import pandas as pd"],
                    [f"{ov['df']} = pd.read_csv({self._val(iv, 'path')}, sep={self._val(iv, 'sep')})"])
        if kind == "json":
            return (["import pandas as pd", "import json"],
                    [f"{ov['df']} = pd.DataFrame(json.loads({self._val(iv, 'json_str')}))"])
        if kind == "numpy":
            arr  = self._val(iv, "array")
            cols = self._val(iv, "columns")
            return (["import pandas as pd"],
                    [f"_cols = [c.strip() for c in {cols}.split(',') if c.strip()] or None",
                     f"{ov['df']} = pd.DataFrame({arr}, columns=_cols)"])
        if kind == "sample":
            rows = self._val(iv, "rows"); cols = self._val(iv, "cols"); seed = self._val(iv, "seed")
            return (["import numpy as np", "import pandas as pd"],
                    [f"_rng = np.random.default_rng({seed})",
                     f"{ov['df']} = pd.DataFrame({{f'col_{{i}}': _rng.random({rows}) for i in range({cols})}})",
                     f"{ov['df']}['label'] = _rng.integers(0, 2, {rows})"])
        return [], [f"# unknown kind {kind!r}"]
