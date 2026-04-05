"""Pandas nodes — DataFrame creation, inspection, filtering, transforms."""
from __future__ import annotations
import json
from core.node import BaseNode
from core.node import PortType

CATEGORY = "Pandas"


# ── Creation ──────────────────────────────────────────────────────────────────

class PdFromCsvNode(BaseNode):
    type_name = "pd_from_csv"
    label = "From CSV"
    category = CATEGORY
    description = "pd.read_csv(path, sep=sep)"

    def _setup_ports(self):
        self.add_input("path", PortType.STRING, "data.csv")
        self.add_input("sep",  PortType.STRING, ",")
        self.add_output("df",  PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            import pandas as pd
            path = inputs.get("path", "data.csv") or "data.csv"
            sep  = inputs.get("sep",  ",") or ","
            return {"df": pd.read_csv(path, sep=sep)}
        except Exception:
            return {"df": None}


class PdFromNumpyNode(BaseNode):
    type_name = "pd_from_numpy"
    label = "From NumPy"
    category = CATEGORY
    description = "pd.DataFrame(array) — optional comma-separated column names"

    def _setup_ports(self):
        self.add_input("array",   PortType.NDARRAY)
        self.add_input("columns", PortType.STRING, "")
        self.add_output("df",     PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            import pandas as pd
            array   = inputs.get("array")
            col_str = inputs.get("columns", "") or ""
            if array is None:
                return {"df": None}
            cols = [c.strip() for c in col_str.split(",") if c.strip()] if col_str.strip() else None
            return {"df": pd.DataFrame(array, columns=cols)}
        except Exception:
            return {"df": None}


class PdFromDictNode(BaseNode):
    type_name = "pd_from_dict"
    label = "From JSON Dict"
    category = CATEGORY
    description = "pd.DataFrame(json.loads(json_str))"

    def _setup_ports(self):
        self.add_input("json_str", PortType.STRING, "{}")
        self.add_output("df",      PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            import pandas as pd
            json_str = inputs.get("json_str", "{}") or "{}"
            return {"df": pd.DataFrame(json.loads(json_str))}
        except Exception:
            return {"df": None}


class PdMakeSampleNode(BaseNode):
    type_name = "pd_make_sample"
    label = "Make Sample"
    category = CATEGORY
    description = "Random float DataFrame with col_0..col_n + binary label column"

    def _setup_ports(self):
        self.add_input("rows", PortType.INT,  100)
        self.add_input("cols", PortType.INT,  4)
        self.add_input("seed", PortType.INT,  42)
        self.add_output("df",  PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            import numpy as np
            import pandas as pd
            rows = int(inputs.get("rows", 100) or 100)
            cols = int(inputs.get("cols", 4)   or 4)
            seed = int(inputs.get("seed", 42)  if inputs.get("seed") is not None else 42)
            rng = np.random.default_rng(seed)
            data = {f"col_{i}": rng.random(rows) for i in range(cols)}
            data["label"] = rng.integers(0, 2, rows)
            return {"df": pd.DataFrame(data)}
        except Exception:
            return {"df": None}


# ── Inspection ────────────────────────────────────────────────────────────────

class PdHeadNode(BaseNode):
    type_name = "pd_head"
    label = "Head"
    category = CATEGORY
    description = "df.head(n).to_string()"

    def _setup_ports(self):
        self.add_input("df",     PortType.DATAFRAME)
        self.add_input("n",      PortType.INT, 5)
        self.add_output("result", PortType.STRING)

    def execute(self, inputs):
        try:
            df = inputs.get("df")
            n  = inputs.get("n", 5)
            if df is None:
                return {"result": "None"}
            return {"result": df.head(int(n)).to_string()}
        except Exception:
            return {"result": "error"}


class PdDescribeNode(BaseNode):
    type_name = "pd_describe"
    label = "Describe"
    category = CATEGORY
    description = "df.describe().to_string()"

    def _setup_ports(self):
        self.add_input("df",     PortType.DATAFRAME)
        self.add_output("result", PortType.STRING)

    def execute(self, inputs):
        try:
            df = inputs.get("df")
            if df is None:
                return {"result": "None"}
            return {"result": df.describe().to_string()}
        except Exception:
            return {"result": "error"}


class PdInfoNode(BaseNode):
    type_name = "pd_info"
    label = "Info"
    category = CATEGORY
    description = "shape, columns, dtypes summary"

    def _setup_ports(self):
        self.add_input("df",     PortType.DATAFRAME)
        self.add_output("result", PortType.STRING)

    def execute(self, inputs):
        try:
            df = inputs.get("df")
            if df is None:
                return {"result": "None"}
            dtypes = {col: str(dt) for col, dt in df.dtypes.items()}
            return {"result": f"shape={df.shape} columns={list(df.columns)} dtypes={dtypes}"}
        except Exception:
            return {"result": "error"}


class PdShapeNode(BaseNode):
    type_name = "pd_shape"
    label = "Shape"
    category = CATEGORY
    description = "Return df.shape as rows, cols"

    def _setup_ports(self):
        self.add_input("df",   PortType.DATAFRAME)
        self.add_output("rows", PortType.INT)
        self.add_output("cols", PortType.INT)

    def execute(self, inputs):
        try:
            df = inputs.get("df")
            if df is None:
                return {"rows": 0, "cols": 0}
            return {"rows": df.shape[0], "cols": df.shape[1]}
        except Exception:
            return {"rows": 0, "cols": 0}


# ── Selection & Filter ────────────────────────────────────────────────────────

class PdSelectColsNode(BaseNode):
    type_name = "pd_select_cols"
    label = "Select Columns"
    category = CATEGORY
    description = "df[col_list] — comma-separated column names"

    def _setup_ports(self):
        self.add_input("df",      PortType.DATAFRAME)
        self.add_input("columns", PortType.STRING, "col1,col2")
        self.add_output("result", PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            df      = inputs.get("df")
            col_str = inputs.get("columns", "") or ""
            if df is None:
                return {"result": None}
            cols = [c.strip() for c in col_str.split(",") if c.strip()]
            return {"result": df[cols]}
        except Exception:
            return {"result": None}


class PdDropColsNode(BaseNode):
    type_name = "pd_drop_cols"
    label = "Drop Columns"
    category = CATEGORY
    description = "df.drop(columns=[...]) — comma-separated"

    def _setup_ports(self):
        self.add_input("df",      PortType.DATAFRAME)
        self.add_input("columns", PortType.STRING, "col1")
        self.add_output("result", PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            df      = inputs.get("df")
            col_str = inputs.get("columns", "") or ""
            if df is None:
                return {"result": None}
            cols = [c.strip() for c in col_str.split(",") if c.strip()]
            return {"result": df.drop(columns=cols)}
        except Exception:
            return {"result": None}


class PdFilterRowsNode(BaseNode):
    type_name = "pd_filter_rows"
    label = "Filter Rows"
    category = CATEGORY
    description = "Filter rows: column op value. op: ==, !=, >, <, >=, <="

    def _setup_ports(self):
        self.add_input("df",     PortType.DATAFRAME)
        self.add_input("column", PortType.STRING, "label")
        self.add_input("op",     PortType.STRING, "==")
        self.add_input("value",  PortType.FLOAT,  1.0)
        self.add_output("result", PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            import operator as op_mod
            df     = inputs.get("df")
            col    = inputs.get("column", "label") or "label"
            op_str = inputs.get("op", "==") or "=="
            val    = inputs.get("value", 1.0)
            if df is None:
                return {"result": None}
            ops = {
                "==": op_mod.eq, "!=": op_mod.ne,
                ">":  op_mod.gt, "<":  op_mod.lt,
                ">=": op_mod.ge, "<=": op_mod.le,
            }
            fn = ops.get(op_str, op_mod.eq)
            mask = fn(df[col], val)
            return {"result": df[mask]}
        except Exception:
            return {"result": None}


class PdGetColumnNode(BaseNode):
    type_name = "pd_get_column"
    label = "Get Column"
    category = CATEGORY
    description = "df[column] — returns a Series"

    def _setup_ports(self):
        self.add_input("df",     PortType.DATAFRAME)
        self.add_input("column", PortType.STRING, "col_0")
        self.add_output("series", PortType.SERIES)

    def execute(self, inputs):
        try:
            df  = inputs.get("df")
            col = inputs.get("column", "col_0") or "col_0"
            if df is None:
                return {"series": None}
            return {"series": df[col]}
        except Exception:
            return {"series": None}


# ── Transforms ────────────────────────────────────────────────────────────────

class PdDropNaNode(BaseNode):
    type_name = "pd_dropna"
    label = "Drop NA"
    category = CATEGORY
    description = "df.dropna()"

    def _setup_ports(self):
        self.add_input("df",     PortType.DATAFRAME)
        self.add_output("result", PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            df = inputs.get("df")
            if df is None:
                return {"result": None}
            return {"result": df.dropna()}
        except Exception:
            return {"result": None}


class PdFillNaNode(BaseNode):
    type_name = "pd_fillna"
    label = "Fill NA"
    category = CATEGORY
    description = "df.fillna(value)"

    def _setup_ports(self):
        self.add_input("df",    PortType.DATAFRAME)
        self.add_input("value", PortType.FLOAT, 0.0)
        self.add_output("result", PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            df  = inputs.get("df")
            val = inputs.get("value", 0.0)
            if df is None:
                return {"result": None}
            return {"result": df.fillna(float(val))}
        except Exception:
            return {"result": None}


class PdSortNode(BaseNode):
    type_name = "pd_sort"
    label = "Sort"
    category = CATEGORY
    description = "df.sort_values(by, ascending)"

    def _setup_ports(self):
        self.add_input("df",        PortType.DATAFRAME)
        self.add_input("by",        PortType.STRING, "col_0")
        self.add_input("ascending", PortType.BOOL,   True)
        self.add_output("result",   PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            df  = inputs.get("df")
            by  = inputs.get("by", "col_0") or "col_0"
            asc = inputs.get("ascending", True)
            if df is None:
                return {"result": None}
            return {"result": df.sort_values(by=by, ascending=bool(asc))}
        except Exception:
            return {"result": None}


class PdResetIndexNode(BaseNode):
    type_name = "pd_reset_index"
    label = "Reset Index"
    category = CATEGORY
    description = "df.reset_index(drop=True)"

    def _setup_ports(self):
        self.add_input("df",     PortType.DATAFRAME)
        self.add_output("result", PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            df = inputs.get("df")
            if df is None:
                return {"result": None}
            return {"result": df.reset_index(drop=True)}
        except Exception:
            return {"result": None}


class PdRenameColNode(BaseNode):
    type_name = "pd_rename_col"
    label = "Rename Column"
    category = CATEGORY
    description = "Rename a single column old_name -> new_name"

    def _setup_ports(self):
        self.add_input("df",       PortType.DATAFRAME)
        self.add_input("old_name", PortType.STRING, "col_0")
        self.add_input("new_name", PortType.STRING, "x")
        self.add_output("result",  PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            df       = inputs.get("df")
            old_name = inputs.get("old_name", "col_0") or "col_0"
            new_name = inputs.get("new_name", "x")     or "x"
            if df is None:
                return {"result": None}
            return {"result": df.rename(columns={old_name: new_name})}
        except Exception:
            return {"result": None}


class PdToNumpyNode(BaseNode):
    type_name = "pd_to_numpy"
    label = "To NumPy"
    category = CATEGORY
    description = "df.values — convert DataFrame to ndarray"

    def _setup_ports(self):
        self.add_input("df",     PortType.DATAFRAME)
        self.add_output("array", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            df = inputs.get("df")
            if df is None:
                return {"array": None}
            return {"array": df.values}
        except Exception:
            return {"array": None}


class PdNormalizeNode(BaseNode):
    type_name = "pd_normalize"
    label = "Normalize"
    category = CATEGORY
    description = "(df - df.min()) / (df.max() - df.min()) on numeric cols"

    def _setup_ports(self):
        self.add_input("df",     PortType.DATAFRAME)
        self.add_output("result", PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            df = inputs.get("df")
            if df is None:
                return {"result": None}
            num = df.select_dtypes(include="number")
            rng = num.max() - num.min()
            rng[rng == 0] = 1  # avoid division by zero
            out = df.copy()
            out[num.columns] = (num - num.min()) / rng
            return {"result": out}
        except Exception:
            return {"result": None}


# ── Aggregation ───────────────────────────────────────────────────────────────

class PdGroupByNode(BaseNode):
    type_name = "pd_groupby"
    label = "Group By"
    category = CATEGORY
    description = "df.groupby(by).agg(agg)"

    def _setup_ports(self):
        self.add_input("df",     PortType.DATAFRAME)
        self.add_input("by",     PortType.STRING, "label")
        self.add_input("agg",    PortType.STRING, "mean")
        self.add_output("result", PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            df  = inputs.get("df")
            by  = inputs.get("by",  "label") or "label"
            agg = inputs.get("agg", "mean")  or "mean"
            if df is None:
                return {"result": None}
            return {"result": df.groupby(by).agg(agg).reset_index()}
        except Exception:
            return {"result": None}


class PdCorrelationNode(BaseNode):
    type_name = "pd_correlation"
    label = "Correlation"
    category = CATEGORY
    description = "df.corr() — pairwise correlation matrix"

    def _setup_ports(self):
        self.add_input("df",     PortType.DATAFRAME)
        self.add_output("result", PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            df = inputs.get("df")
            if df is None:
                return {"result": None}
            return {"result": df.select_dtypes(include="number").corr()}
        except Exception:
            return {"result": None}


# ── XY Split & Merge ──────────────────────────────────────────────────────────

class PdXYSplitNode(BaseNode):
    type_name = "pd_xy_split"
    label = "XY Split"
    category = CATEGORY
    description = "Split DataFrame into features X and label series y"

    def _setup_ports(self):
        self.add_input("df",        PortType.DATAFRAME)
        self.add_input("label_col", PortType.STRING,   "label")
        self.add_output("X",        PortType.DATAFRAME)
        self.add_output("y",        PortType.SERIES)

    def execute(self, inputs):
        try:
            df        = inputs.get("df")
            label_col = inputs.get("label_col", "label") or "label"
            if df is None:
                return {"X": None, "y": None}
            y = df[label_col]
            X = df.drop(columns=[label_col])
            return {"X": X, "y": y}
        except Exception:
            return {"X": None, "y": None}


class PdMergeNode(BaseNode):
    type_name = "pd_merge"
    label = "Merge"
    category = CATEGORY
    description = "pd.merge(left, right, on=on, how=how)"

    def _setup_ports(self):
        self.add_input("left",    PortType.DATAFRAME)
        self.add_input("right",   PortType.DATAFRAME)
        self.add_input("on",      PortType.STRING,   "index")
        self.add_input("how",     PortType.STRING,   "inner")
        self.add_output("result", PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            import pandas as pd
            left  = inputs.get("left")
            right = inputs.get("right")
            on    = inputs.get("on",  "index") or "index"
            how   = inputs.get("how", "inner") or "inner"
            if left is None or right is None:
                return {"result": None}
            if on == "index":
                return {"result": pd.merge(left, right, left_index=True, right_index=True, how=how)}
            return {"result": pd.merge(left, right, on=on, how=how)}
        except Exception:
            return {"result": None}
