"""Consolidated DataFrame node — every df→? operation in one place.

Absorbs all the df-input nodes:
  - PdTransformNode original ops (dropna, fillna, drop_cols, select_cols,
    rename_col, reset_index, normalize, sort, filter_rows, correlation)
  - PdInfoNode modes (describe, head, info)
  - PdGetColumnNode (get_column → series)
  - PdGroupByNode (groupby → DataFrame)
  - PdXYSplitNode (xy_split → X, y)
  - PdToNumpyNode (to_numpy → array)
  - PdShapeNode (shape → rows, cols)

Outputs (each populated by the relevant op; others stay None/0/""):
  result   — DATAFRAME (transforms, groupby, X side of xy_split)
  series   — SERIES    (get_column, y side of xy_split)
  array    — NDARRAY   (to_numpy)
  info     — STRING    (describe, head, info, shape summary)
  rows     — INT       (shape)
  cols     — INT       (shape)
"""
from __future__ import annotations
import operator as _op
from core.node import BaseNode, PortType


_TRANSFORMS  = ["dropna", "fillna", "drop_cols", "select_cols", "rename_col",
                "reset_index", "normalize", "sort", "filter_rows", "correlation"]
_GROUPBY_OPS = ["groupby"]
_DF_TO_DF    = _TRANSFORMS + _GROUPBY_OPS
_INFO_OPS    = ["describe", "head", "info", "shape"]   # → info / rows / cols
_EXTRACT_OPS = ["get_column", "to_numpy", "xy_split"]
_OPS = _DF_TO_DF + _INFO_OPS + _EXTRACT_OPS

_COMP = {"==": _op.eq, "!=": _op.ne, ">": _op.gt, "<": _op.lt,
         ">=": _op.ge, "<=": _op.le}


class PdTransformNode(BaseNode):
    type_name   = "pd_transform"
    label       = "DataFrame Op"
    category    = "Pandas"
    description = (
        "Any df → result operation. Pick `op` from the dropdown — the "
        "inspector hides fields that don't apply. Output port depends on op:\n"
        "  transforms / groupby     → result (DATAFRAME)\n"
        "  get_column               → series (SERIES)\n"
        "  to_numpy                 → array  (NDARRAY)\n"
        "  xy_split                 → result=X, series=y\n"
        "  describe / head / info   → info   (STRING)\n"
        "  shape                    → rows, cols (INT) + info"
    )

    def relevant_inputs(self, values):
        op = (values.get("op") or "dropna").strip()
        per_op = {
            # transforms
            "dropna":      [],
            "fillna":      ["value"],
            "drop_cols":   ["columns"],
            "select_cols": ["columns"],
            "rename_col":  ["old_name", "new_name"],
            "reset_index": [],
            "normalize":   [],
            "sort":        ["by", "ascending"],
            "filter_rows": ["column", "compare", "value"],
            "correlation": [],
            "groupby":     ["by", "agg"],
            # info
            "describe":    [],
            "head":        ["n"],
            "info":        [],
            "shape":       [],
            # extract
            "get_column":  ["column"],
            "to_numpy":    [],
            "xy_split":    ["label_col"],
        }
        return ["op"] + per_op.get(op, [])

    def _setup_ports(self):
        self.add_input("df",        PortType.DATAFRAME)
        self.add_input("op",        PortType.STRING, "dropna", choices=_OPS)
        # union of per-op args
        self.add_input("columns",   PortType.STRING, "col1,col2", optional=True)
        self.add_input("column",    PortType.STRING, "label",     optional=True)
        self.add_input("old_name",  PortType.STRING, "col_0",     optional=True)
        self.add_input("new_name",  PortType.STRING, "x",         optional=True)
        self.add_input("by",        PortType.STRING, "col_0",     optional=True)
        self.add_input("ascending", PortType.BOOL,   True,        optional=True)
        self.add_input("compare",   PortType.STRING, "==",
                       choices=list(_COMP.keys()), optional=True)
        self.add_input("value",     PortType.FLOAT,  0.0,         optional=True)
        self.add_input("agg",       PortType.STRING, "mean",      optional=True)
        self.add_input("n",         PortType.INT,    5,           optional=True)
        self.add_input("label_col", PortType.STRING, "label",     optional=True)
        # outputs — each op populates a subset
        self.add_output("result", PortType.DATAFRAME)
        self.add_output("series", PortType.SERIES)
        self.add_output("array",  PortType.NDARRAY)
        self.add_output("info",   PortType.STRING)
        self.add_output("rows",   PortType.INT)
        self.add_output("cols",   PortType.INT)

    def _empty(self) -> dict:
        return {"result": None, "series": None, "array": None,
                "info": "", "rows": 0, "cols": 0}

    def execute(self, inputs):
        out = self._empty()
        try:
            df = inputs.get("df")
            if df is None:
                return out
            op = (inputs.get("op") or "dropna").strip()

            # ── transforms (df → df) ────────────────────────────────────────
            if op == "dropna":
                out["result"] = df.dropna(); return out
            if op == "fillna":
                out["result"] = df.fillna(float(inputs.get("value", 0.0) or 0.0)); return out
            if op == "drop_cols":
                cols = [c.strip() for c in (inputs.get("columns") or "").split(",") if c.strip()]
                out["result"] = df.drop(columns=cols); return out
            if op == "select_cols":
                cols = [c.strip() for c in (inputs.get("columns") or "").split(",") if c.strip()]
                out["result"] = df[cols]; return out
            if op == "rename_col":
                old = inputs.get("old_name") or "col_0"
                new = inputs.get("new_name") or "x"
                out["result"] = df.rename(columns={old: new}); return out
            if op == "reset_index":
                out["result"] = df.reset_index(drop=True); return out
            if op == "normalize":
                num = df.select_dtypes(include="number")
                rng = num.max() - num.min()
                rng[rng == 0] = 1
                copy = df.copy()
                copy[num.columns] = (num - num.min()) / rng
                out["result"] = copy; return out
            if op == "sort":
                out["result"] = df.sort_values(by=inputs.get("by") or "col_0",
                                                ascending=bool(inputs.get("ascending", True)))
                return out
            if op == "filter_rows":
                col = inputs.get("column") or "label"
                fn  = _COMP.get(inputs.get("compare") or "==", _op.eq)
                out["result"] = df[fn(df[col], inputs.get("value", 0.0))]; return out
            if op == "correlation":
                out["result"] = df.select_dtypes(include="number").corr(); return out
            if op == "groupby":
                out["result"] = (df.groupby(inputs.get("by") or "label")
                                   .agg(inputs.get("agg") or "mean")
                                   .reset_index())
                return out

            # ── info (df → string / int outputs) ────────────────────────────
            if op == "describe":
                out["info"] = df.describe().to_string(); return out
            if op == "head":
                out["info"] = df.head(int(inputs.get("n", 5) or 5)).to_string(); return out
            if op == "info":
                dtypes = {col: str(dt) for col, dt in df.dtypes.items()}
                out["info"] = f"shape={df.shape} columns={list(df.columns)} dtypes={dtypes}"
                return out
            if op == "shape":
                out["rows"] = int(df.shape[0]); out["cols"] = int(df.shape[1])
                out["info"] = f"{out['rows']} x {out['cols']}"
                return out

            # ── extract (df → series / array / X+y) ─────────────────────────
            if op == "get_column":
                col = inputs.get("column") or "col_0"
                out["series"] = df[col]; return out
            if op == "to_numpy":
                out["array"] = df.values; return out
            if op == "xy_split":
                lbl = inputs.get("label_col") or "label"
                out["result"] = df.drop(columns=[lbl])
                out["series"] = df[lbl]
                return out

            return out
        except Exception:
            return self._empty()

    def export(self, iv, ov):
        df = self._val(iv, "df")
        op = (self.inputs["op"].default_value or "dropna")
        out = ov.get("result", "_pd_result")
        ser = ov.get("series", "_pd_series")
        arr = ov.get("array",  "_pd_array")
        info = ov.get("info",  "_pd_info")
        rows = ov.get("rows",  "_pd_rows")
        cols = ov.get("cols",  "_pd_cols")

        if op == "dropna":      return [], [f"{out} = {df}.dropna()"]
        if op == "fillna":      return [], [f"{out} = {df}.fillna({self._val(iv, 'value')})"]
        if op == "drop_cols":
            c = self._val(iv, "columns")
            return [], [f"_cols = [c.strip() for c in {c}.split(',') if c.strip()]",
                        f"{out} = {df}.drop(columns=_cols)"]
        if op == "select_cols":
            c = self._val(iv, "columns")
            return [], [f"_cols = [c.strip() for c in {c}.split(',') if c.strip()]",
                        f"{out} = {df}[_cols]"]
        if op == "rename_col":
            old = self._val(iv, "old_name"); new = self._val(iv, "new_name")
            return [], [f"{out} = {df}.rename(columns={{{old}: {new}}})"]
        if op == "reset_index": return [], [f"{out} = {df}.reset_index(drop=True)"]
        if op == "normalize":
            return [], [f"_num = {df}.select_dtypes(include='number')",
                        f"_rng = _num.max() - _num.min(); _rng[_rng == 0] = 1",
                        f"{out} = {df}.copy()",
                        f"{out}[_num.columns] = (_num - _num.min()) / _rng"]
        if op == "sort":
            by = self._val(iv, "by"); asc = self._val(iv, "ascending")
            return [], [f"{out} = {df}.sort_values(by={by}, ascending={asc})"]
        if op == "filter_rows":
            col = self._val(iv, "column"); cmp = self._val(iv, "compare"); val = self._val(iv, "value")
            return [], [
                f"_cmp = {{'==': lambda a,b: a==b, '!=': lambda a,b: a!=b, "
                f"'>': lambda a,b: a>b, '<': lambda a,b: a<b, "
                f"'>=': lambda a,b: a>=b, '<=': lambda a,b: a<=b}}[{cmp}]",
                f"{out} = {df}[_cmp({df}[{col}], {val})]",
            ]
        if op == "correlation": return [], [f"{out} = {df}.select_dtypes(include='number').corr()"]
        if op == "groupby":
            by = self._val(iv, "by"); agg = self._val(iv, "agg")
            return [], [f"{out} = {df}.groupby({by}).agg({agg}).reset_index()"]
        if op == "describe": return [], [f"{info} = {df}.describe().to_string()"]
        if op == "head":     return [], [f"{info} = {df}.head({self._val(iv, 'n')}).to_string()"]
        if op == "info":
            return [], [
                f"{info} = f\"shape={{{df}.shape}} columns={{list({df}.columns)}} \""
                f"f\"dtypes={{{{c: str(t) for c, t in {df}.dtypes.items()}}}}\""
            ]
        if op == "shape":
            return [], [f"{rows} = {df}.shape[0]", f"{cols} = {df}.shape[1]",
                        f"{info} = f\"{{{rows}}} x {{{cols}}}\""]
        if op == "get_column":
            return [], [f"{ser} = {df}[{self._val(iv, 'column')}]"]
        if op == "to_numpy":   return [], [f"{arr} = {df}.values"]
        if op == "xy_split":
            lbl = self._val(iv, "label_col")
            return [], [f"{ser} = {df}[{lbl}]", f"{out} = {df}.drop(columns=[{lbl}])"]
        return [], [f"# unknown pd op {op!r}"]
