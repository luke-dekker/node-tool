"""Pandas column and row selection/filtering nodes."""
import operator as _op
from core.node import BaseNode, PortType

C = "Pandas"

_OPS = {"==": _op.eq, "!=": _op.ne, ">": _op.gt, "<": _op.lt, ">=": _op.ge, "<=": _op.le}


class PdSelectColsNode(BaseNode):
    type_name = "pd_select_cols"
    label = "Select Columns"
    category = C
    description = "df[col_list] — comma-separated column names"

    def _setup_ports(self):
        self.add_input("df",      PortType.DATAFRAME)
        self.add_input("columns", PortType.STRING, "col1,col2")
        self.add_output("result", PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            df = inputs.get("df")
            if df is None:
                return {"result": None}
            cols = [c.strip() for c in (inputs.get("columns") or "").split(",") if c.strip()]
            return {"result": df[cols]}
        except Exception:
            return {"result": None}


class PdDropColsNode(BaseNode):
    type_name = "pd_drop_cols"
    label = "Drop Columns"
    category = C
    description = "df.drop(columns=[...]) — comma-separated"

    def _setup_ports(self):
        self.add_input("df",      PortType.DATAFRAME)
        self.add_input("columns", PortType.STRING, "col1")
        self.add_output("result", PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            df = inputs.get("df")
            if df is None:
                return {"result": None}
            cols = [c.strip() for c in (inputs.get("columns") or "").split(",") if c.strip()]
            return {"result": df.drop(columns=cols)}
        except Exception:
            return {"result": None}


class PdFilterRowsNode(BaseNode):
    type_name = "pd_filter_rows"
    label = "Filter Rows"
    category = C
    description = "Filter rows: column op value. op: ==, !=, >, <, >=, <="

    def _setup_ports(self):
        self.add_input("df",     PortType.DATAFRAME)
        self.add_input("column", PortType.STRING, "label")
        self.add_input("op",     PortType.STRING, "==")
        self.add_input("value",  PortType.FLOAT,  1.0)
        self.add_output("result", PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            df  = inputs.get("df")
            col = inputs.get("column") or "label"
            fn  = _OPS.get(inputs.get("op") or "==", _op.eq)
            val = inputs.get("value", 1.0)
            if df is None:
                return {"result": None}
            return {"result": df[fn(df[col], val)]}
        except Exception:
            return {"result": None}


class PdGetColumnNode(BaseNode):
    type_name = "pd_get_column"
    label = "Get Column"
    category = C
    description = "df[column] — returns a Series"

    def _setup_ports(self):
        self.add_input("df",     PortType.DATAFRAME)
        self.add_input("column", PortType.STRING, "col_0")
        self.add_output("series", PortType.SERIES)

    def execute(self, inputs):
        try:
            df  = inputs.get("df")
            col = inputs.get("column") or "col_0"
            return {"series": df[col] if df is not None else None}
        except Exception:
            return {"series": None}
