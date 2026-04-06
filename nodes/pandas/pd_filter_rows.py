"""Filter rows: column op value node."""
import operator as _op
from core.node import BaseNode, PortType

_OPS = {"==": _op.eq, "!=": _op.ne, ">": _op.gt, "<": _op.lt, ">=": _op.ge, "<=": _op.le}


class PdFilterRowsNode(BaseNode):
    type_name = "pd_filter_rows"
    label = "Filter Rows"
    category = "Pandas"
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

    def export(self, iv, ov):
        df = self._val(iv, "df")
        col = self._val(iv, "column")
        op = self._val(iv, "op")
        val = self._val(iv, "value")
        return (
            [],
            [f"{ov['result']} = {df}[{df}[{col}].apply(lambda _x: eval(f'{{_x}} {{{op}}} {{{val}}}'))]"],
        )
