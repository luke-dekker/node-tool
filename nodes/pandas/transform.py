"""Pandas DataFrame transformation nodes."""
from core.node import BaseNode, PortType

C = "Pandas"


class PdDropNaNode(BaseNode):
    type_name = "pd_dropna"
    label = "Drop NA"
    category = C
    description = "df.dropna()"

    def _setup_ports(self):
        self.add_input("df",      PortType.DATAFRAME)
        self.add_output("result", PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            df = inputs.get("df")
            return {"result": df.dropna() if df is not None else None}
        except Exception:
            return {"result": None}


class PdFillNaNode(BaseNode):
    type_name = "pd_fillna"
    label = "Fill NA"
    category = C
    description = "df.fillna(value)"

    def _setup_ports(self):
        self.add_input("df",    PortType.DATAFRAME)
        self.add_input("value", PortType.FLOAT, 0.0)
        self.add_output("result", PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            df = inputs.get("df")
            return {"result": df.fillna(float(inputs.get("value", 0.0))) if df is not None else None}
        except Exception:
            return {"result": None}


class PdSortNode(BaseNode):
    type_name = "pd_sort"
    label = "Sort"
    category = C
    description = "df.sort_values(by, ascending)"

    def _setup_ports(self):
        self.add_input("df",        PortType.DATAFRAME)
        self.add_input("by",        PortType.STRING, "col_0")
        self.add_input("ascending", PortType.BOOL,   True)
        self.add_output("result",   PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            df = inputs.get("df")
            if df is None:
                return {"result": None}
            return {"result": df.sort_values(by=inputs.get("by") or "col_0",
                                             ascending=bool(inputs.get("ascending", True)))}
        except Exception:
            return {"result": None}


class PdResetIndexNode(BaseNode):
    type_name = "pd_reset_index"
    label = "Reset Index"
    category = C
    description = "df.reset_index(drop=True)"

    def _setup_ports(self):
        self.add_input("df",      PortType.DATAFRAME)
        self.add_output("result", PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            df = inputs.get("df")
            return {"result": df.reset_index(drop=True) if df is not None else None}
        except Exception:
            return {"result": None}


class PdRenameColNode(BaseNode):
    type_name = "pd_rename_col"
    label = "Rename Column"
    category = C
    description = "Rename a single column old_name -> new_name"

    def _setup_ports(self):
        self.add_input("df",       PortType.DATAFRAME)
        self.add_input("old_name", PortType.STRING, "col_0")
        self.add_input("new_name", PortType.STRING, "x")
        self.add_output("result",  PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            df = inputs.get("df")
            if df is None:
                return {"result": None}
            return {"result": df.rename(columns={inputs.get("old_name") or "col_0":
                                                  inputs.get("new_name") or "x"})}
        except Exception:
            return {"result": None}


class PdToNumpyNode(BaseNode):
    type_name = "pd_to_numpy"
    label = "To NumPy"
    category = C
    description = "df.values — convert DataFrame to ndarray"

    def _setup_ports(self):
        self.add_input("df",     PortType.DATAFRAME)
        self.add_output("array", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            df = inputs.get("df")
            return {"array": df.values if df is not None else None}
        except Exception:
            return {"array": None}


class PdNormalizeNode(BaseNode):
    type_name = "pd_normalize"
    label = "Normalize"
    category = C
    description = "(df - df.min()) / (df.max() - df.min()) on numeric cols"

    def _setup_ports(self):
        self.add_input("df",      PortType.DATAFRAME)
        self.add_output("result", PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            df = inputs.get("df")
            if df is None:
                return {"result": None}
            num = df.select_dtypes(include="number")
            rng = num.max() - num.min()
            rng[rng == 0] = 1
            out = df.copy()
            out[num.columns] = (num - num.min()) / rng
            return {"result": out}
        except Exception:
            return {"result": None}
