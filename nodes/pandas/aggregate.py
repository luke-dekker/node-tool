"""Pandas aggregation, merge, and split nodes."""
from core.node import BaseNode, PortType

C = "Pandas"


class PdGroupByNode(BaseNode):
    type_name = "pd_groupby"
    label = "Group By"
    category = C
    description = "df.groupby(by).agg(agg)"

    def _setup_ports(self):
        self.add_input("df",      PortType.DATAFRAME)
        self.add_input("by",      PortType.STRING, "label")
        self.add_input("agg",     PortType.STRING, "mean")
        self.add_output("result", PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            df = inputs.get("df")
            if df is None:
                return {"result": None}
            return {"result": df.groupby(inputs.get("by") or "label")
                               .agg(inputs.get("agg") or "mean")
                               .reset_index()}
        except Exception:
            return {"result": None}


class PdCorrelationNode(BaseNode):
    type_name = "pd_correlation"
    label = "Correlation"
    category = C
    description = "df.corr() — pairwise correlation matrix"

    def _setup_ports(self):
        self.add_input("df",      PortType.DATAFRAME)
        self.add_output("result", PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            df = inputs.get("df")
            return {"result": df.select_dtypes(include="number").corr() if df is not None else None}
        except Exception:
            return {"result": None}


class PdXYSplitNode(BaseNode):
    type_name = "pd_xy_split"
    label = "XY Split"
    category = C
    description = "Split DataFrame into features X and label series y"

    def _setup_ports(self):
        self.add_input("df",        PortType.DATAFRAME)
        self.add_input("label_col", PortType.STRING,   "label")
        self.add_output("X",        PortType.DATAFRAME)
        self.add_output("y",        PortType.SERIES)

    def execute(self, inputs):
        null = {"X": None, "y": None}
        try:
            df  = inputs.get("df")
            col = inputs.get("label_col") or "label"
            if df is None:
                return null
            return {"X": df.drop(columns=[col]), "y": df[col]}
        except Exception:
            return null


class PdMergeNode(BaseNode):
    type_name = "pd_merge"
    label = "Merge"
    category = C
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
            on    = inputs.get("on")  or "index"
            how   = inputs.get("how") or "inner"
            if left is None or right is None:
                return {"result": None}
            if on == "index":
                return {"result": pd.merge(left, right, left_index=True, right_index=True, how=how)}
            return {"result": pd.merge(left, right, on=on, how=how)}
        except Exception:
            return {"result": None}
