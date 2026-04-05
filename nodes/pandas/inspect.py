"""Pandas DataFrame inspection nodes."""
from core.node import BaseNode, PortType

C = "Pandas"


class PdHeadNode(BaseNode):
    type_name = "pd_head"
    label = "Head"
    category = C
    description = "df.head(n).to_string()"

    def _setup_ports(self):
        self.add_input("df",      PortType.DATAFRAME)
        self.add_input("n",       PortType.INT, 5)
        self.add_output("result", PortType.STRING)

    def execute(self, inputs):
        try:
            df = inputs.get("df")
            return {"result": df.head(int(inputs.get("n", 5))).to_string() if df is not None else "None"}
        except Exception:
            return {"result": "error"}


class PdDescribeNode(BaseNode):
    type_name = "pd_describe"
    label = "Describe"
    category = C
    description = "df.describe().to_string()"

    def _setup_ports(self):
        self.add_input("df",      PortType.DATAFRAME)
        self.add_output("result", PortType.STRING)

    def execute(self, inputs):
        try:
            df = inputs.get("df")
            return {"result": df.describe().to_string() if df is not None else "None"}
        except Exception:
            return {"result": "error"}


class PdInfoNode(BaseNode):
    type_name = "pd_info"
    label = "Info"
    category = C
    description = "shape, columns, dtypes summary"

    def _setup_ports(self):
        self.add_input("df",      PortType.DATAFRAME)
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
    category = C
    description = "Return df.shape as rows, cols"

    def _setup_ports(self):
        self.add_input("df",   PortType.DATAFRAME)
        self.add_output("rows", PortType.INT)
        self.add_output("cols", PortType.INT)

    def execute(self, inputs):
        try:
            df = inputs.get("df")
            return {"rows": df.shape[0], "cols": df.shape[1]} if df is not None else {"rows": 0, "cols": 0}
        except Exception:
            return {"rows": 0, "cols": 0}
