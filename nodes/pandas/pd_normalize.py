"""(df - df.min()) / (df.max() - df.min()) on numeric cols node."""
from core.node import BaseNode, PortType


class PdNormalizeNode(BaseNode):
    type_name = "pd_normalize"
    label = "Normalize"
    category = "Pandas"
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

    def export(self, iv, ov):
        df = self._val(iv, "df")
        return (
            [],
            [
                f"_num = {df}.select_dtypes(include='number')",
                f"_rng = _num.max() - _num.min(); _rng[_rng == 0] = 1",
                f"{ov['result']} = {df}.copy()",
                f"{ov['result']}[_num.columns] = (_num - _num.min()) / _rng",
            ],
        )
