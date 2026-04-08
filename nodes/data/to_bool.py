"""ToBoolNode — coerce any value to bool."""
from core.node import BaseNode, PortType


class ToBoolNode(BaseNode):
    type_name = "to_bool"
    label = "To Bool"
    category = "Python"
    subcategory = "Data"
    description = "Converts Value to bool. Strings: '', '0', 'false', 'no', 'none' are False."

    def _setup_ports(self) -> None:
        self.add_input("Value", PortType.ANY, 0)
        self.add_output("Result", PortType.BOOL)

    def execute(self, inputs):
        v = inputs["Value"]
        if isinstance(v, str):
            return {"Result": v.lower() not in ("", "0", "false", "no", "none")}
        return {"Result": bool(v)}

    def export(self, iv, ov):
        v = self._val(iv, "Value")
        return [], [
            f"_v = {v}",
            f"{ov['Result']} = (_v.lower() not in ('', '0', 'false', 'no', 'none')) "
            f"if isinstance(_v, str) else bool(_v)",
        ]
