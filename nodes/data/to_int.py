"""ToIntNode — coerce any value to int."""
from core.node import BaseNode, PortType


class ToIntNode(BaseNode):
    type_name = "to_int"
    label = "To Int"
    category = "Python"
    subcategory = "Data"
    description = "Converts Value to integer (returns 0 on failure)."

    def _setup_ports(self) -> None:
        self.add_input("Value", PortType.ANY, 0)
        self.add_output("Result", PortType.INT)

    def execute(self, inputs):
        try:
            return {"Result": int(float(inputs["Value"]))}
        except (ValueError, TypeError):
            return {"Result": 0}

    def export(self, iv, ov):
        v = self._val(iv, "Value")
        return [], [
            f"try: {ov['Result']} = int(float({v}))",
            f"except (ValueError, TypeError): {ov['Result']} = 0",
        ]
