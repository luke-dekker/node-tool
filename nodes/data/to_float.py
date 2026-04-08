"""ToFloatNode — coerce any value to float."""
from core.node import BaseNode, PortType


class ToFloatNode(BaseNode):
    type_name = "to_float"
    label = "To Float"
    category = "Python"
    subcategory = "Data"
    description = "Converts Value to float (returns 0.0 on failure)."

    def _setup_ports(self) -> None:
        self.add_input("Value", PortType.ANY, 0)
        self.add_output("Result", PortType.FLOAT)

    def execute(self, inputs):
        try:
            return {"Result": float(inputs["Value"])}
        except (ValueError, TypeError):
            return {"Result": 0.0}

    def export(self, iv, ov):
        v = self._val(iv, "Value")
        return [], [
            f"try: {ov['Result']} = float({v})",
            f"except (ValueError, TypeError): {ov['Result']} = 0.0",
        ]
