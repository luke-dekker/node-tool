"""ToStringNode — coerce any value to string."""
from core.node import BaseNode, PortType


class ToStringNode(BaseNode):
    type_name = "to_string"
    label = "To String"
    category = "Python"
    subcategory = "Data"
    description = "Converts Value to string."

    def _setup_ports(self) -> None:
        self.add_input("Value", PortType.ANY, None)
        self.add_output("Result", PortType.STRING)

    def execute(self, inputs):
        return {"Result": str(inputs["Value"])}

    def export(self, iv, ov):
        return [], [f"{ov['Result']} = str({self._val(iv, 'Value')})"]
