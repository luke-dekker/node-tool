"""StringConstNode — outputs a constant string."""
from core.node import BaseNode, PortType


class StringConstNode(BaseNode):
    type_name = "string_const"
    label = "String Const"
    category = "Python"
    subcategory = "Data"
    description = "Outputs a constant string value."

    def _setup_ports(self) -> None:
        self.add_input("Value", PortType.STRING, "")
        self.add_output("Value", PortType.STRING)

    def execute(self, inputs):
        return {"Value": str(inputs["Value"])}

    def export(self, iv, ov):
        return [], [f"{ov['Value']} = str({self._val(iv, 'Value')})"]
