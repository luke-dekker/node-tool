"""IntConstNode — outputs a constant int."""
from core.node import BaseNode, PortType


class IntConstNode(BaseNode):
    type_name = "int_const"
    label = "Int Const"
    category = "Python"
    subcategory = "Data"
    description = "Outputs a constant integer value."

    def _setup_ports(self) -> None:
        self.add_input("Value", PortType.INT, 0)
        self.add_output("Value", PortType.INT)

    def execute(self, inputs):
        return {"Value": int(inputs["Value"])}

    def export(self, iv, ov):
        return [], [f"{ov['Value']} = int({self._val(iv, 'Value')})"]
