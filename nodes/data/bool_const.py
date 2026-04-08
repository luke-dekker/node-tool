"""BoolConstNode — outputs a constant bool."""
from core.node import BaseNode, PortType


class BoolConstNode(BaseNode):
    type_name = "bool_const"
    label = "Bool Const"
    category = "Python"
    subcategory = "Data"
    description = "Outputs a constant boolean value."

    def _setup_ports(self) -> None:
        self.add_input("Value", PortType.BOOL, False)
        self.add_output("Value", PortType.BOOL)

    def execute(self, inputs):
        return {"Value": bool(inputs["Value"])}

    def export(self, iv, ov):
        return [], [f"{ov['Value']} = bool({self._val(iv, 'Value')})"]
