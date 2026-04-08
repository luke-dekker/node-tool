"""FloatConstNode — outputs a constant float."""
from core.node import BaseNode, PortType


class FloatConstNode(BaseNode):
    type_name = "float_const"
    label = "Float Const"
    category = "Python"
    subcategory = "Data"
    description = "Outputs a constant float value."

    def _setup_ports(self) -> None:
        self.add_input("Value", PortType.FLOAT, 0.0)
        self.add_output("Value", PortType.FLOAT)

    def execute(self, inputs):
        return {"Value": float(inputs["Value"])}

    def export(self, iv, ov):
        return [], [f"{ov['Value']} = float({self._val(iv, 'Value')})"]
