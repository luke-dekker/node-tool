"""ClampNode — clamp a value between min and max."""
from core.node import BaseNode, PortType


class ClampNode(BaseNode):
    type_name = "clamp"
    label = "Clamp"
    category = "Math"
    description = "Clamps Value between Min and Max."

    def _setup_ports(self):
        self.add_input("Value", PortType.FLOAT, 0.0)
        self.add_input("Min", PortType.FLOAT, 0.0)
        self.add_input("Max", PortType.FLOAT, 1.0)
        self.add_output("Result", PortType.FLOAT)

    def execute(self, inputs):
        lo, hi = inputs["Min"], inputs["Max"]
        if lo > hi:
            lo, hi = hi, lo
        return {"Result": max(lo, min(hi, inputs["Value"]))}
