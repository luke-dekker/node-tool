"""MapRangeNode — remap a value from one numeric range to another."""
from core.node import BaseNode, PortType


class MapRangeNode(BaseNode):
    type_name = "map_range"
    label = "Map Range"
    category = "Math"
    description = "Re-maps Value from [InMin, InMax] to [OutMin, OutMax]."

    def _setup_ports(self):
        self.add_input("Value",  PortType.FLOAT, 0.5)
        self.add_input("InMin",  PortType.FLOAT, 0.0)
        self.add_input("InMax",  PortType.FLOAT, 1.0)
        self.add_input("OutMin", PortType.FLOAT, 0.0)
        self.add_input("OutMax", PortType.FLOAT, 100.0)
        self.add_output("Result", PortType.FLOAT)

    def execute(self, inputs):
        denom = inputs["InMax"] - inputs["InMin"]
        if denom == 0:
            return {"Result": inputs["OutMin"]}
        t = (inputs["Value"] - inputs["InMin"]) / denom
        return {"Result": inputs["OutMin"] + t * (inputs["OutMax"] - inputs["OutMin"])}
