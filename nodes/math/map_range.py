"""MapRangeNode — remap a value from one numeric range to another."""
from core.node import BaseNode, PortType


class MapRangeNode(BaseNode):
    type_name = "map_range"
    label = "Map Range"
    category = "Python"
    subcategory = "Math"
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

    def export(self, iv, ov):
        v  = self._val(iv,"Value"); i0 = self._val(iv,"InMin"); i1 = self._val(iv,"InMax")
        o0 = self._val(iv,"OutMin"); o1 = self._val(iv,"OutMax")
        return [], [
            f"_denom = {i1} - {i0}",
            f"{ov['Result']} = {o0} + ({v} - {i0}) / _denom * ({o1} - {o0}) if _denom != 0 else {o0}",
        ]
