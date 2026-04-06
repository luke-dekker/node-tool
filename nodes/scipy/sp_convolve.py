from core.node import BaseNode, PortType

class SpConvolveNode(BaseNode):
    type_name = "sp_convolve"; label = "Convolve"; category = "SciPy"
    description = "np.convolve(a, b, mode)"
    def _setup_ports(self):
        self.add_input("a", PortType.NDARRAY)
        self.add_input("b", PortType.NDARRAY)
        self.add_input("mode", PortType.STRING, "full")
        self.add_output("result", PortType.NDARRAY)
    def execute(self, inputs):
        null = {"result": None}
        try:
            import numpy as np
            a, b = inputs.get("a"), inputs.get("b")
            if a is None or b is None: return null
            return {"result": np.convolve(a.flatten(), b.flatten(), mode=inputs.get("mode","full") or "full")}
        except Exception: return null
    def export(self, iv, ov):
        a, b, mode = self._val(iv,"a"), self._val(iv,"b"), self._val(iv,"mode")
        return ["import numpy as np"], [f"{ov['result']} = np.convolve({a}.flatten(), {b}.flatten(), mode={mode})"]
