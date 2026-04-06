from core.node import BaseNode, PortType

class SpHistogramNode(BaseNode):
    type_name = "sp_histogram"; label = "Histogram"; category = "SciPy"
    description = "np.histogram(array, bins) — counts and bin_edges"
    def _setup_ports(self):
        self.add_input("array", PortType.NDARRAY)
        self.add_input("bins", PortType.INT, 20)
        self.add_output("counts", PortType.NDARRAY)
        self.add_output("bin_edges", PortType.NDARRAY)
    def execute(self, inputs):
        null = {"counts": None, "bin_edges": None}
        try:
            import numpy as np
            arr = inputs.get("array")
            if arr is None: return null
            counts, edges = np.histogram(arr.flatten(), bins=int(inputs.get("bins", 20)))
            return {"counts": counts, "bin_edges": edges}
        except Exception: return null
    def export(self, iv, ov):
        return ["import numpy as np"], [
            f"{ov['counts']}, {ov['bin_edges']} = np.histogram({self._val(iv,'array')}.flatten(), bins={self._val(iv,'bins')})"]
