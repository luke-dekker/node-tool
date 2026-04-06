from core.node import BaseNode, PortType

class SpNormPdfNode(BaseNode):
    type_name = "sp_norm_pdf"; label = "Normal PDF"; category = "SciPy"
    description = "scipy.stats.norm.pdf(x, loc, scale)"
    def _setup_ports(self):
        self.add_input("x", PortType.NDARRAY)
        self.add_input("loc", PortType.FLOAT, 0.0)
        self.add_input("scale", PortType.FLOAT, 1.0)
        self.add_output("pdf", PortType.NDARRAY)
    def execute(self, inputs):
        try:
            from scipy.stats import norm
            x = inputs.get("x")
            if x is None: return {"pdf": None}
            return {"pdf": norm.pdf(x, loc=float(inputs.get("loc", 0.0)), scale=float(inputs.get("scale", 1.0)))}
        except Exception: return {"pdf": None}
    def export(self, iv, ov):
        return ["from scipy.stats import norm"], [
            f"{ov['pdf']} = norm.pdf({self._val(iv,'x')}, loc={self._val(iv,'loc')}, scale={self._val(iv,'scale')})"]
