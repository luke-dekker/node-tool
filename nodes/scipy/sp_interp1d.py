from core.node import BaseNode, PortType

class SpInterp1dNode(BaseNode):
    type_name = "sp_interp1d"; label = "Interp1d"; category = "SciPy"
    description = "scipy.interpolate.interp1d(x, y, kind) evaluated at x_new"
    def _setup_ports(self):
        self.add_input("x", PortType.NDARRAY)
        self.add_input("y", PortType.NDARRAY)
        self.add_input("x_new", PortType.NDARRAY)
        self.add_input("kind", PortType.STRING, "linear")
        self.add_output("y_new", PortType.NDARRAY)
    def execute(self, inputs):
        null = {"y_new": None}
        try:
            from scipy.interpolate import interp1d
            x, y, xn = inputs.get("x"), inputs.get("y"), inputs.get("x_new")
            if any(v is None for v in [x, y, xn]): return null
            f = interp1d(x.flatten(), y.flatten(), kind=inputs.get("kind","linear") or "linear",
                         bounds_error=False, fill_value="extrapolate")
            return {"y_new": f(xn.flatten())}
        except Exception: return null
    def export(self, iv, ov):
        x, y, xn, k = self._val(iv,"x"), self._val(iv,"y"), self._val(iv,"x_new"), self._val(iv,"kind")
        return ["from scipy.interpolate import interp1d"], [
            f"_f_interp = interp1d({x}.flatten(), {y}.flatten(), kind={k}, bounds_error=False, fill_value='extrapolate')",
            f"{ov['y_new']} = _f_interp({xn}.flatten())",
        ]
