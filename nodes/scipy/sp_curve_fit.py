from core.node import BaseNode, PortType

class SpCurveFitNode(BaseNode):
    type_name = "sp_curve_fit"; label = "Curve Fit"; category = "SciPy"
    description = "scipy.optimize.curve_fit using func_str like 'a*x+b'"
    def _setup_ports(self):
        self.add_input("x", PortType.NDARRAY)
        self.add_input("y", PortType.NDARRAY)
        self.add_input("func_str", PortType.STRING, "a*x+b")
        self.add_output("params", PortType.NDARRAY)
        self.add_output("info", PortType.STRING)
    def execute(self, inputs):
        null = {"params": None, "info": "error"}
        try:
            import numpy as np; from scipy.optimize import curve_fit
            x, y = inputs.get("x"), inputs.get("y")
            func_str = inputs.get("func_str","a*x+b") or "a*x+b"
            if x is None or y is None: return null
            pnames = [p for p in ["a","b","c","d"] if p in func_str]
            if not pnames: return {"params": None, "info": "No params (a,b,c,d) found"}
            fn = eval(f"lambda x,{','.join(pnames)}: {func_str}", {"__builtins__":{}}, {"np":np})
            popt, _ = curve_fit(fn, x.flatten(), y.flatten(), maxfev=5000)
            return {"params": np.array(popt), "info": ", ".join(f"{n}={v:.6g}" for n,v in zip(pnames,popt))}
        except Exception as e: return {"params": None, "info": str(e)}
    def export(self, iv, ov):
        x, y, fs = self._val(iv,"x"), self._val(iv,"y"), self._val(iv,"func_str")
        p = ov['params']
        return ["import numpy as np", "from scipy.optimize import curve_fit"], [
            f"_param_names = [p for p in ['a','b','c','d'] if p in {fs}]",
            f'_fn = eval(f"lambda x,{{chr(44).join(_param_names)}}: {{{fs}}}")',
            f"{p}, _ = curve_fit(_fn, {x}.flatten(), {y}.flatten(), maxfev=5000)",
            f"{ov['info']} = ', '.join(f'{{n}}={{v:.6g}}' for n,v in zip(_param_names, {p}))",
        ]
