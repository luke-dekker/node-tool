"""Consolidated interpolation / curve-fit node.

Replaces SpInterp1dNode and SpCurveFitNode with a single mode-dispatched node.

Outputs:
  result  — np.ndarray  (interp1d → y_new, curve_fit → param array)
  info    — str         (curve_fit → 'a=…, b=…' summary, interp1d → '')
"""
from __future__ import annotations
from core.node import BaseNode, PortType


_OPS = ["interp1d", "curve_fit"]


class SpFitNode(BaseNode):
    type_name   = "sp_fit"
    label       = "Interp / Fit"
    category    = "SciPy"
    description = (
        "Pick `op`:\n"
        "  interp1d   — fit interp1d(x, y, kind) and evaluate at `x_new`\n"
        "  curve_fit  — fit `func_str` (uses params a..d) to (x, y); result=params"
    )

    def _setup_ports(self):
        self.add_input("op",       PortType.STRING, "interp1d", choices=_OPS)
        self.add_input("x",        PortType.NDARRAY)
        self.add_input("y",        PortType.NDARRAY)
        self.add_input("x_new",    PortType.NDARRAY, optional=True)
        self.add_input("kind",     PortType.STRING, "linear",
                       choices=["linear", "nearest", "cubic", "quadratic"], optional=True)
        self.add_input("func_str", PortType.STRING, "a*x+b", optional=True)
        self.add_output("result", PortType.NDARRAY)
        self.add_output("info",   PortType.STRING)

    def relevant_inputs(self, values):
        op = (values.get("op") or "interp1d").strip()
        if op == "interp1d":  return ["op", "kind"]      # x, y, x_new wired
        if op == "curve_fit": return ["op", "func_str"]
        return ["op"]

    def execute(self, inputs):
        out = {"result": None, "info": ""}
        try:
            x, y = inputs.get("x"), inputs.get("y")
            if x is None or y is None:
                return out
            op = (inputs.get("op") or "interp1d").strip()
            if op == "interp1d":
                from scipy.interpolate import interp1d
                xn = inputs.get("x_new")
                if xn is None:
                    return out
                f = interp1d(x.flatten(), y.flatten(),
                             kind=inputs.get("kind", "linear") or "linear",
                             bounds_error=False, fill_value="extrapolate")
                out["result"] = f(xn.flatten())
                return out
            if op == "curve_fit":
                import numpy as np
                from scipy.optimize import curve_fit
                func_str = inputs.get("func_str") or "a*x+b"
                pnames = [p for p in ["a", "b", "c", "d"] if p in func_str]
                if not pnames:
                    return out | {"info": "no params (a,b,c,d) found in func_str"}
                fn = eval(f"lambda x,{','.join(pnames)}: {func_str}",
                          {"__builtins__": {}}, {"np": np})
                popt, _ = curve_fit(fn, x.flatten(), y.flatten(), maxfev=5000)
                out["result"] = np.array(popt)
                out["info"]   = ", ".join(f"{n}={v:.6g}" for n, v in zip(pnames, popt))
                return out
            return out
        except Exception as e:
            return out | {"info": str(e)}

    def export(self, iv, ov):
        op = (self.inputs["op"].default_value or "interp1d")
        x, y = self._val(iv, "x"), self._val(iv, "y")
        if op == "interp1d":
            xn, k = self._val(iv, "x_new"), self._val(iv, "kind")
            return ["from scipy.interpolate import interp1d"], [
                f"_f = interp1d({x}.flatten(), {y}.flatten(), kind={k}, bounds_error=False, fill_value='extrapolate')",
                f"{ov['result']} = _f({xn}.flatten())",
            ]
        if op == "curve_fit":
            fs = self._val(iv, "func_str")
            return ["import numpy as np", "from scipy.optimize import curve_fit"], [
                f"_pn = [p for p in ['a','b','c','d'] if p in {fs}]",
                f'_fn = eval(f"lambda x,{{chr(44).join(_pn)}}: {{{fs}}}")',
                f"{ov['result']}, _ = curve_fit(_fn, {x}.flatten(), {y}.flatten(), maxfev=5000)",
                f"{ov['info']} = ', '.join(f'{{n}}={{v:.6g}}' for n,v in zip(_pn, {ov['result']}))",
            ]
        return [], [f"# unknown op {op!r}"]
