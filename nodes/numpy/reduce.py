"""NpReduceNode — reduction operations with optional axis via Op dropdown."""
import numpy as np
from core.node import BaseNode, PortType

_SENTINEL = -99  # axis value meaning "reduce over all elements"

_OPS = {
    "mean":    np.mean,
    "std":     np.std,
    "sum":     np.sum,
    "min":     np.min,
    "max":     np.max,
    "median":  np.median,
    "var":     np.var,
    "prod":    np.prod,
    "any":     np.any,
    "all":     np.all,
}


class NpReduceNode(BaseNode):
    type_name = "np_reduce"
    label = "Reduce"
    category = "NumPy"
    description = "Reduction ops. Op dropdown selects function. axis=-99 reduces all elements."

    def _setup_ports(self):
        self.add_input("array",  PortType.NDARRAY)
        self.add_input("Op",     PortType.STRING, "mean", choices=list(_OPS))
        self.add_input("axis",   PortType.INT,    _SENTINEL)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            arr = inputs.get("array")
            if arr is None:
                return {"result": None}
            op  = str(inputs.get("Op") or "mean").strip().lower()
            fn  = _OPS.get(op, np.mean)
            ax  = inputs.get("axis", _SENTINEL)
            ax  = None if (ax is None or int(ax) == _SENTINEL) else int(ax)
            return {"result": fn(arr, axis=ax)}
        except Exception:
            return {"result": None}
