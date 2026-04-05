"""NpArrayFuncNode — single-array element-wise ops via Op dropdown."""
import numpy as np
from core.node import BaseNode, PortType

_OPS = {
    "abs":       lambda a: np.abs(a),
    "sqrt":      lambda a: np.sqrt(a),
    "log":       lambda a: np.log(np.clip(a, 1e-12, None)),
    "exp":       lambda a: np.exp(a),
    "transpose": lambda a: np.transpose(a),
    "flatten":   lambda a: a.flatten(),
    "normalize": lambda a: (a - a.min()) / (a.max() - a.min()) if a.max() != a.min()
                           else np.zeros_like(a, dtype=float),
    "sign":      lambda a: np.sign(a),
    "cumsum":    lambda a: np.cumsum(a),
    "diff":      lambda a: np.diff(a),
}


class NpArrayFuncNode(BaseNode):
    type_name = "np_array_func"
    label = "Array Func"
    category = "NumPy"
    description = "Single-array operations. Op dropdown selects the function."

    def _setup_ports(self):
        self.add_input("array", PortType.NDARRAY)
        self.add_input("Op",    PortType.STRING, "abs", choices=list(_OPS))
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            arr = inputs.get("array")
            if arr is None:
                return {"result": None}
            op = str(inputs.get("Op") or "abs").strip().lower()
            fn = _OPS.get(op, _OPS["abs"])
            return {"result": fn(arr)}
        except Exception:
            return {"result": None}
