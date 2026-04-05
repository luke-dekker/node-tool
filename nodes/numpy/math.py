"""NumPy element-wise math nodes."""
import numpy as np
from core.node import BaseNode, PortType

C = "NumPy"


class NpAbsNode(BaseNode):
    type_name = "np_abs"
    label = "Abs"
    category = C
    description = "np.abs(array)"

    def _setup_ports(self):
        self.add_input("array", PortType.NDARRAY)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            arr = inputs.get("array")
            return {"result": np.abs(arr) if arr is not None else None}
        except Exception:
            return {"result": None}


class NpSqrtNode(BaseNode):
    type_name = "np_sqrt"
    label = "Sqrt"
    category = C
    description = "np.sqrt(array)"

    def _setup_ports(self):
        self.add_input("array", PortType.NDARRAY)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            arr = inputs.get("array")
            return {"result": np.sqrt(arr) if arr is not None else None}
        except Exception:
            return {"result": None}


class NpLogNode(BaseNode):
    type_name = "np_log"
    label = "Log"
    category = C
    description = "np.log(array) — clipped to avoid -inf"

    def _setup_ports(self):
        self.add_input("array", PortType.NDARRAY)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            arr = inputs.get("array")
            return {"result": np.log(np.clip(arr, 1e-12, None)) if arr is not None else None}
        except Exception:
            return {"result": None}


class NpExpNode(BaseNode):
    type_name = "np_exp"
    label = "Exp"
    category = C
    description = "np.exp(array)"

    def _setup_ports(self):
        self.add_input("array", PortType.NDARRAY)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            arr = inputs.get("array")
            return {"result": np.exp(arr) if arr is not None else None}
        except Exception:
            return {"result": None}


class NpClipNode(BaseNode):
    type_name = "np_clip"
    label = "Clip"
    category = C
    description = "np.clip(array, a_min, a_max)"

    def _setup_ports(self):
        self.add_input("array", PortType.NDARRAY)
        self.add_input("a_min", PortType.FLOAT, 0.0)
        self.add_input("a_max", PortType.FLOAT, 1.0)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            arr = inputs.get("array")
            if arr is None:
                return {"result": None}
            return {"result": np.clip(arr, float(inputs["a_min"]), float(inputs["a_max"]))}
        except Exception:
            return {"result": None}


class NpNormalizeNode(BaseNode):
    type_name = "np_normalize"
    label = "Normalize"
    category = C
    description = "(x - min) / (max - min) per array"

    def _setup_ports(self):
        self.add_input("array", PortType.NDARRAY)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            arr = inputs.get("array")
            if arr is None:
                return {"result": None}
            mn, mx = arr.min(), arr.max()
            rng = mx - mn
            return {"result": (arr - mn) / rng if rng != 0 else np.zeros_like(arr, dtype=float)}
        except Exception:
            return {"result": None}
