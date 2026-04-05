"""NumPy reduction / statistics nodes."""
import numpy as np
from core.node import BaseNode, PortType

C = "NumPy"
_SENTINEL = -99  # axis sentinel meaning "all elements"


def _ax(inputs):
    v = inputs.get("axis", _SENTINEL)
    return None if (v is None or int(v) == _SENTINEL) else int(v)


class NpMeanNode(BaseNode):
    type_name = "np_mean"
    label = "Mean"
    category = C
    description = "np.mean. axis=-99 means over all elements."

    def _setup_ports(self):
        self.add_input("array",  PortType.NDARRAY)
        self.add_input("axis",   PortType.INT, _SENTINEL)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            arr = inputs.get("array")
            return {"result": np.mean(arr, axis=_ax(inputs)) if arr is not None else None}
        except Exception:
            return {"result": None}


class NpStdNode(BaseNode):
    type_name = "np_std"
    label = "Std Dev"
    category = C
    description = "np.std. axis=-99 means over all elements."

    def _setup_ports(self):
        self.add_input("array",  PortType.NDARRAY)
        self.add_input("axis",   PortType.INT, _SENTINEL)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            arr = inputs.get("array")
            return {"result": np.std(arr, axis=_ax(inputs)) if arr is not None else None}
        except Exception:
            return {"result": None}


class NpSumNode(BaseNode):
    type_name = "np_sum"
    label = "Sum"
    category = C
    description = "np.sum. axis=-99 means over all elements."

    def _setup_ports(self):
        self.add_input("array",  PortType.NDARRAY)
        self.add_input("axis",   PortType.INT, _SENTINEL)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            arr = inputs.get("array")
            return {"result": np.sum(arr, axis=_ax(inputs)) if arr is not None else None}
        except Exception:
            return {"result": None}


class NpMinNode(BaseNode):
    type_name = "np_min"
    label = "Min"
    category = C
    description = "np.min(array)"

    def _setup_ports(self):
        self.add_input("array", PortType.NDARRAY)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            arr = inputs.get("array")
            return {"result": np.min(arr) if arr is not None else None}
        except Exception:
            return {"result": None}


class NpMaxNode(BaseNode):
    type_name = "np_max"
    label = "Max"
    category = C
    description = "np.max(array)"

    def _setup_ports(self):
        self.add_input("array", PortType.NDARRAY)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            arr = inputs.get("array")
            return {"result": np.max(arr) if arr is not None else None}
        except Exception:
            return {"result": None}
