"""Reshape array to shape string node."""
import numpy as np
from core.node import BaseNode, PortType


class NpReshapeNode(BaseNode):
    type_name = "np_reshape"
    label = "Reshape"
    category = "NumPy"
    description = "Reshape array to shape string, e.g. '2,3'"

    def _setup_ports(self):
        self.add_input("array", PortType.NDARRAY)
        self.add_input("shape", PortType.STRING, "2,3")
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            arr = inputs.get("array")
            if arr is None:
                return {"result": None}
            shape = tuple(int(x) for x in (inputs.get("shape") or "2,3").split(",") if x.strip())
            return {"result": arr.reshape(shape)}
        except Exception:
            return {"result": None}

    def export(self, iv, ov):
        arr = self._val(iv, "array")
        shape = self._val(iv, "shape")
        return (
            ["import numpy as np"],
            [f"{ov['result']} = {arr}.reshape(tuple(int(x) for x in {shape}.split(',') if x.strip()))"],
        )
