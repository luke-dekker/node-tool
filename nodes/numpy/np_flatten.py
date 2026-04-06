"""array.flatten() node."""
import numpy as np
from core.node import BaseNode, PortType


class NpFlattenNode(BaseNode):
    type_name = "np_flatten"
    label = "Flatten"
    category = "NumPy"
    description = "array.flatten()"

    def _setup_ports(self):
        self.add_input("array", PortType.NDARRAY)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            arr = inputs.get("array")
            return {"result": arr.flatten() if arr is not None else None}
        except Exception:
            return {"result": None}

    def export(self, iv, ov):
        return (
            [],
            [f"{ov['result']} = {self._val(iv, 'array')}.flatten()"],
        )
