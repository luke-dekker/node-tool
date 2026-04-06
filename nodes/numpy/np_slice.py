"""array[start:end:step] along first axis node."""
import numpy as np
from core.node import BaseNode, PortType


class NpSliceNode(BaseNode):
    type_name = "np_slice"
    label = "Slice"
    category = "NumPy"
    description = "array[start:end:step] along first axis"

    def _setup_ports(self):
        self.add_input("array", PortType.NDARRAY)
        self.add_input("start", PortType.INT, 0)
        self.add_input("end",   PortType.INT, 10)
        self.add_input("step",  PortType.INT, 1)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            arr = inputs.get("array")
            if arr is None:
                return {"result": None}
            return {"result": arr[int(inputs["start"]):int(inputs["end"]):int(inputs["step"])]}
        except Exception:
            return {"result": None}

    def export(self, iv, ov):
        arr = self._val(iv, "array")
        s = self._val(iv, "start")
        e = self._val(iv, "end")
        st = self._val(iv, "step")
        return (
            [],
            [f"{ov['result']} = {arr}[{s}:{e}:{st}]"],
        )
