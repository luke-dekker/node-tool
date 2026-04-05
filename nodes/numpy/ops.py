"""NumPy array manipulation nodes."""
import numpy as np
from core.node import BaseNode, PortType

C = "NumPy"


class NpReshapeNode(BaseNode):
    type_name = "np_reshape"
    label = "Reshape"
    category = C
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


class NpTransposeNode(BaseNode):
    type_name = "np_transpose"
    label = "Transpose"
    category = C
    description = "np.transpose(array)"

    def _setup_ports(self):
        self.add_input("array", PortType.NDARRAY)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            arr = inputs.get("array")
            return {"result": np.transpose(arr) if arr is not None else None}
        except Exception:
            return {"result": None}


class NpFlattenNode(BaseNode):
    type_name = "np_flatten"
    label = "Flatten"
    category = C
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


class NpConcatNode(BaseNode):
    type_name = "np_concat"
    label = "Concatenate"
    category = C
    description = "np.concatenate([a, b], axis=axis)"

    def _setup_ports(self):
        self.add_input("a",    PortType.NDARRAY)
        self.add_input("b",    PortType.NDARRAY)
        self.add_input("axis", PortType.INT, 0)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            a, b = inputs.get("a"), inputs.get("b")
            if a is None or b is None:
                return {"result": None}
            return {"result": np.concatenate([a, b], axis=int(inputs.get("axis", 0)))}
        except Exception:
            return {"result": None}


class NpStackNode(BaseNode):
    type_name = "np_stack"
    label = "Stack"
    category = C
    description = "np.stack([a, b], axis=axis)"

    def _setup_ports(self):
        self.add_input("a",    PortType.NDARRAY)
        self.add_input("b",    PortType.NDARRAY)
        self.add_input("axis", PortType.INT, 0)
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            a, b = inputs.get("a"), inputs.get("b")
            if a is None or b is None:
                return {"result": None}
            return {"result": np.stack([a, b], axis=int(inputs.get("axis", 0)))}
        except Exception:
            return {"result": None}


class NpSliceNode(BaseNode):
    type_name = "np_slice"
    label = "Slice"
    category = C
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


class NpWhereNode(BaseNode):
    type_name = "np_where"
    label = "Where"
    category = C
    description = "np.where(condition>0, x, y)"

    def _setup_ports(self):
        self.add_input("condition", PortType.NDARRAY)
        self.add_input("x",         PortType.NDARRAY)
        self.add_input("y",         PortType.NDARRAY)
        self.add_output("result",   PortType.NDARRAY)

    def execute(self, inputs):
        try:
            cond, x, y = inputs.get("condition"), inputs.get("x"), inputs.get("y")
            if any(v is None for v in [cond, x, y]):
                return {"result": None}
            return {"result": np.where(cond > 0, x, y)}
        except Exception:
            return {"result": None}
