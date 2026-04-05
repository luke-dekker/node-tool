"""NumPy array creation nodes."""
import numpy as np
from core.node import BaseNode, PortType

C = "NumPy"


def _parse_shape(s, default="3,4"):
    s = s or default
    return tuple(int(x) for x in s.split(",") if x.strip())


class NpArangeNode(BaseNode):
    type_name = "np_arange"
    label = "Arange"
    category = C
    description = "np.arange(start, stop, step)"

    def _setup_ports(self):
        self.add_input("start", PortType.FLOAT, 0.0)
        self.add_input("stop",  PortType.FLOAT, 10.0)
        self.add_input("step",  PortType.FLOAT, 1.0)
        self.add_output("array", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            start, stop, step = inputs["start"], inputs["stop"], inputs["step"]
            if any(v is None for v in [start, stop, step]):
                return {"array": None}
            return {"array": np.arange(float(start), float(stop), float(step))}
        except Exception:
            return {"array": None}


class NpLinspaceNode(BaseNode):
    type_name = "np_linspace"
    label = "Linspace"
    category = C
    description = "np.linspace(start, stop, num)"

    def _setup_ports(self):
        self.add_input("start", PortType.FLOAT, 0.0)
        self.add_input("stop",  PortType.FLOAT, 1.0)
        self.add_input("num",   PortType.INT,   50)
        self.add_output("array", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            start, stop, num = inputs["start"], inputs["stop"], inputs["num"]
            if any(v is None for v in [start, stop, num]):
                return {"array": None}
            return {"array": np.linspace(float(start), float(stop), int(num))}
        except Exception:
            return {"array": None}


class NpZerosNode(BaseNode):
    type_name = "np_zeros"
    label = "Zeros"
    category = C
    description = "np.zeros with shape string, e.g. '3,4'"

    def _setup_ports(self):
        self.add_input("shape", PortType.STRING, "3,4")
        self.add_output("array", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            return {"array": np.zeros(_parse_shape(inputs.get("shape", "3,4")))}
        except Exception:
            return {"array": None}


class NpOnesNode(BaseNode):
    type_name = "np_ones"
    label = "Ones"
    category = C
    description = "np.ones with shape string, e.g. '3,4'"

    def _setup_ports(self):
        self.add_input("shape", PortType.STRING, "3,4")
        self.add_output("array", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            return {"array": np.ones(_parse_shape(inputs.get("shape", "3,4")))}
        except Exception:
            return {"array": None}


class NpRandNode(BaseNode):
    type_name = "np_rand"
    label = "Random Uniform"
    category = C
    description = "np.random.rand — shape string, seed>=0 to fix"

    def _setup_ports(self):
        self.add_input("shape", PortType.STRING, "3,4")
        self.add_input("seed",  PortType.INT,    -1)
        self.add_output("array", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            shape = _parse_shape(inputs.get("shape", "3,4"))
            seed  = inputs.get("seed", -1)
            if seed is not None and int(seed) >= 0:
                np.random.seed(int(seed))
            return {"array": np.random.rand(*shape)}
        except Exception:
            return {"array": None}


class NpRandnNode(BaseNode):
    type_name = "np_randn"
    label = "Random Normal"
    category = C
    description = "np.random.randn — shape string, seed>=0 to fix"

    def _setup_ports(self):
        self.add_input("shape", PortType.STRING, "3,4")
        self.add_input("seed",  PortType.INT,    -1)
        self.add_output("array", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            shape = _parse_shape(inputs.get("shape", "3,4"))
            seed  = inputs.get("seed", -1)
            if seed is not None and int(seed) >= 0:
                np.random.seed(int(seed))
            return {"array": np.random.randn(*shape)}
        except Exception:
            return {"array": None}


class NpFromListNode(BaseNode):
    type_name = "np_from_list"
    label = "From List"
    category = C
    description = "Parse comma-separated values string into ndarray."

    def _setup_ports(self):
        self.add_input("values", PortType.STRING, "1,2,3,4")
        self.add_input("dtype",  PortType.STRING, "float32")
        self.add_output("array", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            vals  = [float(x) for x in (inputs.get("values") or "").split(",") if x.strip()]
            dtype = inputs.get("dtype") or "float32"
            return {"array": np.array(vals, dtype=dtype)}
        except Exception:
            return {"array": None}


class NpEyeNode(BaseNode):
    type_name = "np_eye"
    label = "Eye"
    category = C
    description = "np.eye(n) — identity matrix"

    def _setup_ports(self):
        self.add_input("n", PortType.INT, 3)
        self.add_output("array", PortType.NDARRAY)

    def execute(self, inputs):
        try:
            n = inputs.get("n")
            return {"array": np.eye(int(n)) if n is not None else None}
        except Exception:
            return {"array": None}
