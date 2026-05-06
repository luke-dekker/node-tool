"""Consolidated NumPy array-creation node.

Replaces eight per-kind nodes (arange, linspace, ones, zeros, eye, rand,
randn, from_list). Pick a kind from the dropdown — only the inputs that
kind reads are consulted, the rest are ignored.
"""
from __future__ import annotations
import numpy as np

from core.node import BaseNode, PortType


_KINDS = ["arange", "linspace", "ones", "zeros", "eye", "rand", "randn", "from_list"]


def _parse_shape(s: str, default: str = "3,4") -> tuple[int, ...]:
    s = (s or default).strip() or default
    return tuple(int(x) for x in s.split(",") if x.strip())


class NpCreateNode(BaseNode):
    type_name   = "np_create"
    label       = "Create Array"
    category    = "NumPy"
    subcategory = ""
    description = (
        "Create a NumPy array. Pick `kind`:\n"
        "  arange    — start, stop, step\n"
        "  linspace  — start, stop, num\n"
        "  ones      — shape (e.g. '3,4')\n"
        "  zeros     — shape\n"
        "  eye       — n (square identity)\n"
        "  rand      — shape, seed (uniform [0,1))\n"
        "  randn     — shape, seed (standard normal)\n"
        "  from_list — values (comma-separated), dtype"
    )

    def _setup_ports(self):
        self.add_input("kind",   PortType.STRING, default="zeros", choices=_KINDS)
        self.add_input("shape",  PortType.STRING, default="3,4",
                       description="Comma-separated dims (ones/zeros/rand/randn)")
        self.add_input("start",  PortType.FLOAT,  default=0.0,
                       description="(arange/linspace) range start")
        self.add_input("stop",   PortType.FLOAT,  default=10.0,
                       description="(arange/linspace) range stop")
        self.add_input("step",   PortType.FLOAT,  default=1.0,
                       description="(arange) range step")
        self.add_input("num",    PortType.INT,    default=50,
                       description="(linspace) sample count")
        self.add_input("n",      PortType.INT,    default=3,
                       description="(eye) identity matrix size")
        self.add_input("seed",   PortType.INT,    default=-1,
                       description="(rand/randn) RNG seed; <0 = unset")
        self.add_input("values", PortType.STRING, default="1,2,3,4",
                       description="(from_list) comma-separated numbers")
        self.add_input("dtype",  PortType.STRING, default="float32",
                       description="(from_list) numpy dtype string")
        self.add_output("array", PortType.NDARRAY)

    def execute(self, inputs):
        kind = (inputs.get("kind") or "zeros").strip().lower()
        try:
            if kind == "arange":
                return {"array": np.arange(
                    float(inputs.get("start") or 0.0),
                    float(inputs.get("stop")  or 10.0),
                    float(inputs.get("step")  or 1.0),
                )}
            if kind == "linspace":
                return {"array": np.linspace(
                    float(inputs.get("start") or 0.0),
                    float(inputs.get("stop")  or 1.0),
                    int(inputs.get("num")     or 50),
                )}
            if kind == "ones":
                return {"array": np.ones(_parse_shape(inputs.get("shape")))}
            if kind == "zeros":
                return {"array": np.zeros(_parse_shape(inputs.get("shape")))}
            if kind == "eye":
                return {"array": np.eye(int(inputs.get("n") or 3))}
            if kind in ("rand", "randn"):
                shape = _parse_shape(inputs.get("shape"))
                seed = inputs.get("seed", -1)
                if seed is not None and int(seed) >= 0:
                    np.random.seed(int(seed))
                fn = np.random.rand if kind == "rand" else np.random.randn
                return {"array": fn(*shape)}
            if kind == "from_list":
                vals  = [float(x) for x
                         in (inputs.get("values") or "").split(",") if x.strip()]
                dtype = inputs.get("dtype") or "float32"
                return {"array": np.array(vals, dtype=dtype)}
            return {"array": None}
        except Exception:
            return {"array": None}

    def export(self, iv, ov):
        kind = (self.inputs["kind"].default_value or "zeros").strip().lower()
        out  = ov.get("array", "_arr")
        imp  = ["import numpy as np"]
        if kind == "arange":
            return imp, [f"{out} = np.arange("
                         f"{self._val(iv, 'start')}, {self._val(iv, 'stop')}, "
                         f"{self._val(iv, 'step')})"]
        if kind == "linspace":
            return imp, [f"{out} = np.linspace("
                         f"{self._val(iv, 'start')}, {self._val(iv, 'stop')}, "
                         f"{self._val(iv, 'num')})"]
        if kind in ("ones", "zeros"):
            shape = self._val(iv, "shape")
            return imp, [
                f"{out} = np.{kind}(tuple(int(x) for x in {shape}.split(',') if x.strip()))"
            ]
        if kind == "eye":
            return imp, [f"{out} = np.eye({self._val(iv, 'n')})"]
        if kind in ("rand", "randn"):
            shape = self._val(iv, "shape")
            seed  = self._val(iv, "seed")
            return imp, [
                f"_shape = tuple(int(x) for x in {shape}.split(',') if x.strip())",
                f"if int({seed}) >= 0: np.random.seed(int({seed}))",
                f"{out} = np.random.{kind}(*_shape)",
            ]
        if kind == "from_list":
            vals  = self._val(iv, "values")
            dtype = self._val(iv, "dtype")
            return imp, [
                f"{out} = np.array([float(x) for x in {vals}.split(',') if x.strip()], dtype={dtype})"
            ]
        return imp, [f"{out} = None"]
