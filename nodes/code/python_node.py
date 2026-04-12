"""PythonNode — run arbitrary Python. Inputs a, b, c. Write: result = ..."""
import math as _math
import re as _re

try:
    import numpy as _np
except ImportError:
    _np = None
try:
    import pandas as _pd
except ImportError:
    _pd = None
try:
    import torch as _torch
except ImportError:
    _torch = None

from core.node import BaseNode, PortType

_GLOBALS = {k: v for k, v in
            {"math": _math, "re": _re, "np": _np, "pd": _pd, "torch": _torch}.items()
            if v is not None}


class PythonNode(BaseNode):
    type_name = "python"
    label = "Python"
    category = "Core"
    subcategory = ""
    description = "Run Python. Inputs: a, b, c (any type). Write: result = <expression>"

    def _setup_ports(self):
        self.add_input("a",    PortType.ANY,    None)
        self.add_input("b",    PortType.ANY,    None)
        self.add_input("c",    PortType.ANY,    None)
        self.add_input("code", PortType.STRING, "result = a")
        self.add_output("result", PortType.ANY)

    def execute(self, inputs):
        ns = {"a": inputs["a"], "b": inputs["b"], "c": inputs["c"], **_GLOBALS}
        try:
            exec(inputs.get("code") or "result = None", ns)
        except Exception:
            pass
        return {"result": ns.get("result")}

    def export(self, iv, ov):
        code = self.inputs["code"].default_value or "result = None"
        a = self._val(iv,"a"); b = self._val(iv,"b"); c = self._val(iv,"c")
        return [], [
            f"_py_ns = {{'a': {a}, 'b': {b}, 'c': {c}}}",
            f"import math as _m, re; _py_ns.update({{'math':_m,'re':re}})",
            f"exec({repr(code)}, _py_ns)",
            f"{ov['result']} = _py_ns.get('result')",
        ]
