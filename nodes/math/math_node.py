"""MathNode — all common math operations in one node, selected by Op string."""
import math as _math
import random
from core.node import BaseNode, PortType

_OPS = {
    "add":      lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "multiply": lambda a, b: a * b,
    "divide":   lambda a, b: a / b if b != 0 else 0.0,
    "power":    lambda a, b: float(a ** b),
    "sqrt":     lambda a, b: _math.sqrt(max(0.0, a)),
    "abs":      lambda a, b: abs(a),
    "sin":      lambda a, b: _math.sin(_math.radians(a)),
    "cos":      lambda a, b: _math.cos(_math.radians(a)),
    "tan":      lambda a, b: _math.tan(_math.radians(a)),
    "round":    lambda a, b: float(round(a, int(b))),
    "random":   lambda a, b: random.uniform(min(a, b), max(a, b)),
}

_OP_LIST = ", ".join(_OPS)


class MathNode(BaseNode):
    type_name = "math"
    label = "Math"
    category = "Math"
    description = f"Math operations. Op: {_OP_LIST}. B unused for unary ops (sqrt, abs, sin, cos, tan)."

    def _setup_ports(self):
        self.add_input("A",  PortType.FLOAT,  0.0)
        self.add_input("B",  PortType.FLOAT,  1.0)
        self.add_input("Op", PortType.STRING, "add", choices=list(_OPS))
        self.add_output("Result", PortType.FLOAT)

    def execute(self, inputs):
        op = str(inputs.get("Op") or "add").strip().lower()
        fn = _OPS.get(op, _OPS["add"])
        try:
            return {"Result": float(fn(inputs["A"], inputs["B"]))}
        except Exception:
            return {"Result": 0.0}
