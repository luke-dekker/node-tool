"""LogicNode — boolean and comparison operations in one node, selected by Op string."""
from core.node import BaseNode, PortType

_OPS = {
    "and": lambda a, b: bool(a) and bool(b),
    "or":  lambda a, b: bool(a) or bool(b),
    "not": lambda a, b: not bool(a),
    "eq":  lambda a, b: a == b,
    "neq": lambda a, b: a != b,
    "lt":  lambda a, b: a < b,
    "lte": lambda a, b: a <= b,
    "gt":  lambda a, b: a > b,
    "gte": lambda a, b: a >= b,
}

_OP_LIST = ", ".join(_OPS)


class LogicNode(BaseNode):
    type_name = "logic"
    label = "Logic"
    category = "Logic"
    description = f"Logic and comparison operations. Op: {_OP_LIST}. B unused for 'not'."

    def _setup_ports(self):
        self.add_input("A",  PortType.ANY,    False)
        self.add_input("B",  PortType.ANY,    False)
        self.add_input("Op", PortType.STRING, "and", choices=list(_OPS))
        self.add_output("Result", PortType.BOOL)

    def execute(self, inputs):
        op = str(inputs.get("Op") or "and").strip().lower()
        fn = _OPS.get(op, _OPS["and"])
        try:
            return {"Result": fn(inputs["A"], inputs["B"])}
        except Exception:
            return {"Result": False}
