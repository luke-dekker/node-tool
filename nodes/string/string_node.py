"""StringNode — common string operations in one node, selected by Op string."""
from core.node import BaseNode, PortType

_OPS = {
    "upper":    lambda a, b: str(a).upper(),
    "lower":    lambda a, b: str(a).lower(),
    "strip":    lambda a, b: str(a).strip(),
    "reverse":  lambda a, b: str(a)[::-1],
    "length":   lambda a, b: len(str(a)),
    "concat":   lambda a, b: str(a) + str(b),
    "contains": lambda a, b: str(b) in str(a),
    "repeat":   lambda a, b: str(a) * max(0, int(float(b))),
}

_OP_LIST = ", ".join(_OPS)


class StringNode(BaseNode):
    type_name = "string_op"
    label = "String"
    category = "String"
    description = f"String operations. Op: {_OP_LIST}. B used by concat/contains/repeat."

    def _setup_ports(self):
        self.add_input("A",  PortType.STRING, "")
        self.add_input("B",  PortType.STRING, "")
        self.add_input("Op", PortType.STRING, "upper", choices=list(_OPS))
        self.add_output("Result", PortType.ANY)

    def execute(self, inputs):
        op = str(inputs.get("Op") or "upper").strip().lower()
        fn = _OPS.get(op, _OPS["upper"])
        try:
            return {"Result": fn(inputs["A"], inputs["B"])}
        except Exception:
            return {"Result": None}
