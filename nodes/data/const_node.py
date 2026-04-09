"""ConstNode — single Const node with a Type dropdown.

Replaces FloatConstNode / IntConstNode / BoolConstNode / StringConstNode with
one node whose `Type` input chooses how the literal is interpreted. The four
legacy single-type const nodes are kept in the registry for loading old saves
but new graphs should use this consolidated form.

Why a separate STRING port for the value: keeping it as STRING means the user
types the literal once and the node coerces it to the chosen type at execute
time. Per-type input ports would require swapping the port out when Type
changes, which is more complex than the value it provides.
"""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


_TYPES = ["float", "int", "bool", "string"]
_TYPE_PORT = {
    "float":  PortType.FLOAT,
    "int":    PortType.INT,
    "bool":   PortType.BOOL,
    "string": PortType.STRING,
}


class ConstNode(BaseNode):
    type_name   = "const"
    label       = "Const"
    category    = "Python"
    subcategory = "Data"
    description = (
        "Constant literal with a Type dropdown. Type=float/int/bool/string "
        "controls how the Value string is parsed. Output is ANY-typed so it "
        "can wire into any port; downstream PortType.coerce() handles the "
        "conversion. Replaces the four single-type const nodes."
    )

    def _setup_ports(self) -> None:
        self.add_input("Type",  PortType.STRING, default="float", choices=_TYPES)
        self.add_input("Value", PortType.STRING, default="0")
        self.add_output("Value", PortType.ANY)

    def _coerce(self, type_name: str, raw: str) -> Any:
        type_name = (type_name or "float").strip().lower()
        raw = "" if raw is None else str(raw)
        try:
            if type_name == "float":
                return float(raw) if raw else 0.0
            if type_name == "int":
                return int(float(raw)) if raw else 0
            if type_name == "bool":
                return raw.strip().lower() not in ("", "0", "false", "no", "none", "off")
            return raw
        except (ValueError, TypeError):
            return {"float": 0.0, "int": 0, "bool": False, "string": ""}[type_name]

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"Value": self._coerce(inputs.get("Type"), inputs.get("Value"))}

    def export(self, iv, ov):
        # Emit a literal whose form matches the chosen Type
        type_name = str(self.inputs["Type"].default_value or "float").strip().lower()
        raw       = str(self.inputs["Value"].default_value or "")
        out       = ov.get("Value", "_const")

        if type_name == "float":
            try:
                literal = repr(float(raw))
            except (ValueError, TypeError):
                literal = "0.0"
        elif type_name == "int":
            try:
                literal = repr(int(float(raw)))
            except (ValueError, TypeError):
                literal = "0"
        elif type_name == "bool":
            literal = "True" if raw.strip().lower() not in (
                "", "0", "false", "no", "none", "off"
            ) else "False"
        else:
            literal = repr(raw)

        return [], [f"{out} = {literal}"]
