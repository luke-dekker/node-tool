"""CastNode — single Cast node with a target Type dropdown.

Replaces ToFloatNode / ToIntNode / ToBoolNode / ToStringNode with one node
whose `Type` input chooses the conversion target. The legacy single-type
cast nodes are kept for loading old saves; new graphs should use this.
"""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


_TYPES = ["float", "int", "bool", "string"]


class CastNode(BaseNode):
    type_name   = "cast"
    label       = "Cast"
    category    = "Python"
    subcategory = "Data"
    description = (
        "Coerce a value to float / int / bool / string. Replaces the four "
        "single-type ToFloat/ToInt/ToBool/ToString nodes. Failed conversions "
        "fall back to the type's zero value."
    )

    def _setup_ports(self) -> None:
        self.add_input("Value", PortType.ANY, default=None)
        self.add_input("Type",  PortType.STRING, default="float", choices=_TYPES)
        self.add_output("Result", PortType.ANY)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        v = inputs.get("Value")
        t = (inputs.get("Type") or "float").strip().lower()
        try:
            if t == "float":
                return {"Result": float(v) if v is not None else 0.0}
            if t == "int":
                return {"Result": int(float(v)) if v is not None else 0}
            if t == "bool":
                if isinstance(v, str):
                    return {"Result": v.strip().lower() not in (
                        "", "0", "false", "no", "none", "off")}
                return {"Result": bool(v)}
            return {"Result": str(v) if v is not None else ""}
        except (ValueError, TypeError):
            return {"Result": {"float": 0.0, "int": 0, "bool": False, "string": ""}[t]}

    def export(self, iv, ov):
        v = iv.get("Value") or "None"
        t = str(self.inputs["Type"].default_value or "float").strip().lower()
        out = ov.get("Result", "_cast")
        if t == "float":
            return [], [
                f"try: {out} = float({v})",
                f"except (ValueError, TypeError): {out} = 0.0",
            ]
        if t == "int":
            return [], [
                f"try: {out} = int(float({v}))",
                f"except (ValueError, TypeError): {out} = 0",
            ]
        if t == "bool":
            return [], [
                f"_v = {v}",
                f"{out} = (_v.strip().lower() not in ('', '0', 'false', 'no', 'none', 'off')) "
                f"if isinstance(_v, str) else bool(_v)",
            ]
        return [], [f"{out} = str({v}) if {v} is not None else ''"]
