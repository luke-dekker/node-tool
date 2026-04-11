"""Example node definitions — copy and modify for your domain.

A node needs 5 things:
    1. type_name  — unique string ID (prefix with your domain: "rob_", "aud_", etc.)
    2. label      — display name shown on the canvas
    3. category   — palette group (must match what your plugin registers)
    4. _setup_ports() — declare inputs and outputs with their types
    5. execute()  — the actual computation

Optional but recommended:
    6. export()   — generate Python code for the Code panel / .py export
    7. description — tooltip text
    8. subcategory — sub-group within the palette category
"""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


class ExampleNode(BaseNode):
    """Simplest possible node — takes a float, doubles it, outputs the result."""

    # ── Required class attributes ────────────────────────────────────────
    type_name   = "example_double"              # unique ID (prefix with your domain)
    label       = "Double"                       # shown on canvas
    category    = "Example"                      # palette group
    description = "Multiplies the input by 2."   # tooltip

    # ── Port setup ───────────────────────────────────────────────────────
    def _setup_ports(self) -> None:
        # Inputs: name, type, default value
        self.add_input("value", PortType.FLOAT, 1.0)
        self.add_input("factor", PortType.FLOAT, 2.0)

        # Outputs: name, type
        self.add_output("result", PortType.FLOAT)

    # ── Execution ────────────────────────────────────────────────────────
    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        # inputs dict has one entry per input port, keyed by name.
        # Values are already coerced to the declared port type.
        value = float(inputs.get("value") or 0)
        factor = float(inputs.get("factor") or 2)
        return {"result": value * factor}

    # ── Export (optional but recommended) ────────────────────────────────
    def export(self, iv: dict, ov: dict) -> tuple[list[str], list[str]]:
        # iv: {port_name: upstream_variable_name_or_None}
        # ov: {port_name: variable_name_to_assign}
        # Returns: (imports_list, code_lines_list)
        v = self._val(iv, "value")    # helper: returns var name or literal default
        f = self._val(iv, "factor")
        return [], [f"{ov['result']} = {v} * {f}"]


class ExampleProcessorNode(BaseNode):
    """Example node using a custom port type (EXAMPLE_DATA)."""

    type_name   = "example_processor"
    label       = "Process Data"
    category    = "Example"
    description = "Takes custom data, processes it, outputs the result."

    def _setup_ports(self) -> None:
        # Use your plugin's custom port type (registered in __init__.py)
        self.add_input("data_in",  "EXAMPLE_DATA", default=None)
        self.add_input("scale",    PortType.FLOAT,  default=1.0)

        self.add_output("data_out", "EXAMPLE_DATA")
        self.add_output("info",     PortType.STRING)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        data = inputs.get("data_in")
        scale = float(inputs.get("scale") or 1.0)

        if data is None:
            return {"data_out": None, "info": "No input data"}

        # Your domain-specific processing here
        processed = data  # placeholder
        info = f"Processed with scale={scale}"

        return {"data_out": processed, "info": info}

    def export(self, iv, ov):
        d = iv.get("data_in") or "None"
        s = self._val(iv, "scale")
        return [], [
            f"{ov['data_out']} = {d}  # TODO: implement processing",
            f"{ov['info']} = f'Processed with scale={s}'",
        ]
