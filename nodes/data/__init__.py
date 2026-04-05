"""Data nodes: constants, print, type converters."""

from __future__ import annotations
from typing import Any
from core.node import BaseNode
from core.node import PortType


class FloatConstNode(BaseNode):
    type_name = "float_const"
    label = "Float Const"
    category = "Data"
    description = "Outputs a constant float value."

    def _setup_ports(self) -> None:
        self.add_input("Value", PortType.FLOAT, 0.0)
        self.add_output("Value", PortType.FLOAT)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"Value": float(inputs["Value"])}


class IntConstNode(BaseNode):
    type_name = "int_const"
    label = "Int Const"
    category = "Data"
    description = "Outputs a constant integer value."

    def _setup_ports(self) -> None:
        self.add_input("Value", PortType.INT, 0)
        self.add_output("Value", PortType.INT)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"Value": int(inputs["Value"])}


class BoolConstNode(BaseNode):
    type_name = "bool_const"
    label = "Bool Const"
    category = "Data"
    description = "Outputs a constant boolean value."

    def _setup_ports(self) -> None:
        self.add_input("Value", PortType.BOOL, False)
        self.add_output("Value", PortType.BOOL)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"Value": bool(inputs["Value"])}


class StringConstNode(BaseNode):
    type_name = "string_const"
    label = "String Const"
    category = "Data"
    description = "Outputs a constant string value."

    def _setup_ports(self) -> None:
        self.add_input("Value", PortType.STRING, "")
        self.add_output("Value", PortType.STRING)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"Value": str(inputs["Value"])}


class PrintNode(BaseNode):
    type_name = "print"
    label = "Print"
    category = "Data"
    description = "Prints the value to the terminal output. Also passes it through."

    def _setup_ports(self) -> None:
        self.add_input("Value", PortType.ANY, "")
        self.add_input("Label", PortType.STRING, "")
        self.add_output("Value", PortType.ANY)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        label = str(inputs["Label"])
        value = inputs["Value"]
        line = f"{label}: {value}" if label else str(value)
        return {"Value": value, "__terminal__": line}


class ToFloatNode(BaseNode):
    type_name = "to_float"
    label = "To Float"
    category = "Data"
    description = "Converts Value to float."

    def _setup_ports(self) -> None:
        self.add_input("Value", PortType.ANY, 0)
        self.add_output("Result", PortType.FLOAT)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        try:
            return {"Result": float(inputs["Value"])}
        except (ValueError, TypeError):
            return {"Result": 0.0}


class ToIntNode(BaseNode):
    type_name = "to_int"
    label = "To Int"
    category = "Data"
    description = "Converts Value to integer."

    def _setup_ports(self) -> None:
        self.add_input("Value", PortType.ANY, 0)
        self.add_output("Result", PortType.INT)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        try:
            return {"Result": int(float(inputs["Value"]))}
        except (ValueError, TypeError):
            return {"Result": 0}


class ToStringNode(BaseNode):
    type_name = "to_string"
    label = "To String"
    category = "Data"
    description = "Converts Value to string."

    def _setup_ports(self) -> None:
        self.add_input("Value", PortType.ANY, None)
        self.add_output("Result", PortType.STRING)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"Result": str(inputs["Value"])}


class ToBoolNode(BaseNode):
    type_name = "to_bool"
    label = "To Bool"
    category = "Data"
    description = "Converts Value to bool."

    def _setup_ports(self) -> None:
        self.add_input("Value", PortType.ANY, 0)
        self.add_output("Result", PortType.BOOL)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        v = inputs["Value"]
        if isinstance(v, str):
            result = v.lower() not in ("", "0", "false", "no", "none")
        else:
            result = bool(v)
        return {"Result": result}
