"""Math nodes: Add, Subtract, Multiply, Divide, Power, Clamp, MapRange, Round, Abs, Sqrt, Sin, Cos, RandomFloat."""

from __future__ import annotations
import math
import random
from typing import Any
from core.node import BaseNode
from core.node import PortType


class AddNode(BaseNode):
    type_name = "add"
    label = "Add"
    category = "Math"
    description = "Adds two numbers: Result = A + B"

    def _setup_ports(self) -> None:
        self.add_input("A", PortType.FLOAT, 0.0)
        self.add_input("B", PortType.FLOAT, 0.0)
        self.add_output("Result", PortType.FLOAT)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"Result": inputs["A"] + inputs["B"]}


class SubtractNode(BaseNode):
    type_name = "subtract"
    label = "Subtract"
    category = "Math"
    description = "Subtracts B from A: Result = A - B"

    def _setup_ports(self) -> None:
        self.add_input("A", PortType.FLOAT, 0.0)
        self.add_input("B", PortType.FLOAT, 0.0)
        self.add_output("Result", PortType.FLOAT)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"Result": inputs["A"] - inputs["B"]}


class MultiplyNode(BaseNode):
    type_name = "multiply"
    label = "Multiply"
    category = "Math"
    description = "Multiplies two numbers: Result = A * B"

    def _setup_ports(self) -> None:
        self.add_input("A", PortType.FLOAT, 1.0)
        self.add_input("B", PortType.FLOAT, 1.0)
        self.add_output("Result", PortType.FLOAT)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"Result": inputs["A"] * inputs["B"]}


class DivideNode(BaseNode):
    type_name = "divide"
    label = "Divide"
    category = "Math"
    description = "Divides A by B: Result = A / B. Returns 0 on division by zero."

    def _setup_ports(self) -> None:
        self.add_input("A", PortType.FLOAT, 1.0)
        self.add_input("B", PortType.FLOAT, 1.0)
        self.add_output("Result", PortType.FLOAT)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        b = inputs["B"]
        if b == 0:
            return {"Result": 0.0}
        return {"Result": inputs["A"] / b}


class PowerNode(BaseNode):
    type_name = "power"
    label = "Power"
    category = "Math"
    description = "Raises Base to Exponent: Result = Base ^ Exp"

    def _setup_ports(self) -> None:
        self.add_input("Base", PortType.FLOAT, 2.0)
        self.add_input("Exp", PortType.FLOAT, 2.0)
        self.add_output("Result", PortType.FLOAT)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        try:
            return {"Result": float(inputs["Base"] ** inputs["Exp"])}
        except (OverflowError, ZeroDivisionError, ValueError):
            return {"Result": 0.0}


class ClampNode(BaseNode):
    type_name = "clamp"
    label = "Clamp"
    category = "Math"
    description = "Clamps Value between Min and Max."

    def _setup_ports(self) -> None:
        self.add_input("Value", PortType.FLOAT, 0.0)
        self.add_input("Min", PortType.FLOAT, 0.0)
        self.add_input("Max", PortType.FLOAT, 1.0)
        self.add_output("Result", PortType.FLOAT)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        v = inputs["Value"]
        lo = inputs["Min"]
        hi = inputs["Max"]
        if lo > hi:
            lo, hi = hi, lo
        return {"Result": max(lo, min(hi, v))}


class MapRangeNode(BaseNode):
    type_name = "map_range"
    label = "Map Range"
    category = "Math"
    description = "Re-maps Value from [InMin, InMax] to [OutMin, OutMax]."

    def _setup_ports(self) -> None:
        self.add_input("Value", PortType.FLOAT, 0.5)
        self.add_input("InMin", PortType.FLOAT, 0.0)
        self.add_input("InMax", PortType.FLOAT, 1.0)
        self.add_input("OutMin", PortType.FLOAT, 0.0)
        self.add_input("OutMax", PortType.FLOAT, 100.0)
        self.add_output("Result", PortType.FLOAT)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        v = inputs["Value"]
        in_min = inputs["InMin"]
        in_max = inputs["InMax"]
        out_min = inputs["OutMin"]
        out_max = inputs["OutMax"]
        denom = in_max - in_min
        if denom == 0:
            return {"Result": out_min}
        t = (v - in_min) / denom
        return {"Result": out_min + t * (out_max - out_min)}


class RoundNode(BaseNode):
    type_name = "round"
    label = "Round"
    category = "Math"
    description = "Rounds Value to N decimal places."

    def _setup_ports(self) -> None:
        self.add_input("Value", PortType.FLOAT, 0.0)
        self.add_input("Decimals", PortType.INT, 0)
        self.add_output("Result", PortType.FLOAT)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"Result": round(inputs["Value"], int(inputs["Decimals"]))}


class AbsNode(BaseNode):
    type_name = "abs"
    label = "Abs"
    category = "Math"
    description = "Absolute value: Result = |Value|"

    def _setup_ports(self) -> None:
        self.add_input("Value", PortType.FLOAT, 0.0)
        self.add_output("Result", PortType.FLOAT)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"Result": abs(inputs["Value"])}


class SqrtNode(BaseNode):
    type_name = "sqrt"
    label = "Sqrt"
    category = "Math"
    description = "Square root: Result = sqrt(Value). Returns 0 for negative input."

    def _setup_ports(self) -> None:
        self.add_input("Value", PortType.FLOAT, 4.0)
        self.add_output("Result", PortType.FLOAT)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        v = inputs["Value"]
        if v < 0:
            return {"Result": 0.0}
        return {"Result": math.sqrt(v)}


class SinNode(BaseNode):
    type_name = "sin"
    label = "Sin"
    category = "Math"
    description = "Sine of angle in degrees."

    def _setup_ports(self) -> None:
        self.add_input("Degrees", PortType.FLOAT, 0.0)
        self.add_output("Result", PortType.FLOAT)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"Result": math.sin(math.radians(inputs["Degrees"]))}


class CosNode(BaseNode):
    type_name = "cos"
    label = "Cos"
    category = "Math"
    description = "Cosine of angle in degrees."

    def _setup_ports(self) -> None:
        self.add_input("Degrees", PortType.FLOAT, 0.0)
        self.add_output("Result", PortType.FLOAT)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"Result": math.cos(math.radians(inputs["Degrees"]))}


class RandomFloatNode(BaseNode):
    type_name = "random_float"
    label = "Random Float"
    category = "Math"
    description = "Returns a random float between Min and Max."

    def _setup_ports(self) -> None:
        self.add_input("Min", PortType.FLOAT, 0.0)
        self.add_input("Max", PortType.FLOAT, 1.0)
        self.add_output("Result", PortType.FLOAT)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        lo = inputs["Min"]
        hi = inputs["Max"]
        if lo > hi:
            lo, hi = hi, lo
        return {"Result": random.uniform(lo, hi)}
