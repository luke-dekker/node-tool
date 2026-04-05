"""Logic nodes: Compare, And, Or, Not, Branch."""

from __future__ import annotations
from typing import Any
from core.node import BaseNode
from core.node import PortType


class CompareNode(BaseNode):
    type_name = "compare"
    label = "Compare"
    category = "Logic"
    description = "Compares A and B. Op: 0=<, 1=<=, 2==, 3=>=, 4=>, 5=!="

    def _setup_ports(self) -> None:
        self.add_input("A", PortType.FLOAT, 0.0)
        self.add_input("B", PortType.FLOAT, 0.0)
        # 0=less, 1=less_equal, 2=equal, 3=greater_equal, 4=greater, 5=not_equal
        self.add_input("Op", PortType.INT, 2, "0=<, 1=<=, 2===, 3=>=, 4=>, 5=!=")
        self.add_output("Result", PortType.BOOL)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        a, b = inputs["A"], inputs["B"]
        op = int(inputs["Op"])
        ops = [
            lambda x, y: x < y,
            lambda x, y: x <= y,
            lambda x, y: x == y,
            lambda x, y: x >= y,
            lambda x, y: x > y,
            lambda x, y: x != y,
        ]
        fn = ops[op % len(ops)]
        return {"Result": fn(a, b)}


class AndNode(BaseNode):
    type_name = "and"
    label = "And"
    category = "Logic"
    description = "Logical AND: Result = A AND B"

    def _setup_ports(self) -> None:
        self.add_input("A", PortType.BOOL, False)
        self.add_input("B", PortType.BOOL, False)
        self.add_output("Result", PortType.BOOL)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"Result": bool(inputs["A"]) and bool(inputs["B"])}


class OrNode(BaseNode):
    type_name = "or"
    label = "Or"
    category = "Logic"
    description = "Logical OR: Result = A OR B"

    def _setup_ports(self) -> None:
        self.add_input("A", PortType.BOOL, False)
        self.add_input("B", PortType.BOOL, False)
        self.add_output("Result", PortType.BOOL)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"Result": bool(inputs["A"]) or bool(inputs["B"])}


class NotNode(BaseNode):
    type_name = "not"
    label = "Not"
    category = "Logic"
    description = "Logical NOT: Result = NOT Value"

    def _setup_ports(self) -> None:
        self.add_input("Value", PortType.BOOL, False)
        self.add_output("Result", PortType.BOOL)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"Result": not bool(inputs["Value"])}


class BranchNode(BaseNode):
    type_name = "branch"
    label = "Branch"
    category = "Logic"
    description = "If Condition is true, outputs True Value; otherwise False Value."

    def _setup_ports(self) -> None:
        self.add_input("Condition", PortType.BOOL, False)
        self.add_input("True Value", PortType.ANY, 1.0)
        self.add_input("False Value", PortType.ANY, 0.0)
        self.add_output("Result", PortType.ANY)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if bool(inputs["Condition"]):
            return {"Result": inputs["True Value"]}
        return {"Result": inputs["False Value"]}
