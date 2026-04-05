"""String nodes: Concat, Format, Upper, Lower, Strip, Length, Contains, Replace."""

from __future__ import annotations
from typing import Any
from core.node import BaseNode
from core.node import PortType


class ConcatNode(BaseNode):
    type_name = "concat"
    label = "Concat"
    category = "String"
    description = "Concatenates two strings: Result = A + B"

    def _setup_ports(self) -> None:
        self.add_input("A", PortType.STRING, "")
        self.add_input("B", PortType.STRING, "")
        self.add_output("Result", PortType.STRING)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"Result": str(inputs["A"]) + str(inputs["B"])}


class FormatNode(BaseNode):
    type_name = "format"
    label = "Format"
    category = "String"
    description = "Formats a string using {0} and {1} placeholders."

    def _setup_ports(self) -> None:
        self.add_input("Template", PortType.STRING, "{0} = {1}")
        self.add_input("Arg0", PortType.ANY, "")
        self.add_input("Arg1", PortType.ANY, "")
        self.add_output("Result", PortType.STRING)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        try:
            result = str(inputs["Template"]).format(inputs["Arg0"], inputs["Arg1"])
        except (IndexError, KeyError):
            result = str(inputs["Template"])
        return {"Result": result}


class UpperNode(BaseNode):
    type_name = "upper"
    label = "Upper"
    category = "String"
    description = "Converts string to uppercase."

    def _setup_ports(self) -> None:
        self.add_input("Value", PortType.STRING, "")
        self.add_output("Result", PortType.STRING)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"Result": str(inputs["Value"]).upper()}


class LowerNode(BaseNode):
    type_name = "lower"
    label = "Lower"
    category = "String"
    description = "Converts string to lowercase."

    def _setup_ports(self) -> None:
        self.add_input("Value", PortType.STRING, "")
        self.add_output("Result", PortType.STRING)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"Result": str(inputs["Value"]).lower()}


class StripNode(BaseNode):
    type_name = "strip"
    label = "Strip"
    category = "String"
    description = "Strips leading and trailing whitespace."

    def _setup_ports(self) -> None:
        self.add_input("Value", PortType.STRING, "")
        self.add_output("Result", PortType.STRING)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"Result": str(inputs["Value"]).strip()}


class LengthNode(BaseNode):
    type_name = "length"
    label = "Length"
    category = "String"
    description = "Returns the length of the string."

    def _setup_ports(self) -> None:
        self.add_input("Value", PortType.STRING, "")
        self.add_output("Length", PortType.INT)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"Length": len(str(inputs["Value"]))}


class ContainsNode(BaseNode):
    type_name = "contains"
    label = "Contains"
    category = "String"
    description = "Returns True if Haystack contains Needle."

    def _setup_ports(self) -> None:
        self.add_input("Haystack", PortType.STRING, "")
        self.add_input("Needle", PortType.STRING, "")
        self.add_output("Result", PortType.BOOL)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"Result": str(inputs["Needle"]) in str(inputs["Haystack"])}


class ReplaceNode(BaseNode):
    type_name = "replace"
    label = "Replace"
    category = "String"
    description = "Replaces all occurrences of Old with New in Value."

    def _setup_ports(self) -> None:
        self.add_input("Value", PortType.STRING, "")
        self.add_input("Old", PortType.STRING, "")
        self.add_input("New", PortType.STRING, "")
        self.add_output("Result", PortType.STRING)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"Result": str(inputs["Value"]).replace(str(inputs["Old"]), str(inputs["New"]))}
