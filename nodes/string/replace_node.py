"""ReplaceNode — replace all occurrences of a substring."""
from core.node import BaseNode, PortType


class ReplaceNode(BaseNode):
    type_name = "replace"
    label = "Replace"
    category = "String"
    description = "Replaces all occurrences of Old with New in Value."

    def _setup_ports(self):
        self.add_input("Value", PortType.STRING, "")
        self.add_input("Old",   PortType.STRING, "")
        self.add_input("New",   PortType.STRING, "")
        self.add_output("Result", PortType.STRING)

    def execute(self, inputs):
        return {"Result": str(inputs["Value"]).replace(str(inputs["Old"]), str(inputs["New"]))}
