"""PrintNode — print a value to the terminal panel and pass it through."""
from core.node import BaseNode, PortType


class PrintNode(BaseNode):
    type_name = "print"
    label = "Print"
    category = "Python"
    subcategory = "Data"
    description = "Prints Value to the terminal output (with optional Label) and passes it through."

    def _setup_ports(self) -> None:
        self.add_input("Value", PortType.ANY, "")
        self.add_input("Label", PortType.STRING, "")
        self.add_output("Value", PortType.ANY)

    def execute(self, inputs):
        label = str(inputs["Label"])
        value = inputs["Value"]
        line = f"{label}: {value}" if label else str(value)
        return {"Value": value, "__terminal__": line}

    def export(self, iv, ov):
        v = self._val(iv, "Value")
        lab = self._val(iv, "Label")
        return [], [
            f"{ov['Value']} = {v}",
            f"print(f'{{{lab}}}: {{{v}}}' if {lab} else str({v}))",
        ]
