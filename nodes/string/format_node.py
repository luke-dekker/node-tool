"""FormatNode — format a string template with up to two arguments."""
from core.node import BaseNode, PortType


class FormatNode(BaseNode):
    type_name = "format"
    label = "Format"
    category = "Python"
    subcategory = "String"
    description = "Formats Template using {0} and {1} placeholders."

    def _setup_ports(self):
        self.add_input("Template", PortType.STRING, "{0} = {1}")
        self.add_input("Arg0", PortType.ANY, "")
        self.add_input("Arg1", PortType.ANY, "")
        self.add_output("Result", PortType.STRING)

    def execute(self, inputs):
        try:
            return {"Result": str(inputs["Template"]).format(inputs["Arg0"], inputs["Arg1"])}
        except (IndexError, KeyError):
            return {"Result": str(inputs["Template"])}

    def export(self, iv, ov):
        t = self._val(iv,"Template"); a0 = self._val(iv,"Arg0"); a1 = self._val(iv,"Arg1")
        return [], [f"{ov['Result']} = str({t}).format({a0}, {a1})"]
