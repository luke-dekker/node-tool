"""BranchNode — route a value based on a boolean condition."""
from core.node import BaseNode, PortType


class BranchNode(BaseNode):
    type_name = "branch"
    label = "Branch"
    category = "Python"
    subcategory = "Logic"
    description = "Outputs True Value if Condition is true, else False Value."

    def _setup_ports(self):
        self.add_input("Condition",   PortType.BOOL, False)
        self.add_input("True Value",  PortType.ANY,  1.0)
        self.add_input("False Value", PortType.ANY,  0.0)
        self.add_output("Result", PortType.ANY)

    def execute(self, inputs):
        if bool(inputs["Condition"]):
            return {"Result": inputs["True Value"]}
        return {"Result": inputs["False Value"]}

    def export(self, iv, ov):
        cond = self._val(iv,"Condition")
        tv   = self._val(iv,"True Value")
        fv   = self._val(iv,"False Value")
        return [], [f"{ov['Result']} = {tv} if {cond} else {fv}"]
