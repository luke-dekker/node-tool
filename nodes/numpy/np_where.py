"""np.where(condition>0, x, y) node."""
import numpy as np
from core.node import BaseNode, PortType


class NpWhereNode(BaseNode):
    type_name = "np_where"
    label = "Where"
    category = "NumPy"
    description = "np.where(condition>0, x, y)"

    def _setup_ports(self):
        self.add_input("condition", PortType.NDARRAY)
        self.add_input("x",         PortType.NDARRAY)
        self.add_input("y",         PortType.NDARRAY)
        self.add_output("result",   PortType.NDARRAY)

    def execute(self, inputs):
        try:
            cond, x, y = inputs.get("condition"), inputs.get("x"), inputs.get("y")
            if any(v is None for v in [cond, x, y]):
                return {"result": None}
            return {"result": np.where(cond > 0, x, y)}
        except Exception:
            return {"result": None}

    def export(self, iv, ov):
        return (
            ["import numpy as np"],
            [f"{ov['result']} = np.where({self._val(iv, 'condition')} > 0, {self._val(iv, 'x')}, {self._val(iv, 'y')})"],
        )
