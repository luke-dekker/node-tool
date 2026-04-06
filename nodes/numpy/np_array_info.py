"""Print shape, dtype, min, max, mean of array node."""
import numpy as np
from core.node import BaseNode, PortType


class NpArrayInfoNode(BaseNode):
    type_name = "np_array_info"
    label = "Array Info"
    category = "NumPy"
    description = "Print shape, dtype, min, max, mean of array"

    def _setup_ports(self):
        self.add_input("array", PortType.NDARRAY)
        self.add_output("info", PortType.STRING)

    def execute(self, inputs):
        try:
            arr = inputs.get("array")
            if arr is None:
                return {"info": "None"}
            a = np.asarray(arr, dtype=float)
            return {"info": (f"shape={list(arr.shape)} dtype={arr.dtype} "
                             f"min={a.min():.4g} max={a.max():.4g} mean={a.mean():.4g}")}
        except Exception:
            return {"info": "error"}

    def export(self, iv, ov):
        arr = self._val(iv, "array")
        return (
            ["import numpy as np"],
            [
                f"{ov['info']} = f\"shape={{list({arr}.shape)}} dtype={{{arr}.dtype}} "
                f"min={{{arr}.astype(float).min():.4g}} max={{{arr}.astype(float).max():.4g}} mean={{{arr}.astype(float).mean():.4g}}\""
            ],
        )
