"""pd.DataFrame(json.loads(json_str)) node."""
import json
from core.node import BaseNode, PortType


class PdFromDictNode(BaseNode):
    type_name = "pd_from_dict"
    label = "From JSON Dict"
    category = "Pandas"
    description = "pd.DataFrame(json.loads(json_str))"

    def _setup_ports(self):
        self.add_input("json_str", PortType.STRING, "{}")
        self.add_output("df",      PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            import pandas as pd
            return {"df": pd.DataFrame(json.loads(inputs.get("json_str") or "{}"))}
        except Exception:
            return {"df": None}

    def export(self, iv, ov):
        return (
            ["import pandas as pd", "import json"],
            [f"{ov['df']} = pd.DataFrame(json.loads({self._val(iv, 'json_str')}))"],
        )
