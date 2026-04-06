"""Random float DataFrame with col_0..col_n + binary label column node."""
from core.node import BaseNode, PortType


class PdMakeSampleNode(BaseNode):
    type_name = "pd_make_sample"
    label = "Make Sample"
    category = "Pandas"
    description = "Random float DataFrame with col_0..col_n + binary label column"

    def _setup_ports(self):
        self.add_input("rows", PortType.INT, 100)
        self.add_input("cols", PortType.INT, 4)
        self.add_input("seed", PortType.INT, 42)
        self.add_output("df",  PortType.DATAFRAME)

    def execute(self, inputs):
        try:
            import numpy as np
            import pandas as pd
            rows = int(inputs.get("rows") or 100)
            cols = int(inputs.get("cols") or 4)
            seed = int(inputs.get("seed") if inputs.get("seed") is not None else 42)
            rng  = np.random.default_rng(seed)
            data = {f"col_{i}": rng.random(rows) for i in range(cols)}
            data["label"] = rng.integers(0, 2, rows)
            return {"df": pd.DataFrame(data)}
        except Exception:
            return {"df": None}

    def export(self, iv, ov):
        rows = self._val(iv, "rows")
        cols = self._val(iv, "cols")
        seed = self._val(iv, "seed")
        return (
            ["import numpy as np", "import pandas as pd"],
            [
                f"_rng = np.random.default_rng({seed})",
                f"{ov['df']} = pd.DataFrame({{f'col_{{i}}': _rng.random({rows}) for i in range({cols})}}) ",
                f"{ov['df']}['label'] = _rng.integers(0, 2, {rows})",
            ],
        )
