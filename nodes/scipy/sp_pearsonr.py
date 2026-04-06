from core.node import BaseNode, PortType

class SpCorrelationNode(BaseNode):
    type_name = "sp_pearsonr"; label = "Pearson R"; category = "SciPy"
    description = "scipy.stats.pearsonr(x, y)"
    def _setup_ports(self):
        self.add_input("x", PortType.NDARRAY)
        self.add_input("y", PortType.NDARRAY)
        self.add_output("r", PortType.FLOAT)
        self.add_output("pvalue", PortType.FLOAT)
    def execute(self, inputs):
        null = {"r": None, "pvalue": None}
        try:
            from scipy.stats import pearsonr
            x, y = inputs.get("x"), inputs.get("y")
            if x is None or y is None: return null
            r = pearsonr(x.flatten(), y.flatten())
            return {"r": float(r.statistic), "pvalue": float(r.pvalue)}
        except Exception: return null
    def export(self, iv, ov):
        x, y = self._val(iv,"x"), self._val(iv,"y")
        return ["from scipy.stats import pearsonr"], [
            f"_pr = pearsonr({x}.flatten(), {y}.flatten())",
            f"{ov['r']} = float(_pr.statistic)",
            f"{ov['pvalue']} = float(_pr.pvalue)",
        ]
