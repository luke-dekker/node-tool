from core.node import BaseNode, PortType

class SpTTestNode(BaseNode):
    type_name = "sp_ttest"; label = "T-Test"; category = "SciPy"
    description = "scipy.stats.ttest_ind(a, b)"
    def _setup_ports(self):
        self.add_input("a", PortType.NDARRAY)
        self.add_input("b", PortType.NDARRAY)
        self.add_output("statistic", PortType.FLOAT)
        self.add_output("pvalue", PortType.FLOAT)
    def execute(self, inputs):
        null = {"statistic": None, "pvalue": None}
        try:
            from scipy.stats import ttest_ind
            a, b = inputs.get("a"), inputs.get("b")
            if a is None or b is None: return null
            r = ttest_ind(a.flatten(), b.flatten())
            return {"statistic": float(r.statistic), "pvalue": float(r.pvalue)}
        except Exception: return null
    def export(self, iv, ov):
        a, b = self._val(iv,"a"), self._val(iv,"b")
        return ["from scipy.stats import ttest_ind"], [
            f"_ttest = ttest_ind({a}.flatten(), {b}.flatten())",
            f"{ov['statistic']} = float(_ttest.statistic)",
            f"{ov['pvalue']} = float(_ttest.pvalue)",
        ]
