from core.node import BaseNode, PortType

class SpDescribeNode(BaseNode):
    type_name = "sp_describe"; label = "Describe"; category = "SciPy"
    description = "scipy.stats.describe — nobs, mean, variance, skewness, kurtosis"
    def _setup_ports(self):
        self.add_input("array", PortType.NDARRAY)
        self.add_output("info", PortType.STRING)
    def execute(self, inputs):
        try:
            from scipy import stats
            arr = inputs.get("array")
            if arr is None: return {"info": "None"}
            d = stats.describe(arr.flatten())
            return {"info": f"nobs={d.nobs} mean={d.mean:.4g} variance={d.variance:.4g} skewness={d.skewness:.4g} kurtosis={d.kurtosis:.4g}"}
        except Exception: return {"info": "error"}
    def export(self, iv, ov):
        arr = self._val(iv, "array")
        return ["from scipy import stats"], [
            f"_spd = stats.describe({arr}.flatten())",
            f"{ov['info']} = f'nobs={{_spd.nobs}} mean={{_spd.mean:.4g}} variance={{_spd.variance:.4g}} skewness={{_spd.skewness:.4g}} kurtosis={{_spd.kurtosis:.4g}}'",
        ]
