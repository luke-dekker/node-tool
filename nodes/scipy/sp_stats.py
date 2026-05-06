"""Consolidated scipy.stats node.

Replaces SpDescribeNode, SpCorrelationNode, SpTTestNode, SpNormPdfNode with a
single mode-dispatched node. Pick `op`; only the inputs that op reads matter,
and only a subset of output ports carry meaningful values.

Outputs (all 4 always present; unused ones are None / 0):
  array      — np.ndarray (norm_pdf only)
  statistic  — float      (ttest, pearsonr)
  pvalue     — float      (ttest, pearsonr)
  info       — str        (describe summary, or empty)
"""
from __future__ import annotations
from core.node import BaseNode, PortType


_OPS = ["describe", "pearsonr", "ttest", "norm_pdf"]


class SpStatsNode(BaseNode):
    type_name   = "sp_stats"
    label       = "SciPy Stats"
    category    = "SciPy"
    description = (
        "scipy.stats helpers. Pick `op`:\n"
        "  describe  — nobs/mean/variance/skewness/kurtosis  → info\n"
        "  pearsonr  — pearson correlation                   → statistic, pvalue\n"
        "  ttest     — independent t-test                    → statistic, pvalue\n"
        "  norm_pdf  — normal pdf at points x                → array"
    )

    def _setup_ports(self):
        self.add_input("op",    PortType.STRING, "describe", choices=_OPS)
        # one-array ops (describe, norm_pdf)
        self.add_input("x",     PortType.NDARRAY, optional=True)
        # two-array ops (pearsonr, ttest)
        self.add_input("y",     PortType.NDARRAY, optional=True)
        # norm_pdf params
        self.add_input("loc",   PortType.FLOAT, 0.0, optional=True)
        self.add_input("scale", PortType.FLOAT, 1.0, optional=True)
        self.add_output("array",     PortType.NDARRAY)
        self.add_output("statistic", PortType.FLOAT)
        self.add_output("pvalue",    PortType.FLOAT)
        self.add_output("info",      PortType.STRING)

    def relevant_inputs(self, values):
        op = (values.get("op") or "describe").strip()
        if op == "norm_pdf":           return ["op", "loc", "scale"]
        if op in ("pearsonr", "ttest"): return ["op"]   # x, y are wired data ports
        return ["op"]                                    # describe

    def execute(self, inputs):
        out = {"array": None, "statistic": 0.0, "pvalue": 0.0, "info": ""}
        try:
            from scipy import stats
            op = (inputs.get("op") or "describe").strip()
            x = inputs.get("x")
            if op == "describe":
                if x is None:
                    return out | {"info": "None"}
                d = stats.describe(x.flatten())
                out["info"] = (f"nobs={d.nobs} mean={d.mean:.4g} variance={d.variance:.4g} "
                               f"skewness={d.skewness:.4g} kurtosis={d.kurtosis:.4g}")
                return out
            if op == "norm_pdf":
                if x is None:
                    return out
                from scipy.stats import norm
                out["array"] = norm.pdf(x, loc=float(inputs.get("loc", 0.0)),
                                          scale=float(inputs.get("scale", 1.0)))
                return out
            if op in ("pearsonr", "ttest"):
                y = inputs.get("y")
                if x is None or y is None:
                    return out
                if op == "pearsonr":
                    r = stats.pearsonr(x.flatten(), y.flatten())
                else:
                    r = stats.ttest_ind(x.flatten(), y.flatten())
                out["statistic"] = float(r.statistic)
                out["pvalue"]    = float(r.pvalue)
                return out
            return out
        except Exception:
            return out

    def export(self, iv, ov):
        op = (self.inputs["op"].default_value or "describe")
        x = self._val(iv, "x")
        if op == "describe":
            return ["from scipy import stats"], [
                f"_d = stats.describe({x}.flatten())",
                f"{ov['info']} = (f'nobs={{_d.nobs}} mean={{_d.mean:.4g}} variance={{_d.variance:.4g}} '"
                f" f'skewness={{_d.skewness:.4g}} kurtosis={{_d.kurtosis:.4g}}')",
            ]
        if op == "norm_pdf":
            loc = self._val(iv, "loc"); scale = self._val(iv, "scale")
            return ["from scipy.stats import norm"], [
                f"{ov['array']} = norm.pdf({x}, loc={loc}, scale={scale})"
            ]
        if op == "pearsonr":
            y = self._val(iv, "y")
            return ["from scipy.stats import pearsonr"], [
                f"_r = pearsonr({x}.flatten(), {y}.flatten())",
                f"{ov['statistic']} = float(_r.statistic)",
                f"{ov['pvalue']} = float(_r.pvalue)",
            ]
        if op == "ttest":
            y = self._val(iv, "y")
            return ["from scipy.stats import ttest_ind"], [
                f"_r = ttest_ind({x}.flatten(), {y}.flatten())",
                f"{ov['statistic']} = float(_r.statistic)",
                f"{ov['pvalue']} = float(_r.pvalue)",
            ]
        return [], [f"# unknown op {op!r}"]
