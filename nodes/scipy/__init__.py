"""SciPy nodes — stats, signal, interpolation, optimization."""
from __future__ import annotations
from core.node import BaseNode
from core.node import PortType

CATEGORY = "SciPy"


# ── Stats ─────────────────────────────────────────────────────────────────────

class SpDescribeNode(BaseNode):
    type_name = "sp_describe"
    label = "Describe"
    category = CATEGORY
    description = "scipy.stats.describe — nobs, mean, variance, skewness, kurtosis"

    def _setup_ports(self):
        self.add_input("array", PortType.NDARRAY)
        self.add_output("info", PortType.STRING)

    def execute(self, inputs):
        try:
            from scipy import stats
            arr = inputs.get("array")
            if arr is None:
                return {"info": "None"}
            d = stats.describe(arr.flatten())
            return {"info": (f"nobs={d.nobs} mean={d.mean:.4g} variance={d.variance:.4g} "
                             f"skewness={d.skewness:.4g} kurtosis={d.kurtosis:.4g}")}
        except Exception:
            return {"info": "error"}


class SpNormPdfNode(BaseNode):
    type_name = "sp_norm_pdf"
    label = "Normal PDF"
    category = CATEGORY
    description = "scipy.stats.norm.pdf(x, loc, scale)"

    def _setup_ports(self):
        self.add_input("x",     PortType.NDARRAY)
        self.add_input("loc",   PortType.FLOAT, 0.0)
        self.add_input("scale", PortType.FLOAT, 1.0)
        self.add_output("pdf",  PortType.NDARRAY)

    def execute(self, inputs):
        try:
            from scipy.stats import norm
            x     = inputs.get("x")
            loc   = inputs.get("loc",   0.0)
            scale = inputs.get("scale", 1.0)
            if x is None:
                return {"pdf": None}
            return {"pdf": norm.pdf(x, loc=float(loc), scale=float(scale))}
        except Exception:
            return {"pdf": None}


class SpTTestNode(BaseNode):
    type_name = "sp_ttest"
    label = "T-Test"
    category = CATEGORY
    description = "scipy.stats.ttest_ind(a, b)"

    def _setup_ports(self):
        self.add_input("a",         PortType.NDARRAY)
        self.add_input("b",         PortType.NDARRAY)
        self.add_output("statistic", PortType.FLOAT)
        self.add_output("pvalue",    PortType.FLOAT)

    def execute(self, inputs):
        null = {"statistic": None, "pvalue": None}
        try:
            from scipy.stats import ttest_ind
            a = inputs.get("a")
            b = inputs.get("b")
            if a is None or b is None:
                return null
            result = ttest_ind(a.flatten(), b.flatten())
            return {"statistic": float(result.statistic), "pvalue": float(result.pvalue)}
        except Exception:
            return null


class SpCorrelationNode(BaseNode):
    type_name = "sp_pearsonr"
    label = "Pearson R"
    category = CATEGORY
    description = "scipy.stats.pearsonr(x, y)"

    def _setup_ports(self):
        self.add_input("x",      PortType.NDARRAY)
        self.add_input("y",      PortType.NDARRAY)
        self.add_output("r",     PortType.FLOAT)
        self.add_output("pvalue", PortType.FLOAT)

    def execute(self, inputs):
        null = {"r": None, "pvalue": None}
        try:
            from scipy.stats import pearsonr
            x = inputs.get("x")
            y = inputs.get("y")
            if x is None or y is None:
                return null
            result = pearsonr(x.flatten(), y.flatten())
            return {"r": float(result.statistic), "pvalue": float(result.pvalue)}
        except Exception:
            return null


class SpHistogramNode(BaseNode):
    type_name = "sp_histogram"
    label = "Histogram"
    category = CATEGORY
    description = "np.histogram(array, bins) — counts and bin_edges"

    def _setup_ports(self):
        self.add_input("array",     PortType.NDARRAY)
        self.add_input("bins",      PortType.INT, 20)
        self.add_output("counts",   PortType.NDARRAY)
        self.add_output("bin_edges", PortType.NDARRAY)

    def execute(self, inputs):
        null = {"counts": None, "bin_edges": None}
        try:
            import numpy as np
            arr  = inputs.get("array")
            bins = inputs.get("bins", 20)
            if arr is None:
                return null
            counts, edges = np.histogram(arr.flatten(), bins=int(bins))
            return {"counts": counts, "bin_edges": edges}
        except Exception:
            return null


# ── Signal ────────────────────────────────────────────────────────────────────

class SpFFTNode(BaseNode):
    type_name = "sp_fft"
    label = "FFT"
    category = CATEGORY
    description = "np.fft.rfft — frequencies and magnitude"

    def _setup_ports(self):
        self.add_input("signal",      PortType.NDARRAY)
        self.add_output("frequencies", PortType.NDARRAY)
        self.add_output("magnitude",   PortType.NDARRAY)

    def execute(self, inputs):
        null = {"frequencies": None, "magnitude": None}
        try:
            import numpy as np
            signal = inputs.get("signal")
            if signal is None:
                return null
            sig = signal.flatten()
            fft_vals = np.fft.rfft(sig)
            magnitude = np.abs(fft_vals)
            frequencies = np.fft.rfftfreq(len(sig))
            return {"frequencies": frequencies, "magnitude": magnitude}
        except Exception:
            return null


class SpButterworthNode(BaseNode):
    type_name = "sp_butterworth"
    label = "Butterworth Filter"
    category = CATEGORY
    description = "scipy.signal.butter + sosfilt"

    def _setup_ports(self):
        self.add_input("signal",   PortType.NDARRAY)
        self.add_input("cutoff",   PortType.FLOAT,  0.1)
        self.add_input("order",    PortType.INT,    5)
        self.add_input("btype",    PortType.STRING, "low")
        self.add_output("filtered", PortType.NDARRAY)

    def execute(self, inputs):
        null = {"filtered": None}
        try:
            from scipy.signal import butter, sosfilt
            signal = inputs.get("signal")
            cutoff = inputs.get("cutoff", 0.1)
            order  = inputs.get("order",  5)
            btype  = inputs.get("btype",  "low") or "low"
            if signal is None:
                return null
            sos = butter(int(order), float(cutoff), btype=btype, output="sos")
            return {"filtered": sosfilt(sos, signal.flatten())}
        except Exception:
            return null


class SpConvolveNode(BaseNode):
    type_name = "sp_convolve"
    label = "Convolve"
    category = CATEGORY
    description = "np.convolve(a, b, mode)"

    def _setup_ports(self):
        self.add_input("a",      PortType.NDARRAY)
        self.add_input("b",      PortType.NDARRAY)
        self.add_input("mode",   PortType.STRING, "full")
        self.add_output("result", PortType.NDARRAY)

    def execute(self, inputs):
        null = {"result": None}
        try:
            import numpy as np
            a    = inputs.get("a")
            b    = inputs.get("b")
            mode = inputs.get("mode", "full") or "full"
            if a is None or b is None:
                return null
            return {"result": np.convolve(a.flatten(), b.flatten(), mode=mode)}
        except Exception:
            return null


# ── Interpolation ─────────────────────────────────────────────────────────────

class SpInterp1dNode(BaseNode):
    type_name = "sp_interp1d"
    label = "Interp1d"
    category = CATEGORY
    description = "scipy.interpolate.interp1d(x, y, kind) evaluated at x_new"

    def _setup_ports(self):
        self.add_input("x",     PortType.NDARRAY)
        self.add_input("y",     PortType.NDARRAY)
        self.add_input("x_new", PortType.NDARRAY)
        self.add_input("kind",  PortType.STRING, "linear")
        self.add_output("y_new", PortType.NDARRAY)

    def execute(self, inputs):
        null = {"y_new": None}
        try:
            from scipy.interpolate import interp1d
            x     = inputs.get("x")
            y     = inputs.get("y")
            x_new = inputs.get("x_new")
            kind  = inputs.get("kind", "linear") or "linear"
            if any(v is None for v in [x, y, x_new]):
                return null
            f = interp1d(x.flatten(), y.flatten(), kind=kind, bounds_error=False,
                         fill_value="extrapolate")
            return {"y_new": f(x_new.flatten())}
        except Exception:
            return null


# ── Optimization ──────────────────────────────────────────────────────────────

class SpCurveFitNode(BaseNode):
    type_name = "sp_curve_fit"
    label = "Curve Fit"
    category = CATEGORY
    description = "scipy.optimize.curve_fit using func_str like 'a*x+b'"

    def _setup_ports(self):
        self.add_input("x",        PortType.NDARRAY)
        self.add_input("y",        PortType.NDARRAY)
        self.add_input("func_str", PortType.STRING, "a*x+b")
        self.add_output("params",  PortType.NDARRAY)
        self.add_output("info",    PortType.STRING)

    def execute(self, inputs):
        null = {"params": None, "info": "error"}
        try:
            import numpy as np
            from scipy.optimize import curve_fit
            x        = inputs.get("x")
            y        = inputs.get("y")
            func_str = inputs.get("func_str", "a*x+b") or "a*x+b"
            if x is None or y is None:
                return null

            # Detect parameter names (a, b, c, d) appearing in func_str
            param_names = [p for p in ["a", "b", "c", "d"] if p in func_str]
            if not param_names:
                return {"params": None, "info": "No params (a,b,c,d) found in func_str"}

            # Build lambda: f(x, a, b, ...) = eval(func_str)
            args_str = ", ".join(["x"] + param_names)
            fn_code  = f"lambda {args_str}: {func_str}"
            fn = eval(fn_code, {"__builtins__": {}}, {"np": np})

            popt, _ = curve_fit(fn, x.flatten(), y.flatten(), maxfev=5000)
            info = ", ".join(f"{n}={v:.6g}" for n, v in zip(param_names, popt))
            return {"params": np.array(popt), "info": info}
        except Exception as e:
            return {"params": None, "info": str(e)}
