"""Consolidated signal-processing node.

Replaces SpButterworthNode, SpConvolveNode, SpFFTNode, SpHistogramNode with a
single mode-dispatched node. Pick `op`; only the inputs that op reads matter,
and only a subset of output ports carry meaningful values.

Outputs (all 2 always present; the second is unused for single-output ops):
  primary    — np.ndarray  (filtered / result / magnitude / counts)
  secondary  — np.ndarray  (None / None / frequencies / bin_edges)
"""
from __future__ import annotations
from core.node import BaseNode, PortType


_OPS = ["butter", "convolve", "fft", "histogram"]


class SpSignalNode(BaseNode):
    type_name   = "sp_signal"
    label       = "Signal Op"
    category    = "SciPy"
    description = (
        "Signal-processing helpers. Pick `op`:\n"
        "  butter     — butterworth low/high/band filter (signal, cutoff, order, btype)\n"
        "  convolve   — np.convolve(a, b, mode)\n"
        "  fft        — primary=magnitude, secondary=frequencies\n"
        "  histogram  — primary=counts,    secondary=bin_edges"
    )

    def _setup_ports(self):
        self.add_input("op",     PortType.STRING, "fft", choices=_OPS)
        self.add_input("a",      PortType.NDARRAY, optional=True)
        self.add_input("b",      PortType.NDARRAY, optional=True)
        self.add_input("cutoff", PortType.FLOAT,   0.1, optional=True)
        self.add_input("order",  PortType.INT,     5,   optional=True)
        self.add_input("btype",  PortType.STRING,  "low",
                       choices=["low", "high", "band"], optional=True)
        self.add_input("mode",   PortType.STRING,  "full",
                       choices=["full", "same", "valid"], optional=True)
        self.add_input("bins",   PortType.INT,     20, optional=True)
        self.add_output("primary",   PortType.NDARRAY)
        self.add_output("secondary", PortType.NDARRAY)

    def relevant_inputs(self, values):
        op = (values.get("op") or "fft").strip()
        per_op = {
            "butter":    ["op", "cutoff", "order", "btype"],
            "convolve":  ["op", "mode"],
            "fft":       ["op"],
            "histogram": ["op", "bins"],
        }
        return per_op.get(op, ["op"])

    def execute(self, inputs):
        out = {"primary": None, "secondary": None}
        try:
            import numpy as np
            op = (inputs.get("op") or "fft").strip()
            a = inputs.get("a")
            if a is None:
                return out
            arr = a.flatten()
            if op == "butter":
                from scipy.signal import butter, sosfilt
                sos = butter(int(inputs.get("order", 5)),
                             float(inputs.get("cutoff", 0.1)),
                             btype=inputs.get("btype", "low") or "low",
                             output="sos")
                out["primary"] = sosfilt(sos, arr)
                return out
            if op == "convolve":
                b = inputs.get("b")
                if b is None:
                    return out
                out["primary"] = np.convolve(arr, b.flatten(),
                                             mode=inputs.get("mode", "full") or "full")
                return out
            if op == "fft":
                fft_vals = np.fft.rfft(arr)
                out["primary"]   = np.abs(fft_vals)
                out["secondary"] = np.fft.rfftfreq(len(arr))
                return out
            if op == "histogram":
                counts, edges = np.histogram(arr, bins=int(inputs.get("bins", 20)))
                out["primary"]   = counts
                out["secondary"] = edges
                return out
            return out
        except Exception:
            return out

    def export(self, iv, ov):
        op = (self.inputs["op"].default_value or "fft")
        a = self._val(iv, "a")
        if op == "butter":
            cut, ord_, bt = self._val(iv, "cutoff"), self._val(iv, "order"), self._val(iv, "btype")
            return ["from scipy.signal import butter, sosfilt"], [
                f"_sos = butter(int({ord_}), float({cut}), btype={bt}, output='sos')",
                f"{ov['primary']} = sosfilt(_sos, {a}.flatten())",
            ]
        if op == "convolve":
            b, mode = self._val(iv, "b"), self._val(iv, "mode")
            return ["import numpy as np"], [
                f"{ov['primary']} = np.convolve({a}.flatten(), {b}.flatten(), mode={mode})"
            ]
        if op == "fft":
            return ["import numpy as np"], [
                f"_fft_sig  = {a}.flatten()",
                f"_fft_vals = np.fft.rfft(_fft_sig)",
                f"{ov['primary']}   = np.abs(_fft_vals)",
                f"{ov['secondary']} = np.fft.rfftfreq(len(_fft_sig))",
            ]
        if op == "histogram":
            bins = self._val(iv, "bins")
            return ["import numpy as np"], [
                f"{ov['primary']}, {ov['secondary']} = np.histogram({a}.flatten(), bins={bins})"
            ]
        return [], [f"# unknown op {op!r}"]
