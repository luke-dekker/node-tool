from core.node import BaseNode, PortType

class SpFFTNode(BaseNode):
    type_name = "sp_fft"; label = "FFT"; category = "SciPy"
    description = "np.fft.rfft — frequencies and magnitude"
    def _setup_ports(self):
        self.add_input("signal", PortType.NDARRAY)
        self.add_output("frequencies", PortType.NDARRAY)
        self.add_output("magnitude", PortType.NDARRAY)
    def execute(self, inputs):
        null = {"frequencies": None, "magnitude": None}
        try:
            import numpy as np
            sig = inputs.get("signal")
            if sig is None: return null
            sig = sig.flatten(); fft_vals = np.fft.rfft(sig)
            return {"frequencies": np.fft.rfftfreq(len(sig)), "magnitude": np.abs(fft_vals)}
        except Exception: return null
    def export(self, iv, ov):
        sig = self._val(iv,"signal")
        return ["import numpy as np"], [
            f"_fft_sig = {sig}.flatten()",
            f"_fft_vals = np.fft.rfft(_fft_sig)",
            f"{ov['magnitude']} = np.abs(_fft_vals)",
            f"{ov['frequencies']} = np.fft.rfftfreq(len(_fft_sig))",
        ]
