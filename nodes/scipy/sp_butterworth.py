from core.node import BaseNode, PortType

class SpButterworthNode(BaseNode):
    type_name = "sp_butterworth"; label = "Butterworth Filter"; category = "SciPy"
    description = "scipy.signal.butter + sosfilt"
    def _setup_ports(self):
        self.add_input("signal", PortType.NDARRAY)
        self.add_input("cutoff", PortType.FLOAT, 0.1)
        self.add_input("order", PortType.INT, 5)
        self.add_input("btype", PortType.STRING, "low")
        self.add_output("filtered", PortType.NDARRAY)
    def execute(self, inputs):
        null = {"filtered": None}
        try:
            from scipy.signal import butter, sosfilt
            sig = inputs.get("signal")
            if sig is None: return null
            sos = butter(int(inputs.get("order",5)), float(inputs.get("cutoff",0.1)),
                         btype=inputs.get("btype","low") or "low", output="sos")
            return {"filtered": sosfilt(sos, sig.flatten())}
        except Exception: return null
    def export(self, iv, ov):
        sig,cut,ord_,bt = self._val(iv,"signal"),self._val(iv,"cutoff"),self._val(iv,"order"),self._val(iv,"btype")
        return ["from scipy.signal import butter, sosfilt"], [
            f"_sos = butter(int({ord_}), float({cut}), btype={bt}, output='sos')",
            f"{ov['filtered']} = sosfilt(_sos, {sig}.flatten())",
        ]
