"""Mel Spectrogram Transform node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class MelSpectrogramTransformNode(BaseNode):
    type_name   = "pt_mel_spectrogram_transform"
    label       = "Mel Spectrogram"
    category    = "Datasets"
    subcategory = "Transforms"
    description = "Convert raw audio waveform to mel spectrogram (torchaudio)."

    def _setup_ports(self):
        self.add_input("sample_rate", PortType.INT, default=16000)
        self.add_input("n_mels",      PortType.INT, default=64)
        self.add_input("n_fft",       PortType.INT, default=400)
        self.add_output("transform", PortType.TRANSFORM)

    def execute(self, inputs):
        try:
            import torchaudio.transforms as T
            return {"transform": T.MelSpectrogram(
                sample_rate=int(inputs.get("sample_rate") or 16000),
                n_mels=int(inputs.get("n_mels") or 64),
                n_fft=int(inputs.get("n_fft") or 400),
            )}
        except Exception:
            return {"transform": None}

    def export(self, iv, ov):
        sr = self._val(iv, 'sample_rate'); nm = self._val(iv, 'n_mels'); nf = self._val(iv, 'n_fft')
        return ["import torchaudio.transforms as T"], [
            f"{ov['transform']} = T.MelSpectrogram(sample_rate={sr}, n_mels={nm}, n_fft={nf})"
        ]
