"""Mel Spectrogram Transform node — torchaudio.transforms.MelSpectrogram.

Two simultaneous outputs so the node works in both common patterns:

  Factory pattern: leave `waveform` unwired. The node emits the
  MelSpectrogram callable on `transform`; wire that into ApplyTransform
  or ComposeTransforms so it gets applied per-sample inside a Dataset.

  Inline pattern: wire a raw waveform tensor into `waveform`. The node
  also runs the transform here and emits the result on `spectrogram` —
  useful for live previews, one-off conversion, or building a graph that
  processes a single tensor without going through a DataLoader.

Both outputs are populated when `waveform` is wired; `spectrogram` stays
None in pure factory use.
"""
from __future__ import annotations
from core.node import BaseNode, PortType


class MelSpectrogramTransformNode(BaseNode):
    type_name   = "pt_mel_spectrogram_transform"
    label       = "Mel Spectrogram"
    category    = "Data"
    subcategory = "Transforms"
    description = (
        "Convert raw audio waveform to mel spectrogram (torchaudio.transforms.MelSpectrogram).\n"
        "  • Factory mode (waveform unwired): emit the transform callable on `transform`.\n"
        "  • Inline mode (waveform wired): also emit MelSpectrogram(waveform) on `spectrogram`."
    )

    def _setup_ports(self):
        self.add_input("waveform",    PortType.TENSOR, default=None, optional=True,
                       description="Optional raw audio tensor — shape (T,) / (C, T) / (B, C, T).")
        self.add_input("sample_rate", PortType.INT, default=16000)
        self.add_input("n_mels",      PortType.INT, default=64)
        self.add_input("n_fft",       PortType.INT, default=400)
        self.add_output("transform",   PortType.TRANSFORM,
                        description="torchaudio MelSpectrogram callable — wire into ApplyTransform.")
        self.add_output("spectrogram", PortType.TENSOR,
                        description="Inline output: MelSpectrogram(waveform). None unless waveform is wired.")
        self.add_output("info",        PortType.STRING,
                        description="Status: 'ok (factory)', 'ok (inline (B, n_mels, T))', "
                                    "or an error like 'torchaudio not installed — pip install torchaudio'.")

    def execute(self, inputs):
        # Distinguish "torchaudio missing" from "waveform unwired" so the
        # user can tell why `spectrogram` is None instead of staring at
        # silent failure. Surface the reason on `info`.
        try:
            import torchaudio.transforms as T
        except ImportError:
            return {"transform": None, "spectrogram": None,
                    "info": "torchaudio not installed — pip install torchaudio"}
        try:
            tf = T.MelSpectrogram(
                sample_rate=int(inputs.get("sample_rate") or 16000),
                n_mels=int(inputs.get("n_mels") or 64),
                n_fft=int(inputs.get("n_fft") or 400),
            )
        except Exception as exc:
            return {"transform": None, "spectrogram": None,
                    "info": f"failed to build MelSpectrogram: {exc}"}
        wav = inputs.get("waveform")
        if wav is None:
            return {"transform": tf, "spectrogram": None,
                    "info": "ok (factory) — wire `transform` into ApplyTransform"}
        try:
            spec = tf(wav)
            return {"transform": tf, "spectrogram": spec,
                    "info": f"ok (inline) — spectrogram shape {tuple(spec.shape)}"}
        except Exception as exc:
            return {"transform": tf, "spectrogram": None,
                    "info": f"transform built; inline call failed: {exc}"}

    def export(self, iv, ov):
        sr = self._val(iv, 'sample_rate')
        nm = self._val(iv, 'n_mels')
        nf = self._val(iv, 'n_fft')
        tf_var   = ov.get("transform",   "_mel_tf")
        spec_var = ov.get("spectrogram", "_mel_spec")
        wav      = iv.get("waveform")   # connected variable name, or None

        lines = [f"{tf_var} = T.MelSpectrogram(sample_rate={sr}, n_mels={nm}, n_fft={nf})"]
        if wav:
            lines.append(f"{spec_var} = {tf_var}({wav})")
        else:
            lines.append(f"{spec_var} = None  # waveform unwired — factory mode")
        return ["import torchaudio.transforms as T"], lines
