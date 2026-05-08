"""Mel Spectrogram Transform node — torchaudio.transforms.MelSpectrogram.

Three modes, picked by what (if anything) is wired into `waveform`:

  Factory pattern: leave `waveform` unwired. The node emits the
  MelSpectrogram callable on `transform`; wire that into ApplyTransform
  or ComposeTransforms so it gets applied per-sample inside a Dataset.

  Inline single-tensor: wire a raw waveform tensor (1-D, (C, T), or
  pre-batched (B, T) / (B, C, T)) into `waveform`. The node runs the
  transform and emits the result on `spectrogram`. `lengths` is None
  because we don't know per-sample valid spans for a pre-stacked batch.

  Inline batch (variable-length): wire a list[Tensor] of 1-D audio
  clips. The node pads them to a common length, runs the transform,
  and emits both `spectrogram` (B, n_mels, T_max_frames) AND `lengths`
  (B,) — frame-time per-sample lengths derived from each clip's
  original sample count and the hop length. This is the input-length
  vector CTC's `input_lengths` argument needs, so no separate
  pad/collate node is needed in front.
"""
from __future__ import annotations
from core.node import BaseNode, PortType


class MelSpectrogramTransformNode(BaseNode):
    type_name   = "pt_mel_spectrogram_transform"
    label       = "Mel Spectrogram"
    category    = "Data"
    subcategory = "Transforms"
    description = (
        "Convert raw audio to mel spectrogram (torchaudio.transforms.MelSpectrogram).\n"
        "  • Factory mode (waveform unwired): emit the transform callable on `transform`.\n"
        "  • Inline mode (single Tensor): emit MelSpectrogram(waveform) on `spectrogram`.\n"
        "  • Inline batch (list[Tensor] of variable-length 1-D clips):\n"
        "      pad → transform → emit `spectrogram` + per-sample frame `lengths`\n"
        "      (the input_lengths vector CTC loss needs)."
    )

    def _setup_ports(self):
        # ANY (not TENSOR) so a list[Tensor] from the dataset can wire in
        # directly without a separate pad/collate node.
        self.add_input("waveform",    PortType.ANY, default=None, optional=True,
                       description="Raw audio: 1-D Tensor, (C,T), (B,T), (B,C,T), "
                                   "OR list[Tensor] of variable-length 1-D clips.")
        self.add_input("sample_rate", PortType.INT, default=16000)
        self.add_input("n_mels",      PortType.INT, default=64)
        self.add_input("n_fft",       PortType.INT, default=400)
        # hop_length defaults to None → torchaudio uses win_length // 2 (= n_fft / 2).
        # Common ASR setups use 160 (10 ms @ 16 kHz). Also used to convert
        # sample-time clip lengths to frame-time `lengths` in batch mode.
        self.add_input("hop_length",  PortType.INT, default=0,
                       description="Hop between FFT windows (samples). 0 = torchaudio "
                                   "default (n_fft // 2). Set to 160 for 10 ms @ 16 kHz.")
        self.add_output("transform",   PortType.TRANSFORM,
                        description="torchaudio MelSpectrogram callable — wire into ApplyTransform.")
        self.add_output("spectrogram", PortType.TENSOR,
                        description="MelSpectrogram(waveform). None unless waveform is wired.")
        self.add_output("lengths",     PortType.TENSOR,
                        description="(B,) LongTensor frame-time per-sample lengths. "
                                    "Populated only in batch mode (list[Tensor] input). "
                                    "Wire into LossCompute.input_lengths for CTC.")
        self.add_output("info",        PortType.STRING,
                        description="Status / shape summary or error reason.")

    def execute(self, inputs):
        try:
            import torchaudio.transforms as T
        except ImportError:
            return {"transform": None, "spectrogram": None, "lengths": None,
                    "info": "torchaudio not installed — pip install torchaudio"}
        try:
            n_fft = int(inputs.get("n_fft") or 400)
            kwargs = dict(
                sample_rate=int(inputs.get("sample_rate") or 16000),
                n_mels=int(inputs.get("n_mels") or 64),
                n_fft=n_fft,
            )
            hop = int(inputs.get("hop_length") or 0)
            if hop > 0:
                kwargs["hop_length"] = hop
            tf = T.MelSpectrogram(**kwargs)
        except Exception as exc:
            return {"transform": None, "spectrogram": None, "lengths": None,
                    "info": f"failed to build MelSpectrogram: {exc}"}

        wav = inputs.get("waveform")
        if wav is None:
            return {"transform": tf, "spectrogram": None, "lengths": None,
                    "info": "ok (factory) — wire `transform` into ApplyTransform"}

        # Effective hop for length math: explicit hop, else torchaudio's default.
        eff_hop = hop if hop > 0 else (n_fft // 2)
        try:
            import torch
            sample_lens: list[int] | None = None

            # Variable-length batch → pad in sample-time before running mel.
            if isinstance(wav, (list, tuple)):
                tensors = [t if t.dim() == 1 else t.flatten() for t in wav]
                sample_lens = [int(t.shape[0]) for t in tensors]
                T_max = max(sample_lens) if sample_lens else 0
                ref = tensors[0]
                stacked = torch.zeros(len(tensors), T_max, dtype=ref.dtype, device=ref.device)
                for i, t in enumerate(tensors):
                    stacked[i, :t.shape[0]] = t
                wav = stacked

            spec = tf(wav)

            if sample_lens is not None:
                # Floor division matches torchaudio's frame count for non-center
                # mode and is the safe under-estimate for center=True. Clamp to
                # the actual mel T axis so CTC's lengths can never exceed pred.
                T_frames = spec.shape[-1]
                lens = torch.tensor(
                    [min(n // eff_hop, T_frames) for n in sample_lens],
                    dtype=torch.long,
                )
                return {"transform": tf, "spectrogram": spec, "lengths": lens,
                        "info": f"ok (inline batch) — spec {tuple(spec.shape)} "
                                f"hop={eff_hop} lens min={int(lens.min())} max={int(lens.max())}"}
            return {"transform": tf, "spectrogram": spec, "lengths": None,
                    "info": f"ok (inline) — spec shape {tuple(spec.shape)}"}
        except Exception as exc:
            return {"transform": tf, "spectrogram": None, "lengths": None,
                    "info": f"transform built; inline call failed: {exc}"}

    def export(self, iv, ov):
        sr  = self._val(iv, 'sample_rate')
        nm  = self._val(iv, 'n_mels')
        nf  = self._val(iv, 'n_fft')
        hop = int(self.inputs["hop_length"].default_value or 0)
        eff_hop = hop if hop > 0 else (int(nf) // 2 if isinstance(nf, int) else None)
        tf_var   = ov.get("transform",   "_mel_tf")
        spec_var = ov.get("spectrogram", "_mel_spec")
        lens_var = ov.get("lengths",     "_mel_lens")
        wav      = iv.get("waveform")

        ctor = f"T.MelSpectrogram(sample_rate={sr}, n_mels={nm}, n_fft={nf}"
        if hop > 0:
            ctor += f", hop_length={hop}"
        ctor += ")"
        lines = [f"{tf_var} = {ctor}"]
        if not wav:
            lines += [f"{spec_var} = None  # waveform unwired — factory mode",
                      f"{lens_var} = None"]
            return ["import torchaudio.transforms as T"], lines

        # Inline: handle list[Tensor] (pad + lengths) AND single Tensor uniformly.
        lines += [
            f"_w = {wav}",
            f"_sl = None",
            f"if isinstance(_w, (list, tuple)):",
            f"    _ts = [t if t.dim() == 1 else t.flatten() for t in _w]",
            f"    _sl = [int(t.shape[0]) for t in _ts]",
            f"    _Tmax = max(_sl) if _sl else 0",
            f"    _w = torch.zeros(len(_ts), _Tmax, dtype=_ts[0].dtype if _ts else torch.float32)",
            f"    for _i, _t in enumerate(_ts):",
            f"        _w[_i, :_t.shape[0]] = _t",
            f"{spec_var} = {tf_var}(_w)",
            f"if _sl is not None:",
            f"    _Tf = {spec_var}.shape[-1]",
            f"    {lens_var} = torch.tensor([min(n // {eff_hop}, _Tf) for n in _sl], dtype=torch.long)",
            f"else:",
            f"    {lens_var} = None",
        ]
        return ["import torch", "import torchaudio.transforms as T"], lines
