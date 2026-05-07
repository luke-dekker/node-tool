"""Pad a batch of variable-length 1-D audio tensors into a (B, T_max) tensor
plus a (B,) lengths tensor — the contract CTC's `input_lengths` argument needs.

Why this isn't done in the dataset's collate: variable-length audio comes
out of `_collate_manifest` as a list of tensors when shapes don't stack.
This node pads them, exposes the lengths, and converts mel-spectrogram
hop into pred-time lengths if needed (set `frames_per_sample` to the
mel hop length, e.g. 160 for 10ms hops at 16 kHz, to convert sample-time
lengths into frame-time lengths matching the model's pred output).
"""
from __future__ import annotations
from core.node import BaseNode, PortType


class AudioPadCollateNode(BaseNode):
    type_name   = "pt_audio_pad_collate"
    label       = "Audio Pad / Collate"
    category    = "Data"
    subcategory = "Splits"
    description = (
        "Pad a list of variable-length 1-D audio tensors to (B, T_max)\n"
        "and emit per-sample lengths.\n\n"
        "  audio (list[Tensor] or Tensor) → padded (B, T_max), lengths (B,)\n\n"
        "Set `frames_per_sample` to convert sample-time lengths to model-time\n"
        "(post-MelSpectrogram) lengths. For 16 kHz audio with hop=160 (10 ms),\n"
        "frames_per_sample=160 → input_lengths matches the encoder's T axis."
    )

    def _setup_ports(self):
        self.add_input("audio",             PortType.ANY,    default=None,
                       description="List of 1-D tensors OR an already-stacked (B, T) tensor.")
        self.add_input("frames_per_sample", PortType.INT,    default=1,
                       description="Divide sample-time lengths by this to get frame-time "
                                   "lengths (use mel hop_length, e.g. 160 for 10 ms @ 16 kHz). "
                                   "1 = no conversion.")
        self.add_output("padded",   PortType.TENSOR,
                        description="(B, T_max) FloatTensor — pad value 0.0.")
        self.add_output("lengths",  PortType.TENSOR,
                        description="(B,) LongTensor of original (or frame-converted) lengths.")
        self.add_output("info",     PortType.STRING)

    def execute(self, inputs):
        import torch
        audio = inputs.get("audio")
        fps   = max(1, int(inputs.get("frames_per_sample") or 1))

        if audio is None:
            return {"padded": None, "lengths": None, "info": "no input"}

        # Normalize to a list of 1-D tensors
        if isinstance(audio, torch.Tensor):
            if audio.dim() == 1:
                tensors = [audio]
            elif audio.dim() == 2:
                # Already (B, T) — split into rows so the lengths line up
                # with the trailing-zero stripping logic below.
                tensors = [audio[i] for i in range(audio.shape[0])]
            else:
                return {"padded": None, "lengths": None,
                        "info": f"expected 1-D or 2-D audio tensor, got {audio.dim()}-D"}
        elif isinstance(audio, (list, tuple)):
            tensors = []
            for t in audio:
                if isinstance(t, torch.Tensor) and t.dim() == 1:
                    tensors.append(t)
                elif isinstance(t, torch.Tensor):
                    tensors.append(t.flatten())
                else:
                    return {"padded": None, "lengths": None,
                            "info": f"non-tensor item in audio list: {type(t).__name__}"}
        else:
            return {"padded": None, "lengths": None,
                    "info": f"expected list or tensor, got {type(audio).__name__}"}

        if not tensors:
            return {"padded": None, "lengths": None, "info": "empty batch"}

        sample_lengths = [int(t.shape[0]) for t in tensors]
        T_max = max(sample_lengths)
        B = len(tensors)
        # Use first tensor's dtype/device so padding stays consistent.
        ref = tensors[0]
        padded = torch.zeros(B, T_max, dtype=ref.dtype, device=ref.device)
        for i, t in enumerate(tensors):
            padded[i, :t.shape[0]] = t

        # Convert sample-time → frame-time. // (floor) is the right
        # convention for STFT/mel hop counts: a clip of N samples yields
        # N // hop frames (assuming center=False or with center=True the
        # extra frame from padding still divides cleanly enough for CTC).
        if fps == 1:
            lengths = torch.tensor(sample_lengths, dtype=torch.long)
            unit = "samples"
        else:
            lengths = torch.tensor([n // fps for n in sample_lengths], dtype=torch.long)
            unit = f"frames (hop={fps})"

        info = f"B={B} T_max={T_max} lengths {unit}: min={int(lengths.min())} max={int(lengths.max())}"
        return {"padded": padded, "lengths": lengths, "info": info}

    def export(self, iv, ov):
        fps = int(self.inputs["frames_per_sample"].default_value or 1)
        audio_var   = iv.get("audio") or "[]"
        padded_var  = ov.get("padded",  "_padded")
        lengths_var = ov.get("lengths", "_lengths")
        return ["import torch"], [
            f"_aud = {audio_var}",
            f"if isinstance(_aud, torch.Tensor):",
            f"    _ts = [_aud[i] for i in range(_aud.shape[0])] if _aud.dim() == 2 else [_aud]",
            f"else:",
            f"    _ts = list(_aud)",
            f"_lens = [int(t.shape[0]) for t in _ts]",
            f"_Tmax = max(_lens) if _lens else 0",
            f"{padded_var} = torch.zeros(len(_ts), _Tmax, dtype=_ts[0].dtype if _ts else torch.float32)",
            f"for _i, _t in enumerate(_ts):",
            f"    {padded_var}[_i, :_t.shape[0]] = _t",
            (f"{lengths_var} = torch.tensor([n // {fps} for n in _lens], dtype=torch.long)"
             if fps != 1 else
             f"{lengths_var} = torch.tensor(_lens, dtype=torch.long)"),
        ]
