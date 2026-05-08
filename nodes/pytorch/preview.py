"""PreviewNode — wire anything in, inspector shows it.

One generic "what did this thing produce?" node. Wire any port into
`value`; on each execute the node caches whatever it received and
generates an `inspector_spec` so the React Inspector renders it
appropriately for its type:

  - str                → quoted text
  - list[str]          → numbered list (truncated past 8 items)
  - 1-D float tensor   → shape/dtype/range + ▶Play action
                         (treated as audio @ `sample_rate`)
  - 2-D tensor (H, W)  → shape/dtype/range + image preview
                         (grayscale, normalised to 0..255)
  - 3-D tensor (C,H,W) → image preview (RGB if C==3, else channel mean)
                         OR — if `vocab` is non-empty — CTC greedy decode
                         of (B, T, vocab) logits into text (collapse runs
                         of dupes, drop blanks). Sits in the loss path as
                         a passthrough so it runs every training step.
  - 4-D tensor batch   → image preview of sample 0
  - any other tensor   → shape/dtype/min/max/mean + first 8 values
  - everything else    → repr (truncated past 200 chars)

Sits as a passthrough on the graph: `value` flows straight to `value_out`
unchanged, so this node can sit between any two wired nodes without
disrupting the loss path. Cached value persists between ticks — when
graph.execute() runs after training has paused, the inspector still
shows the most recently observed value.
"""
from __future__ import annotations
import base64
import io
from typing import Any
from core.node import BaseNode, InspectorSpec, PortType


_LIST_PREVIEW = 8           # list / first-N values shown in `lines`
_TEXT_TRUNCATE = 200        # chars per line before truncation
_AUDIO_MIN_SAMPLES = 1000   # below this, treat 1-D tensor as just a vector


def _short(s: str, n: int = _TEXT_TRUNCATE) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def _ctc_greedy_decode(logits, vocab: str, blank_idx: int = 0) -> list[str]:
    """logits: (B, T, C) or (T, B, C) tensor → one decoded string per sample.

    Greedy decode: argmax along C, drop runs of consecutive duplicates,
    then drop the blank token (index 0 by convention). Index 0 is reserved
    for blank, indices 1..N map to vocab[0..N-1].
    """
    import torch
    if logits is None or logits.dim() != 3:
        return []
    # Heuristic from LossCompute: smaller dim is usually batch.
    if logits.shape[0] > logits.shape[1] and logits.shape[1] < 64:
        logits = logits.transpose(0, 1).contiguous()
    i2c = {i + 1: ch for i, ch in enumerate(vocab)}
    out: list[str] = []
    with torch.no_grad():
        ids = logits.argmax(dim=-1)
        for row in ids.tolist():
            collapsed: list[int] = []
            prev = None
            for v in row:
                if v != prev:
                    collapsed.append(v)
                    prev = v
            out.append("".join(i2c.get(v, "") for v in collapsed if v != blank_idx))
    return out


def _format_lines(
    value: Any, sample_rate: int, vocab: str = "", blank_idx: int = 0,
) -> tuple[list[str], str]:
    """Return (lines, kind) describing `value` for the inspector spec.

    `kind` is one of: text, text_list, audio, image, tensor, scalar, repr,
    none — used by `inspector_spec()` to decide which actions to attach.
    When `vocab` is non-empty and `value` is a 3-D float tensor, treat it
    as CTC logits and append decoded text lines.
    """
    if value is None:
        return ["(no value yet — wire something into `value`)"], "none"
    if isinstance(value, str):
        return [f'"{_short(value)}"'], "text"
    if isinstance(value, (list, tuple)) and value and all(isinstance(v, str) for v in value):
        head = list(value)[:_LIST_PREVIEW]
        more = f"  (+{len(value) - len(head)} more)" if len(value) > len(head) else ""
        return [f"[{i}] {_short(v)}" for i, v in enumerate(head)] + ([more] if more else []), "text_list"

    # Try torch tensor first, then numpy ndarray.
    try:
        import torch
        is_tensor = isinstance(value, torch.Tensor)
    except ImportError:
        is_tensor = False
    try:
        import numpy as np
        is_ndarray = isinstance(value, np.ndarray)
    except ImportError:
        is_ndarray = False

    if is_tensor or is_ndarray:
        shape = tuple(value.shape)
        dtype = str(value.dtype)
        if is_tensor:
            t = value.detach().to(dtype=__import__("torch").float32).flatten()
        else:
            t = value.astype("float32").flatten()
        if t.size if is_ndarray else t.numel():
            head_n = min(_LIST_PREVIEW, t.size if is_ndarray else int(t.numel()))
            head_vals = (t[:head_n].tolist() if is_ndarray else t[:head_n].tolist())
            mn = float(t.min()); mx = float(t.max())
            mean = float(t.mean()) if (t.size if is_ndarray else t.numel()) > 0 else 0.0
            lines = [
                f"{'tensor' if is_tensor else 'ndarray'}  shape={list(shape)}  dtype={dtype}",
                f"min={mn:.4g}  max={mx:.4g}  mean={mean:.4g}",
                f"head: {[round(v, 4) for v in head_vals]}",
            ]
        else:
            lines = [f"empty {('tensor' if is_tensor else 'ndarray')} shape={list(shape)} dtype={dtype}"]

        # Type detection: 1-D long tensor → audio
        if len(shape) == 1 and (shape[0] >= _AUDIO_MIN_SAMPLES) and ("float" in dtype):
            secs = shape[0] / max(sample_rate, 1)
            lines.append(f"audio: {secs:.2f}s @ {sample_rate} Hz")
            return lines, "audio"

        # 2-D float tensor → image (grayscale)
        if len(shape) == 2 and "float" in dtype:
            return lines, "image"
        # 3-D tensor (B, T, C) with a vocab → CTC greedy decode → text.
        # Only kicks in for float tensors with a non-empty vocab; without
        # vocab, falls through to image preview as before.
        if len(shape) == 3 and "float" in dtype and vocab and is_tensor:
            try:
                decoded = _ctc_greedy_decode(value, vocab, blank_idx=blank_idx)
                if decoded:
                    head = decoded[:_LIST_PREVIEW]
                    more = (f"  (+{len(decoded) - len(head)} more)"
                            if len(decoded) > len(head) else "")
                    lines.append(f"ctc decoded ({len(decoded)} samples):")
                    lines.extend(f"  [{i}] {_short(s)!r}" for i, s in enumerate(head))
                    if more:
                        lines.append(more)
                    return lines, "text_list"
            except Exception as exc:
                lines.append(f"ctc decode failed: {exc}")
        # 3-D tensor (C,H,W) or (H,W,C) → image
        if len(shape) == 3:
            return lines, "image"
        # 4-D batched (B,C,H,W) → image (sample 0)
        if len(shape) == 4:
            return lines, "image"
        return lines, "tensor"

    if isinstance(value, (int, float, bool)):
        return [f"{type(value).__name__}: {value}"], "scalar"
    return [_short(repr(value))], "repr"


class PreviewNode(BaseNode):
    type_name   = "pt_preview"
    label       = "Preview"
    category    = "Visualization"
    subcategory = ""
    description = (
        "Render whatever was last wired into `value` in the inspector.\n"
        "  • str / list[str]   → quoted text\n"
        "  • float tensor 1D   → audio (▶Play, length × sample_rate)\n"
        "  • float tensor 2-4D → image preview\n"
        "  • other tensors     → shape + dtype + head values\n"
        "Passthrough output so this node can sit anywhere on the graph."
    )

    def __init__(self):
        self._last_value: Any = None
        self._last_kind: str = "none"
        self._last_lines: list[str] = []
        self._sample_rate: int = 16000
        self._vocab: str = ""
        self._blank_idx: int = 0
        super().__init__()

    def _setup_ports(self):
        self.add_input("value",       PortType.ANY, default=None,
                       description="Anything — text, tensor, audio waveform, image, list, scalar.")
        self.add_input("sample_rate", PortType.INT, default=16000,
                       description="Sample rate for audio playback (1-D float tensor inputs).")
        self.add_input("vocab",       PortType.STRING, default="", optional=True,
                       description="If set, decode 3-D float tensors (B, T, C) as CTC logits "
                                   "into text using this vocabulary (blank at index 0).")
        self.add_input("blank_idx",   PortType.INT, default=0, optional=True,
                       description="CTC blank token index (default 0).")
        self.add_output("value_out",  PortType.ANY,
                        description="Passthrough of `value` — wire downstream just like the input was direct.")
        self.add_output("info",       PortType.STRING,
                        description="One-line text summary (kind + shape).")

    def execute(self, inputs):
        value = inputs.get("value")
        # Cache only when we received a fresh value. Polling ticks during
        # training have None upstream and shouldn't clobber the cache.
        if value is not None:
            self._last_value = value
            self._sample_rate = max(1, int(inputs.get("sample_rate") or 16000))
        self._vocab     = str(inputs.get("vocab") or "")
        self._blank_idx = int(inputs.get("blank_idx") or 0)
        # Always recompute the lines from cached value so the inspector
        # gets the latest formatting after a sample-rate change.
        lines, kind = _format_lines(
            self._last_value, self._sample_rate,
            vocab=self._vocab, blank_idx=self._blank_idx,
        )
        self._last_lines = lines
        self._last_kind  = kind
        info = lines[0] if lines else "(empty)"
        return {"value_out": value, "info": info}

    # ── Inspector integration ──────────────────────────────────────────────

    def inspector_spec(self):
        actions: list[tuple[str, str]] = []
        if self._last_kind == "audio":
            actions.append(("▶ Play",        "play_audio"))
        if self._last_kind == "image":
            actions.append(("Show image",    "render_image"))
        if self._last_value is not None:
            actions.append(("Refresh",       "refresh"))
        return InspectorSpec(
            section="Preview",
            lines=list(self._last_lines),
            actions=actions,
        )

    # ── Action methods (called via server's invoke_node_action RPC) ────────

    def refresh(self, app=None) -> dict:
        """No-op trigger so the inspector re-fetches the spec lines."""
        lines, kind = _format_lines(
            self._last_value, self._sample_rate,
            vocab=self._vocab, blank_idx=self._blank_idx,
        )
        self._last_lines = lines
        self._last_kind  = kind
        return {"ok": True, "lines": lines, "kind": kind}

    def play_audio(self, app=None) -> dict:
        """Encode the cached 1-D audio tensor as a data: URL the browser can play."""
        v = self._last_value
        if v is None:
            return {"error": "no value cached"}
        try:
            import torch
            import numpy as np
            import soundfile as sf
            if isinstance(v, torch.Tensor):
                arr = v.detach().to(dtype=torch.float32).cpu().numpy()
            elif isinstance(v, np.ndarray):
                arr = v.astype(np.float32, copy=False)
            else:
                return {"error": f"play_audio: unsupported value type {type(v).__name__}"}
            if arr.ndim != 1:
                # Fall back to first row if it's (B, T) or (C, T).
                arr = arr.reshape(-1) if arr.size <= 5_000_000 else arr.flatten()[:5_000_000]
            buf = io.BytesIO()
            sf.write(buf, arr, self._sample_rate, format="WAV", subtype="FLOAT")
            data = base64.b64encode(buf.getvalue()).decode("ascii")
            secs = len(arr) / max(self._sample_rate, 1)
            return {
                "kind":      "audio",
                "audio_url": f"data:audio/wav;base64,{data}",
                "info":      f"{secs:.2f}s @ {self._sample_rate} Hz, {len(arr)} samples",
            }
        except Exception as exc:
            return {"error": f"play_audio failed: {exc}"}

    def render_image(self, app=None) -> dict:
        """Encode the cached 2-D / 3-D / 4-D tensor as a PNG data URL."""
        v = self._last_value
        if v is None:
            return {"error": "no value cached"}
        try:
            import torch
            import numpy as np
            from PIL import Image
            if isinstance(v, torch.Tensor):
                arr = v.detach().to(dtype=torch.float32).cpu().numpy()
            elif isinstance(v, np.ndarray):
                arr = v.astype(np.float32, copy=False)
            else:
                return {"error": f"render_image: unsupported type {type(v).__name__}"}
            # Squeeze batch + reorder channels-first → channels-last.
            if arr.ndim == 4:
                arr = arr[0]
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[0] < arr.shape[1]:
                arr = np.transpose(arr, (1, 2, 0))
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = arr[:, :, 0]
            # Normalise to 0..255 uint8.
            mn, mx = float(arr.min()), float(arr.max())
            if mx > mn:
                arr = (arr - mn) / (mx - mn)
            else:
                arr = np.zeros_like(arr)
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(arr)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            data = base64.b64encode(buf.getvalue()).decode("ascii")
            return {
                "kind":      "image",
                "image_url": f"data:image/png;base64,{data}",
                "info":      f"image {arr.shape}",
            }
        except Exception as exc:
            return {"error": f"render_image failed: {exc}"}

    def export(self, iv, ov):
        v = iv.get("value") or "None"
        return [], [
            f"# Preview: passthrough — show {v} in the live inspector",
            f"{ov.get('value_out', '_v')} = {v}",
            f"{ov.get('info', '_info')} = repr({v})[:200]",
        ]
