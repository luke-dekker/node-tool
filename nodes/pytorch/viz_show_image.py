"""Show Image visualization node."""
from __future__ import annotations
from typing import Any
import numpy as np
from core.node import BaseNode, PortType


def _to_np(t) -> np.ndarray | None:
    if t is None:
        return None
    try:
        import torch
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().float().numpy()
    except ImportError:
        pass
    if isinstance(t, np.ndarray):
        return t.astype(np.float32)
    return None


def _render(fig) -> np.ndarray:
    import io
    from PIL import Image
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=90)
    buf.seek(0)
    img = np.array(Image.open(buf).convert("RGB"), dtype=np.uint8)
    plt.close(fig)
    return img


def _fig(figsize=(5, 3.5)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.switch_backend("agg")
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0A0A10")
    ax.set_facecolor("#0F0F1A")
    return fig, ax


class ShowImageNode(BaseNode):
    type_name   = "pt_viz_show_image"
    label       = "Show Image"
    category    = "PyTorch"
    subcategory = "Plots"
    description = (
        "Display an image tensor (C,H,W) or (H,W,C) or (H,W). "
        "Values auto-normalised to [0,1]."
    )

    def _setup_ports(self) -> None:
        self.add_input("tensor", PortType.TENSOR, default=None)
        self.add_input("title",  PortType.STRING, default="Image")
        self.add_output("image", PortType.IMAGE)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        arr = _to_np(inputs.get("tensor"))
        if arr is None:
            return {"image": None}
        try:
            # CHW -> HWC
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
                arr = arr.transpose(1, 2, 0)
            # single-channel -> squeeze
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = arr[:, :, 0]
            # normalise
            lo, hi = arr.min(), arr.max()
            if hi > lo:
                arr = (arr - lo) / (hi - lo)
            fig, ax = _fig(figsize=(4, 4))
            cmap = "gray" if arr.ndim == 2 else None
            ax.imshow(arr, cmap=cmap)
            ax.axis("off")
            ax.set_title(inputs.get("title") or "Image")
            return {"image": _render(fig)}
        except Exception:
            return {"image": None}

    def export(self, iv, ov):
        return [], [f"# [{self.label}]: visualization nodes render inline - skipped in export"]
