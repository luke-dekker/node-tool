"""Tensor Histogram visualization node."""
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


class TensorHistogramNode(BaseNode):
    type_name   = "pt_viz_tensor_hist"
    label       = "Tensor Histogram"
    category    = "PyTorch"
    subcategory = "Plots"
    description = "Histogram of all values in a tensor — useful for weight/activation analysis."

    def _setup_ports(self) -> None:
        self.add_input("tensor", PortType.TENSOR, default=None)
        self.add_input("bins",   PortType.INT,    default=50)
        self.add_input("title",  PortType.STRING, default="Tensor Histogram")
        self.add_input("color",  PortType.STRING, default="steelblue")
        self.add_output("image", PortType.IMAGE)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        arr = _to_np(inputs.get("tensor"))
        if arr is None:
            return {"image": None}
        try:
            bins  = int(inputs.get("bins")  or 50)
            title = inputs.get("title") or "Tensor Histogram"
            color = inputs.get("color") or "steelblue"
            fig, ax = _fig()
            ax.hist(arr.flatten(), bins=bins, color=color, edgecolor="none")
            ax.set_title(title)
            ax.set_xlabel("Value")
            ax.set_ylabel("Count")
            return {"image": _render(fig)}
        except Exception:
            return {"image": None}

    def export(self, iv, ov):
        return [], [f"# [{self.label}]: visualization nodes render inline - skipped in export"]
