"""Plot Tensor (heatmap) visualization node."""
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


class PlotTensorNode(BaseNode):
    type_name   = "pt_viz_tensor_heatmap"
    label       = "Plot Tensor"
    category    = "PyTorch"
    subcategory = "Plots"
    description = "Heatmap of a 2-D tensor (or first 2-D slice of higher-rank tensors)."

    def _setup_ports(self) -> None:
        self.add_input("tensor", PortType.TENSOR,  default=None)
        self.add_input("title",  PortType.STRING,  default="Tensor Heatmap")
        self.add_input("cmap",   PortType.STRING,  default="viridis")
        self.add_output("image", PortType.IMAGE)
        self.add_output("error", PortType.STRING,  description="Error message if render failed")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        arr = _to_np(inputs.get("tensor"))
        if arr is None:
            return {"image": None, "error": "no tensor"}
        try:
            while arr.ndim > 2:
                arr = arr[0]
            fig, ax = _fig()
            im = ax.imshow(arr, cmap=inputs.get("cmap") or "viridis", aspect="auto")
            fig.colorbar(im, ax=ax)
            ax.set_title(inputs.get("title") or "Tensor Heatmap")
            return {"image": _render(fig)}
        except Exception as e:
            return {"image": None, "error": str(e)}

    def export(self, iv, ov):
        return [], [f"# [{self.label}]: visualization nodes render inline - skipped in export"]
