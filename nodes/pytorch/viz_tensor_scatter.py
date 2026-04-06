"""Tensor Scatter visualization node."""
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


class TensorScatterNode(BaseNode):
    type_name   = "pt_viz_tensor_scatter"
    label       = "Tensor Scatter"
    category    = "PyTorch"
    subcategory = "Plots"
    description = (
        "Scatter plot. Pass a (N,2) tensor to 'points', or separate 1-D "
        "tensors to 'x' and 'y'. Optionally colour by 'labels'."
    )

    def _setup_ports(self) -> None:
        self.add_input("points", PortType.TENSOR, default=None,
                       description="(N,2) tensor — columns are x and y")
        self.add_input("x",      PortType.TENSOR, default=None)
        self.add_input("y",      PortType.TENSOR, default=None)
        self.add_input("labels", PortType.TENSOR, default=None,
                       description="1-D integer labels for colouring")
        self.add_input("title",  PortType.STRING, default="Scatter")
        self.add_input("alpha",  PortType.FLOAT,  default=0.7)
        self.add_output("image", PortType.IMAGE)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        pts = _to_np(inputs.get("points"))
        xv  = _to_np(inputs.get("x"))
        yv  = _to_np(inputs.get("y"))
        lbl = _to_np(inputs.get("labels"))
        try:
            if pts is not None and pts.ndim == 2 and pts.shape[1] >= 2:
                xv, yv = pts[:, 0], pts[:, 1]
            if xv is None or yv is None:
                return {"image": None}
            alpha = float(inputs.get("alpha") or 0.7)
            title = inputs.get("title") or "Scatter"
            fig, ax = _fig()
            if lbl is not None:
                sc = ax.scatter(xv, yv, c=lbl.flatten(), cmap="tab10", alpha=alpha)
                fig.colorbar(sc, ax=ax)
            else:
                ax.scatter(xv, yv, alpha=alpha)
            ax.set_title(title)
            return {"image": _render(fig)}
        except Exception:
            return {"image": None}

    def export(self, iv, ov):
        return [], [f"# [{self.label}]: visualization nodes render inline - skipped in export"]
