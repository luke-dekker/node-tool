"""Consolidated tensor visualization — replaces viz_tensor, viz_tensor_hist,
viz_tensor_scatter, viz_show_image."""
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


class TensorVizNode(BaseNode):
    type_name   = "pt_tensor_viz"
    label       = "Tensor Viz"
    category    = "Visualization"
    subcategory = ""
    description = (
        "Visualize a tensor: heatmap, histogram, scatter, or image. "
        "Select mode to choose the rendering style."
    )

    def _setup_ports(self) -> None:
        self.add_input("mode",   PortType.STRING, default="heatmap")
        self.add_input("tensor", PortType.TENSOR, default=None)
        self.add_input("title",  PortType.STRING, default="Tensor Viz")
        # heatmap
        self.add_input("cmap",   PortType.STRING, default="viridis")
        # histogram
        self.add_input("bins",   PortType.INT,    default=50)
        self.add_input("color",  PortType.STRING, default="steelblue")
        # scatter
        self.add_input("points", PortType.TENSOR, default=None,
                       description="(N,2) tensor — columns are x and y")
        self.add_input("x",      PortType.TENSOR, default=None)
        self.add_input("y",      PortType.TENSOR, default=None)
        self.add_input("labels", PortType.TENSOR, default=None,
                       description="1-D integer labels for colouring")
        self.add_input("alpha",  PortType.FLOAT,  default=0.7)
        self.add_output("image", PortType.IMAGE)
        self.add_output("error", PortType.STRING, description="Error message if render failed")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        mode = (inputs.get("mode") or "heatmap").strip().lower()
        title = inputs.get("title") or "Tensor Viz"
        try:
            if mode == "histogram":
                return self._histogram(inputs, title)
            elif mode == "scatter":
                return self._scatter(inputs, title)
            elif mode == "image":
                return self._image(inputs, title)
            else:
                return self._heatmap(inputs, title)
        except Exception as e:
            return {"image": None, "error": str(e)}

    def _heatmap(self, inputs, title):
        arr = _to_np(inputs.get("tensor"))
        if arr is None:
            return {"image": None, "error": "no tensor"}
        while arr.ndim > 2:
            arr = arr[0]
        fig, ax = _fig()
        im = ax.imshow(arr, cmap=inputs.get("cmap") or "viridis", aspect="auto")
        fig.colorbar(im, ax=ax)
        ax.set_title(title)
        return {"image": _render(fig), "error": None}

    def _histogram(self, inputs, title):
        arr = _to_np(inputs.get("tensor"))
        if arr is None:
            return {"image": None, "error": "no tensor"}
        bins = int(inputs.get("bins") or 50)
        color = inputs.get("color") or "steelblue"
        fig, ax = _fig()
        ax.hist(arr.flatten(), bins=bins, color=color, edgecolor="none")
        ax.set_title(title)
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        return {"image": _render(fig), "error": None}

    def _scatter(self, inputs, title):
        pts = _to_np(inputs.get("points"))
        xv = _to_np(inputs.get("x"))
        yv = _to_np(inputs.get("y"))
        lbl = _to_np(inputs.get("labels"))
        if pts is not None and pts.ndim == 2 and pts.shape[1] >= 2:
            xv, yv = pts[:, 0], pts[:, 1]
        if xv is None or yv is None:
            return {"image": None, "error": "scatter needs points or x+y"}
        alpha = float(inputs.get("alpha") or 0.7)
        fig, ax = _fig()
        if lbl is not None:
            sc = ax.scatter(xv, yv, c=lbl.flatten(), cmap="tab10", alpha=alpha)
            fig.colorbar(sc, ax=ax)
        else:
            ax.scatter(xv, yv, alpha=alpha)
        ax.set_title(title)
        return {"image": _render(fig), "error": None}

    def _image(self, inputs, title):
        arr = _to_np(inputs.get("tensor"))
        if arr is None:
            return {"image": None, "error": "no tensor"}
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = arr.transpose(1, 2, 0)
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = (arr - lo) / (hi - lo)
        fig, ax = _fig(figsize=(4, 4))
        cmap = "gray" if arr.ndim == 2 else None
        ax.imshow(arr, cmap=cmap)
        ax.axis("off")
        ax.set_title(title)
        return {"image": _render(fig), "error": None}

    def export(self, iv, ov):
        return [], [f"# [{self.label}]: visualization nodes render inline - skipped in export"]
