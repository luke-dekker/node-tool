"""Tensor visualization nodes — accept torch.Tensor, render to IMAGE (RGB numpy array)."""

from __future__ import annotations
from typing import Any
import numpy as np
from core.node import BaseNode
from core.node import PortType


def _to_np(t) -> np.ndarray | None:
    """Convert tensor or array to float32 numpy array, or return None."""
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
    matplotlib.use("Agg")          # must be before pyplot import
    import matplotlib.pyplot as plt
    plt.switch_backend("agg")      # force even if pyplot already loaded
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0A0A10")
    ax.set_facecolor("#0F0F1A")
    return fig, ax


# ── Tensor Heatmap ────────────────────────────────────────────────────────────

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
            # collapse to 2-D: take first slice of every extra dim
            while arr.ndim > 2:
                arr = arr[0]
            fig, ax = _fig()
            im = ax.imshow(arr, cmap=inputs.get("cmap") or "viridis", aspect="auto")
            fig.colorbar(im, ax=ax)
            ax.set_title(inputs.get("title") or "Tensor Heatmap")
            return {"image": _render(fig)}
        except Exception as e:
            return {"image": None, "error": str(e)}


# ── Training Curve ────────────────────────────────────────────────────────────

class PlotTrainingCurveNode(BaseNode):
    type_name   = "pt_viz_training_curve"
    label       = "Training Curve"
    category    = "PyTorch"
    subcategory = "Plots"
    description = "Plot train (and optional val) loss list vs epochs."

    def _setup_ports(self) -> None:
        self.add_input("train_losses", PortType.ANY,    default=None,
                       description="List or array of training losses")
        self.add_input("val_losses",   PortType.ANY,    default=None,
                       description="List or array of validation losses (optional)")
        self.add_input("title",        PortType.STRING, default="Training Curve")
        self.add_output("image",       PortType.IMAGE)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        train = inputs.get("train_losses")
        val   = inputs.get("val_losses")
        if train is None:
            return {"image": None}
        try:
            train = list(train)
            fig, ax = _fig()
            ax.plot(range(1, len(train) + 1), train, color="cyan",   label="train")
            if val is not None:
                val = list(val)
                ax.plot(range(1, len(val) + 1), val,   color="orange", label="val")
                ax.legend()
            ax.set_title(inputs.get("title") or "Training Curve")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            return {"image": _render(fig)}
        except Exception:
            return {"image": None}


# ── Weight / Activation Histogram ─────────────────────────────────────────────

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


# ── Scatter (2-D tensor or two 1-D tensors) ───────────────────────────────────

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


# ── Show Image Tensor ──────────────────────────────────────────────────────────

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
            # CHW → HWC
            if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
                arr = arr.transpose(1, 2, 0)
            # single-channel → squeeze
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


# ── Weight Distribution (model-level) ─────────────────────────────────────────

class WeightHistogramNode(BaseNode):
    type_name   = "pt_viz_weight_hist"
    label       = "Weight Histogram"
    category    = "PyTorch"
    subcategory = "Plots"
    description = "Histogram of all trainable parameter values in a model."

    def _setup_ports(self) -> None:
        self.add_input("model",  PortType.MODULE, default=None)
        self.add_input("bins",   PortType.INT,    default=60)
        self.add_input("title",  PortType.STRING, default="Weight Distribution")
        self.add_output("image", PortType.IMAGE)
        self.add_output("model", PortType.MODULE, description="Pass-through model")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        model = inputs.get("model")
        if model is None:
            return {"image": None, "model": None}
        try:
            import torch
            all_weights = torch.cat([
                p.detach().cpu().flatten()
                for p in model.parameters()
                if p.requires_grad
            ]).numpy()
            bins  = int(inputs.get("bins")  or 60)
            title = inputs.get("title") or "Weight Distribution"
            fig, ax = _fig()
            ax.hist(all_weights, bins=bins, color="steelblue", edgecolor="none")
            ax.set_title(title)
            ax.set_xlabel("Weight value")
            ax.set_ylabel("Count")
            return {"image": _render(fig), "model": model}
        except Exception:
            return {"image": None, "model": model}
