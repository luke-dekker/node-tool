"""Visualization — single multi-mode node renders matplotlib plots to IMAGE.

Replaces the prior 10 per-kind nodes (viz_bar / box / heatmap / hist /
image_grid / line / loss_curve / pca_2d / scatter / conf_matrix). Pick a
chart type from the `kind` dropdown; only the inputs that kind reads are
consulted, the rest are ignored.
"""
from __future__ import annotations
from typing import Any
import numpy as np

from core.node import BaseNode, PortType


CATEGORY = "Visualization"


_KINDS = [
    "line", "scatter", "bar", "hist", "heatmap", "box",
    "conf_matrix", "pca_2d", "loss_curve", "image_grid",
]


def _render_fig(fig) -> np.ndarray:
    """Render a matplotlib figure to an RGB uint8 numpy array."""
    import io
    from PIL import Image
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=90)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    plt.close(fig)
    return np.array(img, dtype=np.uint8)


def _dark_fig(figsize=(5, 3.5)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.switch_backend("agg")
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0A0A10")
    ax.set_facecolor("#0F0F1A")
    return fig, ax


class VizPlotNode(BaseNode):
    type_name   = "viz_plot"
    label       = "Plot"
    category    = CATEGORY
    subcategory = ""
    description = (
        "Render a matplotlib chart to an IMAGE. Pick the chart type via "
        "`kind`. Each kind reads a different subset of the inputs:\n"
        "  line       — data=x, data2=y, color, xlabel, ylabel\n"
        "  scatter    — data=x, data2=y, labels (color groups), alpha\n"
        "  bar        — data=values, labels (comma-separated), color\n"
        "  hist       — data=array, bins, color\n"
        "  heatmap    — data=matrix, cmap\n"
        "  box        — data=DataFrame\n"
        "  conf_matrix — data=matrix (annotated cells)\n"
        "  pca_2d     — data=X (Nx2+), labels (color groups)\n"
        "  loss_curve — data=train_losses, data2=val_losses (optional)\n"
        "  image_grid — data=images (B,C,H,W tensor), nrow"
    )

    def _setup_ports(self):
        self.add_input("kind",   PortType.STRING, default="line", choices=_KINDS)
        # Polymorphic: depending on `kind` this can be NDARRAY, DATAFRAME,
        # TENSOR, or a list. ANY keeps the wire compatible with any of them.
        self.add_input("data",   PortType.ANY,
                       description="Primary input (semantics per kind)")
        self.add_input("data2",  PortType.ANY,
                       description="Secondary input — y for line/scatter, val_losses for loss_curve")
        self.add_input("labels", PortType.ANY,
                       description="Color labels (NDARRAY for scatter/pca_2d) or "
                                   "comma-separated string (bar)")
        self.add_input("title",  PortType.STRING, default="")
        self.add_input("xlabel", PortType.STRING, default="")
        self.add_input("ylabel", PortType.STRING, default="")
        self.add_input("cmap",   PortType.STRING, default="viridis",
                       description="Colormap (heatmap)")
        self.add_input("bins",   PortType.INT,    default=30,
                       description="Bin count (hist)")
        self.add_input("color",  PortType.STRING, default="cyan",
                       description="Series color (line/bar/hist)")
        self.add_input("nrow",   PortType.INT,    default=8,
                       description="Grid columns (image_grid)")
        self.add_input("alpha",  PortType.FLOAT,  default=0.7,
                       description="Marker alpha (scatter)")
        self.add_output("image", PortType.IMAGE)

    def execute(self, inputs):
        kind = (inputs.get("kind") or "line").strip().lower()
        try:
            fn = _RENDERERS.get(kind)
            if fn is None:
                return {"image": None}
            return {"image": fn(inputs)}
        except Exception:
            return {"image": None}

    def export(self, iv, ov):
        return [], [f"# [{self.label}]: visualization nodes render inline - skipped in export"]


# ── Per-kind renderers ────────────────────────────────────────────────────────

def _r_line(inp: dict) -> Any:
    x, y = inp.get("data"), inp.get("data2")
    if x is None or y is None:
        return None
    fig, ax = _dark_fig()
    ax.plot(x, y, color=inp.get("color") or "cyan")
    ax.set_title(inp.get("title") or "Line Plot")
    ax.set_xlabel(inp.get("xlabel") or "x")
    ax.set_ylabel(inp.get("ylabel") or "y")
    return _render_fig(fig)


def _r_scatter(inp: dict) -> Any:
    x, y, labels = inp.get("data"), inp.get("data2"), inp.get("labels")
    if x is None or y is None:
        return None
    fig, ax = _dark_fig()
    alpha = float(inp.get("alpha") or 0.7)
    if labels is not None:
        sc = ax.scatter(x, y, c=labels, cmap="tab10", alpha=alpha)
        fig.colorbar(sc, ax=ax)
    else:
        ax.scatter(x, y, alpha=alpha)
    ax.set_title(inp.get("title") or "Scatter")
    if inp.get("xlabel"): ax.set_xlabel(inp["xlabel"])
    if inp.get("ylabel"): ax.set_ylabel(inp["ylabel"])
    return _render_fig(fig)


def _r_bar(inp: dict) -> Any:
    values = inp.get("data")
    if values is None:
        return None
    raw = inp.get("labels") or ""
    if isinstance(raw, str):
        labels = [s.strip() for s in raw.split(",") if s.strip()] or list(range(len(values)))
    else:
        labels = list(raw) if raw is not None else list(range(len(values)))
    fig, ax = _dark_fig()
    ax.bar(labels, values, color=inp.get("color") or "steelblue")
    ax.set_title(inp.get("title") or "Bar Chart")
    return _render_fig(fig)


def _r_hist(inp: dict) -> Any:
    arr = inp.get("data")
    if arr is None:
        return None
    fig, ax = _dark_fig()
    ax.hist(np.asarray(arr).flatten(),
            bins=int(inp.get("bins") or 30),
            color=inp.get("color") or "steelblue",
            edgecolor="none")
    ax.set_title(inp.get("title") or "Histogram")
    return _render_fig(fig)


def _r_heatmap(inp: dict) -> Any:
    m = inp.get("data")
    if m is None:
        return None
    fig, ax = _dark_fig()
    im = ax.imshow(m, cmap=inp.get("cmap") or "viridis", aspect="auto")
    fig.colorbar(im, ax=ax)
    ax.set_title(inp.get("title") or "Heatmap")
    return _render_fig(fig)


def _r_box(inp: dict) -> Any:
    df = inp.get("data")
    if df is None:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    fig.patch.set_facecolor("#0A0A10")
    ax.set_facecolor("#0F0F1A")
    df.select_dtypes(include="number").plot.box(ax=ax)
    ax.set_title(inp.get("title") or "Box Plot")
    return _render_fig(fig)


def _r_conf_matrix(inp: dict) -> Any:
    m = inp.get("data")
    if m is None:
        return None
    fig, ax = _dark_fig()
    ax.imshow(m, cmap="Blues")
    ax.set_title(inp.get("title") or "Confusion Matrix")
    mx = m.max() if hasattr(m, "max") else float(np.max(m))
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m[i, j]
            ax.text(j, i, str(v), ha="center", va="center",
                    color="white" if v < mx / 2 else "black", fontsize=10)
    return _render_fig(fig)


def _r_pca_2d(inp: dict) -> Any:
    X = inp.get("data")
    if X is None or X.shape[1] < 2:
        return None
    labels = inp.get("labels")
    fig, ax = _dark_fig()
    if labels is not None:
        sc = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", alpha=0.7)
        fig.colorbar(sc, ax=ax)
    else:
        ax.scatter(X[:, 0], X[:, 1], alpha=0.7)
    ax.set_title(inp.get("title") or "PCA 2D")
    ax.set_xlabel(inp.get("xlabel") or "PC1")
    ax.set_ylabel(inp.get("ylabel") or "PC2")
    return _render_fig(fig)


def _r_loss_curve(inp: dict) -> Any:
    train = inp.get("data")
    if train is None:
        return None
    val = inp.get("data2")
    fig, ax = _dark_fig()
    epochs = list(range(1, len(train) + 1))
    ax.plot(epochs, train, color="cyan", label="train")
    if val is not None:
        ax.plot(list(range(1, len(val) + 1)), val, color="orange", label="val")
        ax.legend()
    ax.set_title(inp.get("title") or "Loss Curve")
    ax.set_xlabel(inp.get("xlabel") or "Epoch")
    ax.set_ylabel(inp.get("ylabel") or "Loss")
    return _render_fig(fig)


def _r_image_grid(inp: dict) -> Any:
    images = inp.get("data")
    if images is None:
        return None
    import torch
    from torchvision.utils import make_grid
    if not isinstance(images, torch.Tensor):
        return None
    grid = make_grid(images, nrow=int(inp.get("nrow") or 8),
                     normalize=True, value_range=(0, 1))
    arr = (grid.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    fig, ax = _dark_fig(figsize=(8, 4))
    ax.imshow(arr)
    ax.axis("off")
    ax.set_title(inp.get("title") or "Image Grid")
    return _render_fig(fig)


_RENDERERS = {
    "line":         _r_line,
    "scatter":      _r_scatter,
    "bar":          _r_bar,
    "hist":         _r_hist,
    "heatmap":      _r_heatmap,
    "box":          _r_box,
    "conf_matrix":  _r_conf_matrix,
    "pca_2d":       _r_pca_2d,
    "loss_curve":   _r_loss_curve,
    "image_grid":   _r_image_grid,
}
