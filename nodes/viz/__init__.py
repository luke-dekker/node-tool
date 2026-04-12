"""Visualization nodes — render matplotlib plots to IMAGE (RGB numpy array)."""
from __future__ import annotations
from core.node import BaseNode
from core.node import PortType
import numpy as np

CATEGORY = "Visualization"


class _VizBase(BaseNode):
    """Shared base for viz nodes — inline rendering only, no code export."""
    def export(self, iv, ov):
        return [], [f"# [{self.label}]: visualization nodes render inline - skipped in export"]


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
    """Create a dark-themed matplotlib figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.switch_backend("agg")
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0A0A10")
    ax.set_facecolor("#0F0F1A")
    return fig, ax


# ── Line Plot ─────────────────────────────────────────────────────────────────

class VizLineNode(_VizBase):
    type_name = "viz_line"
    label = "Line Plot"
    category = CATEGORY
    description = "Line plot of y vs x"

    def _setup_ports(self):
        self.add_input("x",      PortType.NDARRAY)
        self.add_input("y",      PortType.NDARRAY)
        self.add_input("title",  PortType.STRING, "Line Plot")
        self.add_input("xlabel", PortType.STRING, "x")
        self.add_input("ylabel", PortType.STRING, "y")
        self.add_input("color",  PortType.STRING, "cyan")
        self.add_output("image", PortType.IMAGE)

    def execute(self, inputs):
        null = {"image": None}
        try:
            x      = inputs.get("x")
            y      = inputs.get("y")
            title  = inputs.get("title",  "Line Plot") or "Line Plot"
            xlabel = inputs.get("xlabel", "x")         or "x"
            ylabel = inputs.get("ylabel", "y")         or "y"
            color  = inputs.get("color",  "cyan")      or "cyan"
            if x is None or y is None:
                return null
            fig, ax = _dark_fig()
            ax.plot(x, y, color=color)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            return {"image": _render_fig(fig)}
        except Exception:
            return null


# ── Scatter Plot ──────────────────────────────────────────────────────────────

class VizScatterNode(_VizBase):
    type_name = "viz_scatter"
    label = "Scatter Plot"
    category = CATEGORY
    description = "Scatter plot — colored by labels if provided"

    def _setup_ports(self):
        self.add_input("x",      PortType.NDARRAY)
        self.add_input("y",      PortType.NDARRAY)
        self.add_input("labels", PortType.NDARRAY)
        self.add_input("title",  PortType.STRING, "Scatter")
        self.add_input("alpha",  PortType.FLOAT,  0.7)
        self.add_output("image", PortType.IMAGE)

    def execute(self, inputs):
        null = {"image": None}
        try:
            x      = inputs.get("x")
            y      = inputs.get("y")
            labels = inputs.get("labels")
            title  = inputs.get("title", "Scatter") or "Scatter"
            alpha  = inputs.get("alpha", 0.7)
            if x is None or y is None:
                return null
            fig, ax = _dark_fig()
            if labels is not None:
                scatter = ax.scatter(x, y, c=labels, alpha=float(alpha), cmap="tab10")
                fig.colorbar(scatter, ax=ax)
            else:
                ax.scatter(x, y, alpha=float(alpha))
            ax.set_title(title)
            return {"image": _render_fig(fig)}
        except Exception:
            return null


# ── Bar Chart ─────────────────────────────────────────────────────────────────

class VizBarNode(_VizBase):
    type_name = "viz_bar"
    label = "Bar Chart"
    category = CATEGORY
    description = "Bar chart — labels is comma-separated string"

    def _setup_ports(self):
        self.add_input("values", PortType.NDARRAY)
        self.add_input("labels", PortType.STRING, "")
        self.add_input("title",  PortType.STRING, "Bar Chart")
        self.add_input("color",  PortType.STRING, "steelblue")
        self.add_output("image", PortType.IMAGE)

    def execute(self, inputs):
        null = {"image": None}
        try:
            values    = inputs.get("values")
            labels_str = inputs.get("labels", "") or ""
            title     = inputs.get("title",  "Bar Chart") or "Bar Chart"
            color     = inputs.get("color",  "steelblue") or "steelblue"
            if values is None:
                return null
            labels = ([l.strip() for l in labels_str.split(",") if l.strip()]
                      if labels_str.strip() else list(range(len(values))))
            fig, ax = _dark_fig()
            ax.bar(labels, values, color=color)
            ax.set_title(title)
            return {"image": _render_fig(fig)}
        except Exception:
            return null


# ── Histogram ─────────────────────────────────────────────────────────────────

class VizHistNode(_VizBase):
    type_name = "viz_hist"
    label = "Histogram"
    category = CATEGORY
    description = "Histogram of array"

    def _setup_ports(self):
        self.add_input("array", PortType.NDARRAY)
        self.add_input("bins",  PortType.INT,    30)
        self.add_input("title", PortType.STRING, "Histogram")
        self.add_input("color", PortType.STRING, "steelblue")
        self.add_output("image", PortType.IMAGE)

    def execute(self, inputs):
        null = {"image": None}
        try:
            arr   = inputs.get("array")
            bins  = inputs.get("bins",  30)
            title = inputs.get("title", "Histogram") or "Histogram"
            color = inputs.get("color", "steelblue")  or "steelblue"
            if arr is None:
                return null
            fig, ax = _dark_fig()
            ax.hist(arr.flatten(), bins=int(bins), color=color, edgecolor="none")
            ax.set_title(title)
            return {"image": _render_fig(fig)}
        except Exception:
            return null


# ── Heatmap ───────────────────────────────────────────────────────────────────

class VizHeatmapNode(_VizBase):
    type_name = "viz_heatmap"
    label = "Heatmap"
    category = CATEGORY
    description = "Heatmap via imshow"

    def _setup_ports(self):
        self.add_input("matrix", PortType.NDARRAY)
        self.add_input("title",  PortType.STRING, "Heatmap")
        self.add_input("cmap",   PortType.STRING, "viridis")
        self.add_output("image", PortType.IMAGE)

    def execute(self, inputs):
        null = {"image": None}
        try:
            import matplotlib.pyplot as plt
            matrix = inputs.get("matrix")
            title  = inputs.get("title", "Heatmap") or "Heatmap"
            cmap   = inputs.get("cmap",  "viridis") or "viridis"
            if matrix is None:
                return null
            fig, ax = _dark_fig()
            im = ax.imshow(matrix, cmap=cmap, aspect="auto")
            fig.colorbar(im, ax=ax)
            ax.set_title(title)
            return {"image": _render_fig(fig)}
        except Exception:
            return null


# ── Box Plot ──────────────────────────────────────────────────────────────────

class VizBoxNode(_VizBase):
    type_name = "viz_box"
    label = "Box Plot"
    category = CATEGORY
    description = "Box plot of DataFrame columns"

    def _setup_ports(self):
        self.add_input("df",    PortType.DATAFRAME)
        self.add_input("title", PortType.STRING, "Box Plot")
        self.add_output("image", PortType.IMAGE)

    def execute(self, inputs):
        null = {"image": None}
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            df    = inputs.get("df")
            title = inputs.get("title", "Box Plot") or "Box Plot"
            if df is None:
                return null
            plt.style.use("dark_background")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            fig.patch.set_facecolor("#0A0A10")
            ax.set_facecolor("#0F0F1A")
            num_df = df.select_dtypes(include="number")
            num_df.plot.box(ax=ax)
            ax.set_title(title)
            return {"image": _render_fig(fig)}
        except Exception:
            return null


# ── Confusion Matrix Viz ──────────────────────────────────────────────────────

class VizConfMatrixNode(_VizBase):
    type_name = "viz_conf_matrix"
    label = "Confusion Matrix"
    category = CATEGORY
    description = "Heatmap of confusion matrix with annotations"

    def _setup_ports(self):
        self.add_input("matrix", PortType.NDARRAY)
        self.add_input("title",  PortType.STRING, "Confusion Matrix")
        self.add_output("image", PortType.IMAGE)

    def execute(self, inputs):
        null = {"image": None}
        try:
            matrix = inputs.get("matrix")
            title  = inputs.get("title", "Confusion Matrix") or "Confusion Matrix"
            if matrix is None:
                return null
            fig, ax = _dark_fig()
            im = ax.imshow(matrix, cmap="Blues")
            ax.set_title(title)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    ax.text(j, i, str(matrix[i, j]),
                            ha="center", va="center",
                            color="white" if matrix[i, j] < matrix.max() / 2 else "black",
                            fontsize=10)
            return {"image": _render_fig(fig)}
        except Exception:
            return null


# ── PCA 2D Scatter ────────────────────────────────────────────────────────────

class VizPCANode(_VizBase):
    type_name = "viz_pca_2d"
    label = "PCA 2D"
    category = CATEGORY
    description = "Scatter of first 2 PCA dims colored by labels"

    def _setup_ports(self):
        self.add_input("X",      PortType.NDARRAY)
        self.add_input("labels", PortType.NDARRAY)
        self.add_input("title",  PortType.STRING, "PCA 2D")
        self.add_output("image", PortType.IMAGE)

    def execute(self, inputs):
        null = {"image": None}
        try:
            X      = inputs.get("X")
            labels = inputs.get("labels")
            title  = inputs.get("title", "PCA 2D") or "PCA 2D"
            if X is None or X.shape[1] < 2:
                return null
            fig, ax = _dark_fig()
            if labels is not None:
                sc = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", alpha=0.7)
                fig.colorbar(sc, ax=ax)
            else:
                ax.scatter(X[:, 0], X[:, 1], alpha=0.7)
            ax.set_title(title)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            return {"image": _render_fig(fig)}
        except Exception:
            return null


# ── Loss Curve ────────────────────────────────────────────────────────────────

class VizLossNode(_VizBase):
    type_name = "viz_loss_curve"
    label = "Loss Curve"
    category = CATEGORY
    description = "Training and optional validation loss curves"

    def _setup_ports(self):
        self.add_input("train_losses", PortType.NDARRAY)
        self.add_input("val_losses",   PortType.NDARRAY)
        self.add_input("title",        PortType.STRING, "Loss Curve")
        self.add_output("image",       PortType.IMAGE)

    def execute(self, inputs):
        null = {"image": None}
        try:
            train = inputs.get("train_losses")
            val   = inputs.get("val_losses")
            title = inputs.get("title", "Loss Curve") or "Loss Curve"
            if train is None:
                return null
            fig, ax = _dark_fig()
            epochs = list(range(1, len(train) + 1))
            ax.plot(epochs, train, color="cyan",  label="train")
            if val is not None:
                vepochs = list(range(1, len(val) + 1))
                ax.plot(vepochs, val, color="orange", label="val")
                ax.legend()
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            return {"image": _render_fig(fig)}
        except Exception:
            return null


# ── Image Grid ────────────────────────────────────────────────────────────────

class VizImageGridNode(_VizBase):
    type_name = "viz_image_grid"
    label = "Image Grid"
    category = CATEGORY
    description = "Grid of image tensors (e.g. MNIST batch)"

    def _setup_ports(self):
        self.add_input("images", PortType.TENSOR)
        self.add_input("title",  PortType.STRING, "Image Grid")
        self.add_input("nrow",   PortType.INT,    8)
        self.add_output("image", PortType.IMAGE)

    def execute(self, inputs):
        null = {"image": None}
        try:
            import torch
            from torchvision.utils import make_grid
            images = inputs.get("images")
            title  = inputs.get("title", "Image Grid") or "Image Grid"
            nrow   = inputs.get("nrow",  8)
            if images is None:
                return null
            if not isinstance(images, torch.Tensor):
                return null
            grid = make_grid(images, nrow=int(nrow), normalize=True, value_range=(0, 1))
            # grid: (C, H, W) float
            arr = (grid.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            if arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
            fig, ax = _dark_fig(figsize=(8, 4))
            ax.imshow(arr)
            ax.axis("off")
            ax.set_title(title)
            return {"image": _render_fig(fig)}
        except Exception:
            return null


# Subcategory stamp
from core.node import BaseNode as _BN
for _n, _c in list(globals().items()):
    if isinstance(_c, type) and issubclass(_c, _BN) and _c is not _BN:
        _c.subcategory = ""
