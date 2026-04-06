"""Training Curve visualization node."""
from __future__ import annotations
from typing import Any
import numpy as np
from core.node import BaseNode, PortType


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

    def export(self, iv, ov):
        return [], [f"# [{self.label}]: visualization nodes render inline - skipped in export"]
