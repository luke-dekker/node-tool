"""Weight Histogram visualization node."""
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

    def export(self, iv, ov):
        return [], [f"# [{self.label}]: visualization nodes render inline - skipped in export"]
