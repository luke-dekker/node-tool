"""Model Info (backbones) node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class ModelInfoNode(BaseNode):
    type_name   = "pt_model_info"
    label       = "Model Info"
    category    = "Models"
    subcategory = "Pretrained"
    description = "Print parameter count, trainable params, and layer summary for any nn.Module."

    def _setup_ports(self):
        self.add_input("model", PortType.MODULE, default=None)
        self.add_output("model", PortType.MODULE)
        self.add_output("info",  PortType.STRING)

    def execute(self, inputs):
        try:
            model = inputs.get("model")
            if model is None:
                return {"model": None, "info": "No model"}
            total     = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            layers    = sum(1 for _ in model.modules()) - 1
            info = (f"{model.__class__.__name__}  "
                    f"total={total:,}  trainable={trainable:,}  layers={layers}")
            return {"model": model, "info": info}
        except Exception:
            return {"model": None, "info": "error"}
