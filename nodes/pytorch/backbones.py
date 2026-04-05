"""Pretrained backbone nodes — load torchvision models with optional pretrained weights."""
from __future__ import annotations
from core.node import BaseNode
from core.node import PortType

CATEGORY = "Models"


class ResNet18Node(BaseNode):
    type_name = "pt_resnet18"
    label = "ResNet-18"
    category = CATEGORY
    description = "torchvision ResNet-18. pretrained=True loads ImageNet weights. num_classes replaces the final FC layer (set 0 to keep original 1000-class head)."

    def _setup_ports(self):
        self.add_input("pretrained",  PortType.BOOL, default=True)
        self.add_input("num_classes", PortType.INT,  default=10)
        self.add_output("model", PortType.MODULE)
        self.add_output("info",  PortType.STRING)

    def execute(self, inputs):
        try:
            import torchvision.models as M
            import torch.nn as nn
            weights = M.ResNet18_Weights.DEFAULT if bool(inputs.get("pretrained", True)) else None
            model = M.resnet18(weights=weights)
            nc = int(inputs.get("num_classes") or 10)
            if nc > 0:
                model.fc = nn.Linear(model.fc.in_features, nc)
            params = sum(p.numel() for p in model.parameters())
            return {"model": model, "info": f"ResNet-18  params={params:,}  out={nc}"}
        except Exception:
            import traceback
            return {"model": None, "info": traceback.format_exc().split("\n")[-2]}


class ResNet50Node(BaseNode):
    type_name = "pt_resnet50"
    label = "ResNet-50"
    category = CATEGORY
    description = "torchvision ResNet-50. pretrained=True loads ImageNet weights. num_classes replaces the final FC layer."

    def _setup_ports(self):
        self.add_input("pretrained",  PortType.BOOL, default=True)
        self.add_input("num_classes", PortType.INT,  default=10)
        self.add_output("model", PortType.MODULE)
        self.add_output("info",  PortType.STRING)

    def execute(self, inputs):
        try:
            import torchvision.models as M
            import torch.nn as nn
            weights = M.ResNet50_Weights.DEFAULT if bool(inputs.get("pretrained", True)) else None
            model = M.resnet50(weights=weights)
            nc = int(inputs.get("num_classes") or 10)
            if nc > 0:
                model.fc = nn.Linear(model.fc.in_features, nc)
            params = sum(p.numel() for p in model.parameters())
            return {"model": model, "info": f"ResNet-50  params={params:,}  out={nc}"}
        except Exception:
            import traceback
            return {"model": None, "info": traceback.format_exc().split("\n")[-2]}


class MobileNetV3Node(BaseNode):
    type_name = "pt_mobilenet_v3"
    label = "MobileNet-V3 Small"
    category = CATEGORY
    description = "torchvision MobileNetV3-Small — lightweight backbone for edge/robotics deployment."

    def _setup_ports(self):
        self.add_input("pretrained",  PortType.BOOL, default=True)
        self.add_input("num_classes", PortType.INT,  default=10)
        self.add_output("model", PortType.MODULE)
        self.add_output("info",  PortType.STRING)

    def execute(self, inputs):
        try:
            import torchvision.models as M
            import torch.nn as nn
            weights = M.MobileNet_V3_Small_Weights.DEFAULT if bool(inputs.get("pretrained", True)) else None
            model = M.mobilenet_v3_small(weights=weights)
            nc = int(inputs.get("num_classes") or 10)
            if nc > 0:
                model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, nc)
            params = sum(p.numel() for p in model.parameters())
            return {"model": model, "info": f"MobileNetV3-Small  params={params:,}  out={nc}"}
        except Exception:
            import traceback
            return {"model": None, "info": traceback.format_exc().split("\n")[-2]}


class EfficientNetB0Node(BaseNode):
    type_name = "pt_efficientnet_b0"
    label = "EfficientNet-B0"
    category = CATEGORY
    description = "torchvision EfficientNet-B0 — strong accuracy/efficiency tradeoff."

    def _setup_ports(self):
        self.add_input("pretrained",  PortType.BOOL, default=True)
        self.add_input("num_classes", PortType.INT,  default=10)
        self.add_output("model", PortType.MODULE)
        self.add_output("info",  PortType.STRING)

    def execute(self, inputs):
        try:
            import torchvision.models as M
            import torch.nn as nn
            weights = M.EfficientNet_B0_Weights.DEFAULT if bool(inputs.get("pretrained", True)) else None
            model = M.efficientnet_b0(weights=weights)
            nc = int(inputs.get("num_classes") or 10)
            if nc > 0:
                model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, nc)
            params = sum(p.numel() for p in model.parameters())
            return {"model": model, "info": f"EfficientNet-B0  params={params:,}  out={nc}"}
        except Exception:
            import traceback
            return {"model": None, "info": traceback.format_exc().split("\n")[-2]}


class FreezeBackboneNode(BaseNode):
    type_name = "pt_freeze_backbone"
    label = "Freeze Layers"
    category = CATEGORY
    description = "Freeze all parameters in a model (requires_grad=False). Pass through to unfreeze_from node or directly to optimizer."

    def _setup_ports(self):
        self.add_input("model",        PortType.MODULE, default=None)
        self.add_input("freeze_all",   PortType.BOOL,   default=True)
        self.add_output("model", PortType.MODULE)
        self.add_output("info",  PortType.STRING)

    def execute(self, inputs):
        try:
            model = inputs.get("model")
            if model is None:
                return {"model": None, "info": "No model"}
            freeze = bool(inputs.get("freeze_all", True))
            for p in model.parameters():
                p.requires_grad = not freeze
            frozen = sum(1 for p in model.parameters() if not p.requires_grad)
            total  = sum(1 for p in model.parameters())
            return {"model": model, "info": f"frozen={frozen}/{total} param groups"}
        except Exception:
            return {"model": None, "info": "error"}


class ModelInfoNode(BaseNode):
    type_name = "pt_model_info"
    label = "Model Info"
    category = CATEGORY
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


class FreezeNamedLayersNode(BaseNode):
    type_name = "pt_freeze_named_layers"
    label = "Freeze Named Layers"
    category = CATEGORY
    description = "Freeze specific layers by name prefix. Enter comma-separated prefixes, e.g. 'encoder,layer1'. Unmatched layers stay unchanged."

    def _setup_ports(self):
        self.add_input("model",   PortType.MODULE, default=None)
        self.add_input("names",   PortType.STRING, default="encoder")
        self.add_input("freeze",  PortType.BOOL,   default=True)
        self.add_output("model",  PortType.MODULE)
        self.add_output("info",   PortType.STRING)

    def execute(self, inputs):
        try:
            model = inputs.get("model")
            if model is None:
                return {"model": None, "info": "No model"}
            names = [n.strip() for n in str(inputs.get("names") or "").split(",") if n.strip()]
            freeze = bool(inputs.get("freeze", True))
            count = 0
            for name, param in model.named_parameters():
                if any(name.startswith(prefix) for prefix in names):
                    param.requires_grad = not freeze
                    count += 1
            action = "Frozen" if freeze else "Unfrozen"
            return {"model": model, "info": f"{action} {count} params matching {names}"}
        except Exception:
            return {"model": None, "info": "error"}


# Subcategory stamp
_SC = "Pretrained"
from core.node import BaseNode as _BN
for _n, _c in list(globals().items()):
    if isinstance(_c, type) and issubclass(_c, _BN) and _c is not _BN:
        _c.subcategory = _SC
