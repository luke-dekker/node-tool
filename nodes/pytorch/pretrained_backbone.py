"""Consolidated pretrained backbone node — replaces resnet18, resnet50,
mobilenet_v3, efficientnet_b0."""
from __future__ import annotations
from core.node import BaseNode, PortType


def _replace_fc(model, nc):
    """Replace final FC layer (ResNet family)."""
    import torch.nn as nn
    model.fc = nn.Linear(model.fc.in_features, nc)


def _replace_classifier_last(model, nc):
    """Replace last layer of classifier sequential (MobileNet/EfficientNet)."""
    import torch.nn as nn
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, nc)


# (weights_attr, constructor_fn_name, head_replace_fn, display_name)
_ARCH_CFG = {
    "resnet18": ("ResNet18_Weights", "resnet18", _replace_fc, "ResNet-18"),
    "resnet50": ("ResNet50_Weights", "resnet50", _replace_fc, "ResNet-50"),
    "mobilenet_v3_small": ("MobileNet_V3_Small_Weights", "mobilenet_v3_small",
                           _replace_classifier_last, "MobileNetV3-Small"),
    "efficientnet_b0": ("EfficientNet_B0_Weights", "efficientnet_b0",
                        _replace_classifier_last, "EfficientNet-B0"),
}


class PretrainedBackboneNode(BaseNode):
    type_name   = "pt_pretrained_backbone"
    label       = "Pretrained Backbone"
    category    = "Models"
    subcategory = "Pretrained"
    description = (
        "Pretrained torchvision backbone. Select architecture: "
        "resnet18, resnet50, mobilenet_v3_small, efficientnet_b0. "
        "num_classes replaces the final head (0 keeps original)."
    )

    def __init__(self):
        self._layer = None
        self._layer_cfg: tuple | None = None
        super().__init__()

    def _setup_ports(self):
        self.add_input("architecture", PortType.STRING, default="resnet18")
        self.add_input("pretrained",   PortType.BOOL,   default=True)
        self.add_input("num_classes",  PortType.INT,    default=10)
        self.add_output("model", PortType.MODULE)
        self.add_output("info",  PortType.STRING)

    def execute(self, inputs):
        try:
            import torchvision.models as M
            arch = (inputs.get("architecture") or "resnet18").strip().lower()
            pretrained = bool(inputs.get("pretrained", True))
            nc = int(inputs.get("num_classes") or 10)

            if arch not in _ARCH_CFG:
                return {"model": None, "info": f"Unknown architecture: {arch}"}

            cfg = (arch, pretrained, nc)
            if self._layer is None or self._layer_cfg != cfg:
                weights_attr, ctor_name, head_fn, display = _ARCH_CFG[arch]
                weights = getattr(M, weights_attr).DEFAULT if pretrained else None
                model = getattr(M, ctor_name)(weights=weights)
                if nc > 0:
                    head_fn(model, nc)
                self._layer = model
                self._layer_cfg = cfg

            params = sum(p.numel() for p in self._layer.parameters())
            _, _, _, display = _ARCH_CFG[arch]
            return {"model": self._layer,
                    "info": f"{display}  params={params:,}  out={nc}"}
        except Exception:
            import traceback
            return {"model": None, "info": traceback.format_exc().split("\n")[-2]}

    def export(self, iv, ov):
        mv = ov.get("model", "_backbone")
        info_var = ov.get("info", "_backbone_info")
        arch = (self.inputs["architecture"].default_value or "resnet18").strip()
        pretrained = self.inputs["pretrained"].default_value
        nc = self.inputs["num_classes"].default_value

        if arch not in _ARCH_CFG:
            return [], [f"# pretrained_backbone: unknown architecture '{arch}'"]

        weights_attr, ctor_name, head_fn, display = _ARCH_CFG[arch]
        lines = [
            f"_weights = M.{weights_attr}.DEFAULT if {bool(pretrained)} else None",
            f"{mv} = M.{ctor_name}(weights=_weights)",
        ]
        if nc and int(nc) > 0:
            if head_fn is _replace_fc:
                lines.append(f"{mv}.fc = nn.Linear({mv}.fc.in_features, {int(nc)})")
            else:
                lines.append(
                    f"{mv}.classifier[-1] = nn.Linear({mv}.classifier[-1].in_features, {int(nc)})"
                )
        lines.append(
            f"{info_var} = f'{display}  params={{sum(p.numel() for p in {mv}.parameters()):,}}  out={int(nc) if nc else 1000}'"
        )
        return ["import torch", "import torch.nn as nn", "import torchvision.models as M"], lines
