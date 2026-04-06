"""Color Jitter Transform node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class ColorJitterTransformNode(BaseNode):
    type_name   = "pt_color_jitter_transform"
    label       = "Color Jitter"
    category    = "Datasets"
    subcategory = "Transforms"
    description = "Randomly change brightness, contrast, saturation, hue."

    def _setup_ports(self):
        self.add_input("brightness", PortType.FLOAT, default=0.2)
        self.add_input("contrast",   PortType.FLOAT, default=0.2)
        self.add_input("saturation", PortType.FLOAT, default=0.2)
        self.add_input("hue",        PortType.FLOAT, default=0.0)
        self.add_output("transform", PortType.TRANSFORM)

    def execute(self, inputs):
        try:
            from torchvision.transforms import ColorJitter
            return {"transform": ColorJitter(
                brightness=float(inputs.get("brightness") or 0),
                contrast=float(inputs.get("contrast") or 0),
                saturation=float(inputs.get("saturation") or 0),
                hue=float(inputs.get("hue") or 0),
            )}
        except Exception:
            return {"transform": None}

    def export(self, iv, ov):
        b = self._val(iv, 'brightness'); c = self._val(iv, 'contrast')
        s = self._val(iv, 'saturation'); h = self._val(iv, 'hue')
        return ["from torchvision.transforms import ColorJitter"], [
            f"{ov['transform']} = ColorJitter(brightness={b}, contrast={c}, saturation={s}, hue={h})"
        ]
