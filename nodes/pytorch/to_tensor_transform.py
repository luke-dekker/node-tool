"""To Tensor Transform node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class ToTensorTransformNode(BaseNode):
    type_name   = "pt_to_tensor_transform"
    label       = "To Tensor"
    category    = "Datasets"
    subcategory = "Transforms"
    description = "Convert PIL image or numpy array to a FloatTensor (torchvision.transforms.ToTensor)."

    def _setup_ports(self):
        self.add_output("transform", PortType.TRANSFORM)

    def execute(self, inputs):
        try:
            from torchvision.transforms import ToTensor
            return {"transform": ToTensor()}
        except Exception:
            return {"transform": None}

    def export(self, iv, ov):
        return ["from torchvision.transforms import ToTensor"], [f"{ov['transform']} = ToTensor()"]
