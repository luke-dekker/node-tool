"""Image Folder Dataset node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class ImageFolderDatasetNode(BaseNode):
    type_name   = "pt_image_folder_dataset"
    label       = "Image Folder Dataset"
    category    = "Datasets"
    subcategory = "Sources"
    description = "Load images from root/class_name/image.jpg folder structure (torchvision ImageFolder)."

    def _setup_ports(self):
        self.add_input("root_path",  PortType.STRING,    default="./data/images")
        self.add_input("transform",  PortType.TRANSFORM, default=None)
        self.add_output("dataset",     PortType.DATASET)
        self.add_output("class_names", PortType.STRING)
        self.add_output("info",        PortType.STRING)

    def execute(self, inputs):
        try:
            from torchvision.datasets import ImageFolder
            root = str(inputs.get("root_path") or "./data/images")
            transform = inputs.get("transform")
            dataset = ImageFolder(root=root, transform=transform)
            names = ", ".join(dataset.classes)
            info = f"ImageFolder: {len(dataset)} images, {len(dataset.classes)} classes"
            return {"dataset": dataset, "class_names": names, "info": info}
        except Exception:
            import traceback
            return {"dataset": None, "class_names": "", "info": traceback.format_exc().split("\n")[-2]}

    def export(self, iv, ov):
        root = self._val(iv, 'root_path'); tf = iv.get('transform')
        tf_arg = f", transform={tf}" if tf else ""
        dsv = ov['dataset']; infov = ov['info']; cnv = ov['class_names']
        return ["from torchvision.datasets import ImageFolder"], [
            f"{dsv} = ImageFolder(root={root}{tf_arg})",
            f"{cnv} = ', '.join({dsv}.classes)",
            f"{infov} = f'ImageFolder: {{len({dsv})}} images'",
        ]
