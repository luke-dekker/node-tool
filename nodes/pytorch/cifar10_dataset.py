"""CIFAR10 Dataset node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class CIFAR10DatasetNode(BaseNode):
    type_name   = "pt_cifar10"
    label       = "CIFAR10 Dataset"
    category    = "Datasets"
    subcategory = "Sources"
    description = "torchvision CIFAR10 dataset wrapped in DataLoader"

    def _setup_ports(self):
        self.add_input("batch_size", PortType.INT, default=32)
        self.add_input("train", PortType.BOOL, default=True)
        self.add_input("download", PortType.BOOL, default=True)
        self.add_input("shuffle", PortType.BOOL, default=True)
        self.add_output("dataloader", PortType.DATALOADER)

    def execute(self, inputs):
        try:
            from torchvision import datasets, transforms
            from torch.utils.data import DataLoader
            dataset = datasets.CIFAR10(
                root="./data",
                train=bool(inputs.get("train", True)),
                download=bool(inputs.get("download", True)),
                transform=transforms.ToTensor()
            )
            loader = DataLoader(
                dataset,
                batch_size=int(inputs.get("batch_size", 32)),
                shuffle=bool(inputs.get("shuffle", True))
            )
            return {"dataloader": loader}
        except ImportError:
            print("[WARN] torchvision not available — CIFAR10DatasetNode returning None")
            return {"dataloader": None}
        except Exception:
            return {"dataloader": None}

    def export(self, iv, ov):
        bs = self._val(iv, 'batch_size'); train = self._val(iv, 'train')
        dl = self._val(iv, 'download'); shuffle = self._val(iv, 'shuffle')
        return [
            "import torch",
            "from torchvision import datasets, transforms",
            "from torch.utils.data import DataLoader",
        ], [
            f"_cifar = datasets.CIFAR10(root='./data', train={train}, download={dl}, transform=transforms.ToTensor())",
            f"{ov['dataloader']} = DataLoader(_cifar, batch_size={bs}, shuffle={shuffle})",
        ]
