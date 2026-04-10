"""CIFAR10 Dataset node — outputs dataloader + tensor preview."""
from __future__ import annotations
from core.node import BaseNode, PortType


class CIFAR10DatasetNode(BaseNode):
    type_name   = "pt_cifar10"
    label       = "CIFAR10 Dataset"
    category    = "Datasets"
    subcategory = "Sources"
    description = (
        "CIFAR10 dataset. Outputs a DataLoader (auto-discovered by the "
        "Training Panel) plus x/label tensor preview for direct model wiring."
    )

    def __init__(self):
        self._cached_loader = None
        self._cached_cfg: tuple = ()
        super().__init__()

    def _setup_ports(self):
        self.add_input("task_id",    PortType.STRING, default="default",
                       description="Pairs this dataset with a Train Output that has the same task_name")
        self.add_input("batch_size", PortType.INT,  default=32)
        self.add_input("train",      PortType.BOOL, default=True)
        self.add_input("download",   PortType.BOOL, default=True)
        self.add_input("shuffle",    PortType.BOOL, default=True)
        self.add_output("x",          PortType.TENSOR,
                        description="One batch of images for preview (B, 3, 32, 32)")
        self.add_output("label",      PortType.TENSOR,
                        description="Corresponding labels (B,)")
        self.add_output("dataloader", PortType.DATALOADER)

    def execute(self, inputs):
        try:
            from torchvision import datasets, transforms
            from torch.utils.data import DataLoader
            bs      = int(inputs.get("batch_size", 32))
            train   = bool(inputs.get("train", True))
            dl_flag = bool(inputs.get("download", True))
            shuffle = bool(inputs.get("shuffle", True))
            cfg = (bs, train, dl_flag, shuffle)
            if self._cached_loader is None or self._cached_cfg != cfg:
                dataset = datasets.CIFAR10(
                    root="./data", train=train, download=dl_flag,
                    transform=transforms.ToTensor()
                )
                self._cached_loader = DataLoader(dataset, batch_size=bs, shuffle=shuffle)
                self._cached_cfg = cfg
            try:
                batch = next(iter(self._cached_loader))
                x, label = batch[0], batch[1]
            except Exception:
                x, label = None, None
            return {"dataloader": self._cached_loader, "x": x, "label": label}
        except ImportError:
            return {"dataloader": None, "x": None, "label": None}
        except Exception:
            return {"dataloader": None, "x": None, "label": None}

    def export(self, iv, ov):
        bs = self._val(iv, 'batch_size'); train = self._val(iv, 'train')
        dl_flag = self._val(iv, 'download'); shuffle = self._val(iv, 'shuffle')
        x_var = ov.get("x", "_cifar_x"); y_var = ov.get("label", "_cifar_label")
        dl_var = ov.get("dataloader", "_cifar_dl")
        return [
            "import torch", "from torchvision import datasets, transforms",
            "from torch.utils.data import DataLoader",
        ], [
            f"_cifar = datasets.CIFAR10(root='./data', train={train}, download={dl_flag}, "
            f"transform=transforms.ToTensor())",
            f"{dl_var} = DataLoader(_cifar, batch_size={bs}, shuffle={shuffle})",
            f"_batch = next(iter({dl_var}))",
            f"{x_var}, {y_var} = _batch[0], _batch[1]",
        ]
