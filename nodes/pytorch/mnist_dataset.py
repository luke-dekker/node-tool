"""MNIST Dataset node — outputs dataloader + tensor preview for direct model wiring.

The new architecture: dataset nodes output tensors directly so the model chain
wires from the dataset node — no separate BatchInput adapter. The `dataloader`
port is auto-discovered by the Training Panel for batch iteration.

    MNISTDataset ─── x ──→ model chain
                 └── label ──→ loss target
                 └── dataloader  (panel discovers, no wire needed)
"""
from __future__ import annotations
from core.node import BaseNode, PortType


class MNISTDatasetNode(BaseNode):
    type_name   = "pt_mnist"
    label       = "MNIST Dataset"
    category    = "Datasets"
    subcategory = "Sources"
    description = (
        "MNIST digit dataset. Outputs a DataLoader (auto-discovered by the "
        "Training Panel) plus x/label tensor preview for direct model wiring. "
        "Wire x → your model chain, label → LossCompute target."
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
        # Tensor preview outputs — live preview samples one batch; during
        # training, GraphAsModule overrides x/label with the current batch
        self.add_output("x",          PortType.TENSOR,
                        description="One batch of images for preview (B, 1, 28, 28)")
        self.add_output("label",      PortType.TENSOR,
                        description="Corresponding labels (B,)")
        # DataLoader for panel auto-discovery — no wire needed from the user
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
                dataset = datasets.MNIST(
                    root="./data", train=train, download=dl_flag,
                    transform=transforms.ToTensor()
                )
                self._cached_loader = DataLoader(
                    dataset, batch_size=bs, shuffle=shuffle
                )
                self._cached_cfg = cfg

            # Sample one batch for tensor preview (live preview / Run Graph)
            try:
                batch = next(iter(self._cached_loader))
                x, label = batch[0], batch[1]
            except Exception:
                x, label = None, None

            return {"dataloader": self._cached_loader, "x": x, "label": label}
        except ImportError:
            print("[WARN] torchvision not available — MNISTDatasetNode returning None")
            return {"dataloader": None, "x": None, "label": None}
        except Exception:
            return {"dataloader": None, "x": None, "label": None}

    def export(self, iv, ov):
        bs      = self._val(iv, 'batch_size')
        train   = self._val(iv, 'train')
        dl_flag = self._val(iv, 'download')
        shuffle = self._val(iv, 'shuffle')
        x_var   = ov.get("x", "_mnist_x")
        y_var   = ov.get("label", "_mnist_label")
        dl_var  = ov.get("dataloader", "_mnist_dl")
        return [
            "import torch",
            "from torchvision import datasets, transforms",
            "from torch.utils.data import DataLoader",
        ], [
            f"_mnist = datasets.MNIST(root='./data', train={train}, download={dl_flag}, "
            f"transform=transforms.ToTensor())",
            f"{dl_var} = DataLoader(_mnist, batch_size={bs}, shuffle={shuffle})",
            f"_batch = next(iter({dl_var}))",
            f"{x_var}, {y_var} = _batch[0], _batch[1]",
        ]
