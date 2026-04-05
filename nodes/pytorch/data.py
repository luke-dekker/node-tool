"""PyTorch tensor creation, dataset, and dataloader nodes."""
from __future__ import annotations
from core.node import BaseNode
from core.node import PortType

CATEGORY = "Analyze"


class RandTensorNode(BaseNode):
    type_name = "pt_rand_tensor"
    label = "Rand Tensor"
    category = CATEGORY
    description = "Create a random tensor with the given shape"

    def _setup_ports(self):
        self.add_input("shape", PortType.STRING, default="1,64")
        self.add_input("requires_grad", PortType.BOOL, default=False)
        self.add_output("tensor", PortType.TENSOR)

    def execute(self, inputs):
        try:
            import torch
            shape_str = inputs.get("shape", "1,64") or "1,64"
            shape = tuple(int(x.strip()) for x in shape_str.split(",") if x.strip())
            requires_grad = bool(inputs.get("requires_grad", False))
            return {"tensor": torch.randn(shape, requires_grad=requires_grad)}
        except Exception:
            return {"tensor": None}


class ZerosTensorNode(BaseNode):
    type_name = "pt_zeros_tensor"
    label = "Zeros Tensor"
    category = CATEGORY
    description = "Create a zeros tensor with the given shape"

    def _setup_ports(self):
        self.add_input("shape", PortType.STRING, default="1,64")
        self.add_output("tensor", PortType.TENSOR)

    def execute(self, inputs):
        try:
            import torch
            shape_str = inputs.get("shape", "1,64") or "1,64"
            shape = tuple(int(x.strip()) for x in shape_str.split(",") if x.strip())
            return {"tensor": torch.zeros(shape)}
        except Exception:
            return {"tensor": None}


class OnesTensorNode(BaseNode):
    type_name = "pt_ones_tensor"
    label = "Ones Tensor"
    category = CATEGORY
    description = "Create a ones tensor with the given shape"

    def _setup_ports(self):
        self.add_input("shape", PortType.STRING, default="1,64")
        self.add_output("tensor", PortType.TENSOR)

    def execute(self, inputs):
        try:
            import torch
            shape_str = inputs.get("shape", "1,64") or "1,64"
            shape = tuple(int(x.strip()) for x in shape_str.split(",") if x.strip())
            return {"tensor": torch.ones(shape)}
        except Exception:
            return {"tensor": None}


class TensorFromListNode(BaseNode):
    type_name = "pt_tensor_from_list"
    label = "Tensor From List"
    category = CATEGORY
    description = "Create a tensor from comma-separated float values"

    def _setup_ports(self):
        self.add_input("values", PortType.STRING, default="0,1,2,3")
        self.add_output("tensor", PortType.TENSOR)

    def execute(self, inputs):
        try:
            import torch
            values_str = inputs.get("values", "0,1,2,3") or "0,1,2,3"
            vals = [float(x.strip()) for x in values_str.split(",") if x.strip()]
            return {"tensor": torch.tensor(vals)}
        except Exception:
            return {"tensor": None}


class TensorShapeNode(BaseNode):
    type_name = "pt_tensor_shape"
    label = "Tensor Shape"
    category = CATEGORY
    description = "Return tensor shape as string"

    def _setup_ports(self):
        self.add_input("tensor", PortType.TENSOR, default=None)
        self.add_output("shape", PortType.STRING)

    def execute(self, inputs):
        try:
            tensor = inputs.get("tensor")
            if tensor is None:
                return {"shape": "None"}
            return {"shape": str(list(tensor.shape))}
        except Exception:
            return {"shape": "error"}


class TensorInfoNode(BaseNode):
    type_name = "pt_tensor_info"
    label = "Tensor Info"
    category = CATEGORY
    description = "Return detailed tensor info as string"

    def _setup_ports(self):
        self.add_input("tensor", PortType.TENSOR, default=None)
        self.add_output("info", PortType.STRING)

    def execute(self, inputs):
        try:
            tensor = inputs.get("tensor")
            if tensor is None:
                return {"info": "None"}
            info = f"shape={list(tensor.shape)} dtype={tensor.dtype} min={tensor.min():.4f} max={tensor.max():.4f}"
            return {"info": info}
        except Exception:
            return {"info": "error"}


class TensorAddNode(BaseNode):
    type_name = "pt_tensor_add"
    label = "Tensor Add"
    category = CATEGORY
    description = "Element-wise addition of two tensors"

    def _setup_ports(self):
        self.add_input("a", PortType.TENSOR, default=None)
        self.add_input("b", PortType.TENSOR, default=None)
        self.add_output("result", PortType.TENSOR)

    def execute(self, inputs):
        try:
            a = inputs.get("a")
            b = inputs.get("b")
            if a is None or b is None:
                return {"result": None}
            return {"result": a + b}
        except Exception:
            return {"result": None}


class TensorMulNode(BaseNode):
    type_name = "pt_tensor_mul"
    label = "Tensor Mul"
    category = CATEGORY
    description = "Element-wise multiplication of two tensors"

    def _setup_ports(self):
        self.add_input("a", PortType.TENSOR, default=None)
        self.add_input("b", PortType.TENSOR, default=None)
        self.add_output("result", PortType.TENSOR)

    def execute(self, inputs):
        try:
            a = inputs.get("a")
            b = inputs.get("b")
            if a is None or b is None:
                return {"result": None}
            return {"result": a * b}
        except Exception:
            return {"result": None}


class ArgmaxNode(BaseNode):
    type_name = "pt_argmax"
    label = "Argmax"
    category = CATEGORY
    description = "torch.argmax(tensor, dim)"

    def _setup_ports(self):
        self.add_input("tensor", PortType.TENSOR, default=None)
        self.add_input("dim", PortType.INT, default=-1)
        self.add_output("result", PortType.TENSOR)

    def execute(self, inputs):
        try:
            import torch
            tensor = inputs.get("tensor")
            if tensor is None:
                return {"result": None}
            dim = int(inputs.get("dim", -1))
            return {"result": torch.argmax(tensor, dim=dim)}
        except Exception:
            return {"result": None}


class SoftmaxOpNode(BaseNode):
    type_name = "pt_softmax_op"
    label = "Softmax Op"
    category = CATEGORY
    description = "torch.softmax(tensor, dim) — functional op"

    def _setup_ports(self):
        self.add_input("tensor", PortType.TENSOR, default=None)
        self.add_input("dim", PortType.INT, default=-1)
        self.add_output("result", PortType.TENSOR)

    def execute(self, inputs):
        try:
            import torch
            tensor = inputs.get("tensor")
            if tensor is None:
                return {"result": None}
            dim = int(inputs.get("dim", -1))
            return {"result": torch.softmax(tensor, dim=dim)}
        except Exception:
            return {"result": None}


class PrintTensorNode(BaseNode):
    type_name = "pt_print_tensor"
    label = "Print Tensor"
    category = CATEGORY
    description = "Log tensor shape to terminal and pass through"

    def _setup_ports(self):
        self.add_input("tensor", PortType.TENSOR, default=None)
        self.add_input("label", PortType.STRING, default="")
        self.add_output("passthrough", PortType.TENSOR)
        self.add_output("__terminal__", PortType.STRING)

    def execute(self, inputs):
        try:
            tensor = inputs.get("tensor")
            label = inputs.get("label", "") or ""
            if tensor is None:
                return {"passthrough": None, "__terminal__": f"{label}: None"}
            shape = list(tensor.shape)
            n = tensor.numel()
            if n <= 128:
                vals = tensor.flatten().tolist()
                vals_str = ", ".join(
                    str(int(v)) if float(v) == int(v) else f"{v:.4g}" for v in vals
                )
                msg = f"{label}: shape={shape}  [{vals_str}]"
            else:
                msg = f"{label}: shape={shape} dtype={tensor.dtype}"
            return {"passthrough": tensor, "__terminal__": msg}
        except Exception:
            return {"passthrough": None, "__terminal__": "error"}


class MNISTDatasetNode(BaseNode):
    type_name = "pt_mnist"
    label = "MNIST Dataset"
    category    = "Datasets"
    subcategory = "Sources"
    description = "torchvision MNIST dataset wrapped in DataLoader"

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
            dataset = datasets.MNIST(
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
            print("[WARN] torchvision not available — MNISTDatasetNode returning None")
            return {"dataloader": None}
        except Exception:
            return {"dataloader": None}


class CIFAR10DatasetNode(BaseNode):
    type_name = "pt_cifar10"
    label = "CIFAR10 Dataset"
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


class DataLoaderInfoNode(BaseNode):
    type_name = "pt_dataloader_info"
    label = "DataLoader Info"
    category    = "Datasets"
    subcategory = "Loader"
    description = "Return DataLoader info string"

    def _setup_ports(self):
        self.add_input("dataloader", PortType.DATALOADER, default=None)
        self.add_output("info", PortType.STRING)

    def execute(self, inputs):
        try:
            dl = inputs.get("dataloader")
            if dl is None:
                return {"info": "None"}
            return {"info": f"batches={len(dl)} batch_size={dl.batch_size}"}
        except Exception:
            return {"info": "error"}


class SampleBatchNode(BaseNode):
    type_name   = "pt_sample_batch"
    label       = "Sample Batch"
    category    = "Datasets"
    subcategory = "Loader"
    description = "Peek at the first batch of a DataLoader and output x (and optionally y) as tensors"

    def _setup_ports(self):
        self.add_input("dataloader", PortType.DATALOADER, default=None)
        self.add_output("x", PortType.TENSOR)
        self.add_output("y", PortType.TENSOR)

    def execute(self, inputs):
        try:
            dl = inputs.get("dataloader")
            if dl is None:
                return {"x": None, "y": None}
            batch = next(iter(dl))
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            y = batch[1] if isinstance(batch, (list, tuple)) and len(batch) > 1 else None
            return {"x": x, "y": y}
        except Exception:
            return {"x": None, "y": None}


# Subcategory stamp
_SC = "Tensors"
from core.node import BaseNode as _BN
for _n, _c in list(globals().items()):
    if isinstance(_c, type) and issubclass(_c, _BN) and _c is not _BN:
        if not hasattr(_c, "subcategory") or _c.subcategory == "":
            _c.subcategory = _SC
