"""PyTorch layer nodes.

Each node IS a layer - it owns a persistent nn.Module (self._layer).

  tensor_in  ->  self._layer(tensor_in)  ->  tensor_out

The graph topology defines the model. Training Config walks the tensor_in
connections backwards to collect layers in order and assemble the model.
Changing structural parameters (sizes, kernel) recreates the layer (resets
weights). Freeze can be toggled without resetting.

get_layers() returns the ordered list of nn.Modules for model assembly,
including any activation configured on the node.

Activation options (case-insensitive):
  none, relu, leakyrelu, sigmoid, tanh, gelu, elu, silu (swish), softmax
"""
from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
from core.node import BaseNode
from core.node import PortType

CATEGORY = "Layers"

_ACT_HELP = "none | relu | leakyrelu | sigmoid | tanh | gelu | elu | silu | softmax"


def _make_activation(name: str) -> nn.Module | None:
    key = (name or "").strip().lower().replace(" ", "").replace("_", "")
    return {
        "relu":      nn.ReLU(),
        "leakyrelu": nn.LeakyReLU(0.01),
        "sigmoid":   nn.Sigmoid(),
        "tanh":      nn.Tanh(),
        "gelu":      nn.GELU(),
        "elu":       nn.ELU(),
        "silu":      nn.SiLU(),
        "swish":     nn.SiLU(),
        "softmax":   nn.Softmax(dim=1),
    }.get(key)


def _forward(layer: nn.Module, act: nn.Module | None, tensor: torch.Tensor | None):
    """Run tensor through layer (+ optional activation). Returns None if no input."""
    if tensor is None:
        return None
    try:
        with torch.no_grad():
            out = layer(tensor)
            if act is not None:
                out = act(out)
        return out
    except Exception:
        return None


# -- Dense / MLP ---------------------------------------------------------------

class FlattenNode(BaseNode):
    type_name   = "pt_flatten"
    label       = "Flatten"
    category    = CATEGORY
    subcategory = "Dense"
    description = "nn.Flatten - collapses all dims after the batch dim."

    def __init__(self):
        self._layer: nn.Flatten | None = None
        self._layer_cfg: tuple | None = None
        super().__init__()

    def _get_layer(self, start_dim: int) -> nn.Flatten:
        cfg = (start_dim,)
        if self._layer is None or self._layer_cfg != cfg:
            self._layer = nn.Flatten(start_dim=start_dim)
            self._layer_cfg = cfg
        return self._layer

    def get_layers(self) -> list[nn.Module]:
        return [self._layer] if self._layer is not None else []

    def _setup_ports(self) -> None:
        self.add_input("tensor_in",   PortType.TENSOR, default=None)
        self.add_input("start_dim",   PortType.INT,    default=1)
        self.add_output("tensor_out", PortType.TENSOR)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        layer = self._get_layer(int(inputs.get("start_dim") or 1))
        return {"tensor_out": _forward(layer, None, inputs.get("tensor_in"))}


class LinearNode(BaseNode):
    type_name   = "pt_linear"
    label       = "Linear"
    category    = CATEGORY
    subcategory = "Dense"
    description = "nn.Linear(in, out) with optional activation. Weights persist across graph runs."

    def __init__(self):
        self._layer: nn.Linear | None = None
        self._layer_cfg: tuple | None = None
        self._act_name: str = ""
        super().__init__()

    def _get_layer(self, in_f: int, out_f: int, bias: bool) -> nn.Linear:
        cfg = (in_f, out_f, bias)
        if self._layer is None or self._layer_cfg != cfg:
            self._layer = nn.Linear(in_f, out_f, bias=bias)
            self._layer_cfg = cfg
        return self._layer

    def get_layers(self) -> list[nn.Module]:
        if self._layer is None:
            return []
        modules: list[nn.Module] = [self._layer]
        act = _make_activation(self._act_name)
        if act is not None:
            modules.append(act)
        return modules

    def _setup_ports(self) -> None:
        self.add_input("tensor_in",    PortType.TENSOR, default=None)
        self.add_input("in_features",  PortType.INT,    default=64)
        self.add_input("out_features", PortType.INT,    default=32)
        self.add_input("bias",         PortType.BOOL,   default=True)
        self.add_input("activation",   PortType.STRING, default="none",
                       description=_ACT_HELP)
        self.add_input("freeze",       PortType.BOOL,   default=False)
        self.add_output("tensor_out",  PortType.TENSOR)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        layer = self._get_layer(
            int(inputs.get("in_features")  or 64),
            int(inputs.get("out_features") or 32),
            bool(inputs.get("bias", True)),
        )
        self._act_name = inputs.get("activation") or ""
        freeze = bool(inputs.get("freeze", False))
        for p in layer.parameters():
            p.requires_grad = not freeze
        act = _make_activation(self._act_name)
        return {"tensor_out": _forward(layer, act, inputs.get("tensor_in"))}


class DropoutNode(BaseNode):
    type_name   = "pt_dropout"
    label       = "Dropout"
    category    = CATEGORY
    subcategory = "Dense"
    description = "nn.Dropout(p). Inactive during graph visualization (eval mode)."

    def __init__(self):
        self._layer: nn.Dropout | None = None
        self._layer_cfg: tuple | None = None
        super().__init__()

    def _get_layer(self, p: float) -> nn.Dropout:
        cfg = (p,)
        if self._layer is None or self._layer_cfg != cfg:
            self._layer = nn.Dropout(p=p)
            self._layer_cfg = cfg
        return self._layer

    def get_layers(self) -> list[nn.Module]:
        return [self._layer] if self._layer is not None else []

    def _setup_ports(self) -> None:
        self.add_input("tensor_in",   PortType.TENSOR, default=None)
        self.add_input("p",           PortType.FLOAT,  default=0.5,
                       description="Drop probability")
        self.add_output("tensor_out", PortType.TENSOR)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        layer = self._get_layer(float(inputs.get("p") or 0.5))
        return {"tensor_out": _forward(layer, None, inputs.get("tensor_in"))}


class BatchNorm1dNode(BaseNode):
    type_name   = "pt_batchnorm1d"
    label       = "BatchNorm1d"
    category    = CATEGORY
    subcategory = "Dense"
    description = "nn.BatchNorm1d(num_features). Running stats update during training."

    def __init__(self):
        self._layer: nn.BatchNorm1d | None = None
        self._layer_cfg: tuple | None = None
        super().__init__()

    def _get_layer(self, num_features: int) -> nn.BatchNorm1d:
        cfg = (num_features,)
        if self._layer is None or self._layer_cfg != cfg:
            self._layer = nn.BatchNorm1d(num_features)
            self._layer_cfg = cfg
        return self._layer

    def get_layers(self) -> list[nn.Module]:
        return [self._layer] if self._layer is not None else []

    def _setup_ports(self) -> None:
        self.add_input("tensor_in",    PortType.TENSOR, default=None)
        self.add_input("num_features", PortType.INT,    default=64)
        self.add_output("tensor_out",  PortType.TENSOR)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        layer = self._get_layer(int(inputs.get("num_features") or 64))
        layer.eval()  # stable for single-sample forward pass on canvas
        return {"tensor_out": _forward(layer, None, inputs.get("tensor_in"))}


class EmbeddingNode(BaseNode):
    type_name   = "pt_embedding"
    label       = "Embedding"
    category    = CATEGORY
    subcategory = "Dense"
    description = "nn.Embedding(num_embeddings, embedding_dim) - lookup table for integer indices."

    def __init__(self):
        self._layer: nn.Embedding | None = None
        self._layer_cfg: tuple | None = None
        super().__init__()

    def _get_layer(self, num_emb: int, emb_dim: int) -> nn.Embedding:
        cfg = (num_emb, emb_dim)
        if self._layer is None or self._layer_cfg != cfg:
            self._layer = nn.Embedding(num_emb, emb_dim)
            self._layer_cfg = cfg
        return self._layer

    def get_layers(self) -> list[nn.Module]:
        return [self._layer] if self._layer is not None else []

    def _setup_ports(self) -> None:
        self.add_input("tensor_in",      PortType.TENSOR, default=None,
                       description="Integer index tensor")
        self.add_input("num_embeddings", PortType.INT,    default=1000)
        self.add_input("embedding_dim",  PortType.INT,    default=64)
        self.add_input("freeze",         PortType.BOOL,   default=False)
        self.add_output("tensor_out",    PortType.TENSOR)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        layer = self._get_layer(
            int(inputs.get("num_embeddings") or 1000),
            int(inputs.get("embedding_dim")  or 64),
        )
        freeze = bool(inputs.get("freeze", False))
        for p in layer.parameters():
            p.requires_grad = not freeze
        return {"tensor_out": _forward(layer, None, inputs.get("tensor_in"))}


class ActivationNode(BaseNode):
    type_name   = "pt_activation"
    label       = "Activation"
    category    = CATEGORY
    subcategory = "Dense"
    description = f"Standalone activation layer. Options: {_ACT_HELP}"

    def __init__(self):
        self._layer: nn.Module | None = None
        self._layer_cfg: tuple | None = None
        super().__init__()

    def _get_layer(self, name: str) -> nn.Module | None:
        cfg = (name,)
        if self._layer_cfg != cfg:
            self._layer = _make_activation(name)
            self._layer_cfg = cfg
        return self._layer

    def get_layers(self) -> list[nn.Module]:
        return [self._layer] if self._layer is not None else []

    def _setup_ports(self) -> None:
        self.add_input("tensor_in",   PortType.TENSOR, default=None)
        self.add_input("activation",  PortType.STRING, default="relu",
                       description=_ACT_HELP)
        self.add_output("tensor_out", PortType.TENSOR)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        act = self._get_layer(inputs.get("activation") or "relu")
        tin = inputs.get("tensor_in")
        if act is None:
            return {"tensor_out": tin}
        return {"tensor_out": _forward(act, None, tin)}


# -- Conv / Spatial -------------------------------------------------------------

class Conv2dNode(BaseNode):
    type_name   = "pt_conv2d"
    label       = "Conv2d"
    category    = CATEGORY
    subcategory = "Conv"
    description = "nn.Conv2d with optional activation. Expects input shape (N, C, H, W)."

    def __init__(self):
        self._layer: nn.Conv2d | None = None
        self._layer_cfg: tuple | None = None
        self._act_name: str = ""
        super().__init__()

    def _get_layer(self, in_ch, out_ch, kernel, stride, padding) -> nn.Conv2d:
        cfg = (in_ch, out_ch, kernel, stride, padding)
        if self._layer is None or self._layer_cfg != cfg:
            self._layer = nn.Conv2d(in_ch, out_ch, kernel_size=kernel,
                                    stride=stride, padding=padding)
            self._layer_cfg = cfg
        return self._layer

    def get_layers(self) -> list[nn.Module]:
        if self._layer is None:
            return []
        modules: list[nn.Module] = [self._layer]
        act = _make_activation(self._act_name)
        if act is not None:
            modules.append(act)
        return modules

    def _setup_ports(self) -> None:
        self.add_input("tensor_in",  PortType.TENSOR, default=None)
        self.add_input("in_ch",      PortType.INT,    default=1)
        self.add_input("out_ch",     PortType.INT,    default=16)
        self.add_input("kernel",     PortType.INT,    default=3)
        self.add_input("stride",     PortType.INT,    default=1)
        self.add_input("padding",    PortType.INT,    default=0)
        self.add_input("activation", PortType.STRING, default="none",
                       description=_ACT_HELP)
        self.add_input("freeze",     PortType.BOOL,   default=False)
        self.add_output("tensor_out", PortType.TENSOR)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        layer = self._get_layer(
            int(inputs.get("in_ch")   or 1),
            int(inputs.get("out_ch")  or 16),
            int(inputs.get("kernel")  or 3),
            int(inputs.get("stride")  or 1),
            int(inputs.get("padding") or 0),
        )
        self._act_name = inputs.get("activation") or ""
        freeze = bool(inputs.get("freeze", False))
        for p in layer.parameters():
            p.requires_grad = not freeze
        act = _make_activation(self._act_name)
        return {"tensor_out": _forward(layer, act, inputs.get("tensor_in"))}


class MaxPool2dNode(BaseNode):
    type_name   = "pt_maxpool2d"
    label       = "MaxPool2d"
    category    = CATEGORY
    subcategory = "Conv"
    description = "nn.MaxPool2d(kernel_size, stride)."

    def __init__(self):
        self._layer: nn.MaxPool2d | None = None
        self._layer_cfg: tuple | None = None
        super().__init__()

    def _get_layer(self, kernel, stride) -> nn.MaxPool2d:
        cfg = (kernel, stride)
        if self._layer is None or self._layer_cfg != cfg:
            self._layer = nn.MaxPool2d(kernel_size=kernel, stride=stride)
            self._layer_cfg = cfg
        return self._layer

    def get_layers(self) -> list[nn.Module]:
        return [self._layer] if self._layer is not None else []

    def _setup_ports(self) -> None:
        self.add_input("tensor_in",   PortType.TENSOR, default=None)
        self.add_input("kernel",      PortType.INT,    default=2)
        self.add_input("stride",      PortType.INT,    default=2)
        self.add_output("tensor_out", PortType.TENSOR)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        layer = self._get_layer(
            int(inputs.get("kernel") or 2),
            int(inputs.get("stride") or 2),
        )
        return {"tensor_out": _forward(layer, None, inputs.get("tensor_in"))}


class AvgPool2dNode(BaseNode):
    type_name   = "pt_avgpool2d"
    label       = "AvgPool2d"
    category    = CATEGORY
    subcategory = "Conv"
    description = "nn.AvgPool2d(kernel_size, stride)."

    def __init__(self):
        self._layer: nn.AvgPool2d | None = None
        self._layer_cfg: tuple | None = None
        super().__init__()

    def _get_layer(self, kernel, stride) -> nn.AvgPool2d:
        cfg = (kernel, stride)
        if self._layer is None or self._layer_cfg != cfg:
            self._layer = nn.AvgPool2d(kernel_size=kernel, stride=stride)
            self._layer_cfg = cfg
        return self._layer

    def get_layers(self) -> list[nn.Module]:
        return [self._layer] if self._layer is not None else []

    def _setup_ports(self) -> None:
        self.add_input("tensor_in",   PortType.TENSOR, default=None)
        self.add_input("kernel",      PortType.INT,    default=2)
        self.add_input("stride",      PortType.INT,    default=2)
        self.add_output("tensor_out", PortType.TENSOR)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        layer = self._get_layer(
            int(inputs.get("kernel") or 2),
            int(inputs.get("stride") or 2),
        )
        return {"tensor_out": _forward(layer, None, inputs.get("tensor_in"))}


class BatchNorm2dNode(BaseNode):
    type_name   = "pt_batchnorm2d"
    label       = "BatchNorm2d"
    category    = CATEGORY
    subcategory = "Conv"
    description = "nn.BatchNorm2d(num_features). Running stats update during training."

    def __init__(self):
        self._layer: nn.BatchNorm2d | None = None
        self._layer_cfg: tuple | None = None
        super().__init__()

    def _get_layer(self, num_features: int) -> nn.BatchNorm2d:
        cfg = (num_features,)
        if self._layer is None or self._layer_cfg != cfg:
            self._layer = nn.BatchNorm2d(num_features)
            self._layer_cfg = cfg
        return self._layer

    def get_layers(self) -> list[nn.Module]:
        return [self._layer] if self._layer is not None else []

    def _setup_ports(self) -> None:
        self.add_input("tensor_in",    PortType.TENSOR, default=None)
        self.add_input("num_features", PortType.INT,    default=16)
        self.add_output("tensor_out",  PortType.TENSOR)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        layer = self._get_layer(int(inputs.get("num_features") or 16))
        layer.eval()
        return {"tensor_out": _forward(layer, None, inputs.get("tensor_in"))}


# -- Sequential (star topology) -------------------------------------------------

class SequentialNode(BaseNode):
    type_name   = "pt_sequential"
    label       = "Sequential"
    category    = "Models"
    subcategory = "Build"
    description = (
        "nn.Sequential(*layers) - star topology. "
        "Wire up to 8 layer nodes (via their tensor_out) into this node."
    )

    def _setup_ports(self) -> None:
        for i in range(1, 9):
            self.add_input(f"layer_{i}", PortType.MODULE, default=None)
        self.add_output("model", PortType.MODULE)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        try:
            layers = [inputs[f"layer_{i}"] for i in range(1, 9)
                      if inputs.get(f"layer_{i}") is not None]
            return {"model": nn.Sequential(*layers) if layers else None}
        except Exception:
            return {"model": None}
