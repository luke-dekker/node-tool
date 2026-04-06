"""BaseNode, PortType, and Port — the three primitives every node depends on."""

from __future__ import annotations
import uuid
from abc import ABC, abstractmethod
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any


# ── Port types ────────────────────────────────────────────────────────────────

class PortType(Enum):
    FLOAT = auto()
    INT = auto()
    BOOL = auto()
    STRING = auto()
    ANY = auto()
    TENSOR = auto()       # torch.Tensor
    MODULE = auto()       # torch.nn.Module
    DATALOADER = auto()   # torch.utils.data.DataLoader
    OPTIMIZER = auto()    # torch.optim.Optimizer
    LOSS_FN = auto()      # callable loss function
    DATAFRAME = auto()    # pd.DataFrame
    NDARRAY = auto()      # np.ndarray
    SERIES = auto()       # pd.Series
    SKLEARN_MODEL = auto() # sklearn estimator
    IMAGE = auto()        # np.ndarray RGB uint8 (H,W,3) — for inline visualization
    SCHEDULER = auto()    # torch.optim.lr_scheduler
    DATASET = auto()      # torch.utils.data.Dataset
    TRANSFORM = auto()    # torchvision / torchaudio transform callable

    def default_value(self) -> Any:
        defaults = {
            PortType.FLOAT: 0.0,
            PortType.INT: 0,
            PortType.BOOL: False,
            PortType.STRING: "",
            PortType.ANY: None,
            PortType.TENSOR: None,
            PortType.MODULE: None,
            PortType.DATALOADER: None,
            PortType.OPTIMIZER: None,
            PortType.LOSS_FN: None,
            PortType.DATAFRAME: None,
            PortType.NDARRAY: None,
            PortType.SERIES: None,
            PortType.SKLEARN_MODEL: None,
            PortType.IMAGE: None,
            PortType.SCHEDULER: None,
            PortType.DATASET: None,
            PortType.TRANSFORM: None,
        }
        return defaults[self]

    def coerce(self, value: Any) -> Any:
        """Coerce a value to this port's type."""
        if value is None:
            return self.default_value()
        try:
            if self == PortType.FLOAT:
                return float(value)
            elif self == PortType.INT:
                return int(value)
            elif self == PortType.BOOL:
                if isinstance(value, str):
                    return value.lower() not in ("", "0", "false", "no")
                return bool(value)
            elif self == PortType.STRING:
                return str(value)
            elif self == PortType.TENSOR:
                if isinstance(value, (int, float, list)):
                    try:
                        import torch
                        return torch.tensor(value)
                    except Exception:
                        return None
                return value
            else:
                return value
        except (ValueError, TypeError):
            return self.default_value()


@dataclass
class Port:
    name: str
    port_type: PortType
    is_input: bool
    default_value: Any = field(default=None)
    description: str = ""
    choices: list = field(default_factory=list)   # non-empty -> render as combo

    def __post_init__(self) -> None:
        if self.default_value is None:
            self.default_value = self.port_type.default_value()


# ── Base node ─────────────────────────────────────────────────────────────────

class BaseNode(ABC):
    """Base class for all nodes in the graph."""

    # Override in subclasses
    type_name: str = "base"
    label: str = "Base Node"
    category: str = "Misc"
    subcategory: str = ""      # optional — palette groups within a category
    description: str = "A base node."

    def __init__(self) -> None:
        self.id: str = str(uuid.uuid4())
        self.inputs: dict[str, Port] = {}
        self.outputs: dict[str, Port] = {}
        self._setup_ports()

    @abstractmethod
    def _setup_ports(self) -> None:
        """Define inputs and outputs by populating self.inputs and self.outputs."""

    @abstractmethod
    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Execute this node with the given input values.
        Returns a dict mapping output port names to their values.
        """

    def add_input(self, name: str, port_type: PortType,
                  default: Any = None, description: str = "",
                  choices: list | None = None) -> Port:
        p = Port(name=name, port_type=port_type, is_input=True,
                 default_value=default, description=description,
                 choices=choices or [])
        if p.default_value is None:
            p.default_value = port_type.default_value()
        self.inputs[name] = p
        return p

    def add_output(self, name: str, port_type: PortType,
                   description: str = "") -> Port:
        p = Port(name=name, port_type=port_type, is_input=False,
                 description=description)
        self.outputs[name] = p
        return p

    def get_input_default(self, name: str) -> Any:
        if name in self.inputs:
            return self.inputs[name].default_value
        return None

    # ── Export helpers (used inside each node's export() method) ─────────────

    def _val(self, iv: dict, port_name: str) -> str:
        """Return a Python expression: connected variable name OR literal default."""
        v = iv.get(port_name)
        if v is not None:
            return v
        default = self.inputs[port_name].default_value
        if isinstance(default, bool):
            return "True" if default else "False"
        if isinstance(default, str):
            return repr(default)
        if default is None:
            return "None"
        return repr(default)

    def _axis(self, iv: dict, port_name: str = "axis") -> str:
        """Translate axis=-99 sentinel → None, else use value."""
        v = iv.get(port_name)
        if v is not None:
            return v
        default = self.inputs[port_name].default_value
        if default is None or (isinstance(default, int) and default == -99):
            return "None"
        return repr(default)

    def export(self, iv: dict, ov: dict) -> tuple[list[str], list[str]]:
        """Override to provide Python code export. Returns (imports, lines)."""
        return [], [f"# [{self.label}]: export not supported"]

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.id[:8]}>"
