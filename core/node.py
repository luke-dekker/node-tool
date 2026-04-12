"""BaseNode, PortType, and Port — the three primitives every node depends on.

PortType is now string-based with an extensible registry (core/port_types.py).
Plugins register domain-specific types at startup; the core ships with base
types (FLOAT, INT, BOOL, STRING, ANY). All existing code that does
`PortType.FLOAT` still works — it's now the string "FLOAT" instead of an
enum value. Comparisons, coercion, and default values all go through the
registry.
"""

from __future__ import annotations
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

# ── Port types (registry-based) ──────────────────────────────────────────────
# Import the PortType class and registry from the new module.
# PortType.FLOAT etc. are string constants. PortTypeRegistry holds the
# coerce/default/color/pin_shape info for each registered type.
from core.port_types import PortType, PortTypeRegistry  # noqa: F401


@dataclass
class Port:
    name: str
    port_type: str          # string type name, e.g. "FLOAT", "TENSOR"
    is_input: bool
    default_value: Any = field(default=None)
    description: str = ""
    choices: list = field(default_factory=list)   # non-empty -> render as combo

    def __post_init__(self) -> None:
        if self.default_value is None:
            self.default_value = PortTypeRegistry.get_default(self.port_type)


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

    @property
    def safe_id(self) -> str:
        """A 6-char Python-safe slice of self.id, suitable for variable names.

        Real graphs use uuid4 IDs whose first 6 chars are always hex (no dashes),
        but hand-edited subgraph JSON files may use arbitrary string IDs that
        contain dashes or other non-alphanumeric chars. This sanitizes them
        into valid Python identifier fragments so layer node export() methods
        can write things like `f'_lin_{self.safe_id}'` without producing
        broken code on edge cases.
        """
        s = self.id[:6]
        return "".join(c if (c.isalnum() or c == "_") else "_" for c in s)

    @abstractmethod
    def _setup_ports(self) -> None:
        """Define inputs and outputs by populating self.inputs and self.outputs."""

    @abstractmethod
    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Execute this node with the given input values.
        Returns a dict mapping output port names to their values.
        """

    def add_input(self, name: str, port_type: str,
                  default: Any = None, description: str = "",
                  choices: list | None = None) -> Port:
        p = Port(name=name, port_type=port_type, is_input=True,
                 default_value=default, description=description,
                 choices=choices or [])
        if p.default_value is None:
            p.default_value = PortTypeRegistry.get_default(port_type)
        self.inputs[name] = p
        return p

    def add_output(self, name: str, port_type: str,
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

    def inspector_ui(self, parent: str, app) -> None:
        """Optional: draw custom DearPyGui widgets for this node instance.

        Called by the Inspector panel after the default port/output readout,
        each time this node is selected. `parent` is the dpg container tag to
        attach widgets to; `app` is the main `App` for graph access and
        `_log`. State should live on `self` (the node instance), not in
        widget tags — widgets are rebuilt on every re-selection.

        Default: no-op. Nodes that want action buttons, live previews, or
        richer controls than ports can provide override this.
        """
        return None

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.id[:8]}>"
