"""Extensible port type registry — the foundation for domain-agnostic graphs.

Instead of a fixed enum with ML-specific types baked in, PortType is a
string-based registry. The core ships with 5 base types (FLOAT, INT, BOOL,
STRING, ANY). Plugins register domain-specific types at startup:

    from core.port_types import PortTypeRegistry
    PortTypeRegistry.register("TENSOR", default=None, color=(255,120,40,255))

Port.port_type stores the string name (e.g., "FLOAT"). The GUI reads colors
and pin shapes from the registry. Coercion and default values are looked up
dynamically. Any domain can add types without touching core code.

Usage in nodes:
    # Old (enum):  self.add_input("x", PortType.FLOAT, 0.0)
    # New (same!): self.add_input("x", PortType.FLOAT, 0.0)
    # PortType.FLOAT is now the string "FLOAT", not an enum value.
    # All comparisons still work: if ptype == PortType.FLOAT: ...
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class PortTypeInfo:
    """Everything the system needs to know about a port type."""
    name:          str
    default_value: Any                         = None
    coerce:        Callable[[Any], Any] | None = None   # None → passthrough
    color:         tuple[int, ...]             = (160, 160, 180, 255)
    pin_shape:     str                         = "circle"  # circle, triangle, quad, circle_filled, etc.
    description:   str                         = ""
    editable:      bool                        = False  # True → widget on canvas, False → text label


class PortTypeRegistry:
    """Global registry of port types. Plugins call register() at startup."""

    _types: dict[str, PortTypeInfo] = {}

    @classmethod
    def register(cls, name: str, *,
                 default: Any = None,
                 coerce: Callable[[Any], Any] | None = None,
                 color: tuple[int, ...] = (160, 160, 180, 255),
                 pin_shape: str = "circle",
                 description: str = "",
                 editable: bool = False) -> str:
        """Register a port type. Returns the name string for convenience.

        editable=True means the port shows an input widget on the canvas
        (like float spinner, text field). editable=False means the port shows
        a text label and must be connected — this is the default for complex
        types like TENSOR, MODULE, POINT_CLOUD, etc.
        """
        cls._types[name] = PortTypeInfo(
            name=name, default_value=default, coerce=coerce,
            color=color, pin_shape=pin_shape, description=description,
            editable=editable,
        )
        return name

    @classmethod
    def get(cls, name: str) -> PortTypeInfo | None:
        return cls._types.get(name)

    @classmethod
    def get_default(cls, name: str) -> Any:
        info = cls._types.get(name)
        return info.default_value if info else None

    @classmethod
    def coerce_value(cls, name: str, value: Any) -> Any:
        """Coerce a value to the given port type. Falls back to passthrough."""
        if value is None:
            return cls.get_default(name)
        info = cls._types.get(name)
        if info is None or info.coerce is None:
            return value
        try:
            return info.coerce(value)
        except (ValueError, TypeError):
            return cls.get_default(name)

    @classmethod
    def get_color(cls, name: str) -> tuple[int, ...]:
        info = cls._types.get(name)
        return info.color if info else (160, 160, 180, 255)

    @classmethod
    def get_pin_shape(cls, name: str) -> str:
        info = cls._types.get(name)
        return info.pin_shape if info else "circle"

    @classmethod
    def is_editable(cls, name: str) -> bool:
        """True if this type gets a widget on the canvas (FLOAT/INT/BOOL/STRING)."""
        info = cls._types.get(name)
        return info.editable if info else False

    @classmethod
    def all_types(cls) -> dict[str, PortTypeInfo]:
        return dict(cls._types)


# ── Coercion helpers for base types ──────────────────────────────────────────

def _coerce_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.lower() not in ("", "0", "false", "no")
    return bool(value)


# ── Base types (always available, any domain) ────────────────────────────────

PortTypeRegistry.register("FLOAT",  default=0.0,   coerce=float, editable=True,
                          color=(80, 200, 120, 255), pin_shape="circle_filled",
                          description="Floating-point scalar")
PortTypeRegistry.register("INT",    default=0,     coerce=int, editable=True,
                          color=(80, 140, 220, 255), pin_shape="triangle_filled",
                          description="Integer scalar")
PortTypeRegistry.register("BOOL",   default=False, coerce=_coerce_bool, editable=True,
                          color=(220, 100, 80, 255), pin_shape="quad_filled",
                          description="Boolean")
PortTypeRegistry.register("STRING", default="",    coerce=str, editable=True,
                          color=(220, 180, 80, 255), pin_shape="circle_filled",
                          description="Text string")
PortTypeRegistry.register("ANY",    default=None,  coerce=None,
                          color=(160, 160, 180, 255), pin_shape="circle",
                          description="Any type (passthrough)")


# Data science types (registered here for now — will move to their plugins
# as numpy/pandas/sklearn follow pytorch's plugin-owned port type pattern).
PortTypeRegistry.register("NDARRAY",       default=None,
                          color=(80, 180, 255, 255),  pin_shape="triangle_filled",
                          description="numpy.ndarray")
PortTypeRegistry.register("DATAFRAME",     default=None,
                          color=(50, 205, 120, 255),  pin_shape="quad_filled",
                          description="pandas.DataFrame")
PortTypeRegistry.register("SERIES",        default=None,
                          color=(150, 230, 80, 255),  pin_shape="circle_filled",
                          description="pandas.Series")
PortTypeRegistry.register("SKLEARN_MODEL", default=None,
                          color=(255, 160, 50, 255),  pin_shape="triangle",
                          description="sklearn estimator")
PortTypeRegistry.register("IMAGE",         default=None,
                          color=(255, 80, 180, 255),  pin_shape="quad",
                          description="RGB uint8 ndarray (H, W, 3)")


# ── Convenience class for backward compat ────────────────────────────────────
# Code that does `PortType.FLOAT` gets the string "FLOAT".
# Code that does `port.port_type == PortType.FLOAT` still works.
# Code that calls `PortType.coerce(value)` uses the registry.

class PortType:
    """Backward-compatible PortType constants. Each is just a string."""
    FLOAT        = "FLOAT"
    INT          = "INT"
    BOOL         = "BOOL"
    STRING       = "STRING"
    ANY          = "ANY"
    TENSOR       = "TENSOR"
    MODULE       = "MODULE"
    DATALOADER   = "DATALOADER"
    OPTIMIZER    = "OPTIMIZER"
    LOSS_FN      = "LOSS_FN"
    SCHEDULER    = "SCHEDULER"
    DATASET      = "DATASET"
    TRANSFORM    = "TRANSFORM"
    NDARRAY      = "NDARRAY"
    DATAFRAME    = "DATAFRAME"
    SERIES       = "SERIES"
    SKLEARN_MODEL = "SKLEARN_MODEL"
    IMAGE        = "IMAGE"

    @staticmethod
    def default_value(name: str) -> Any:
        return PortTypeRegistry.get_default(name)

    @staticmethod
    def coerce(name: str, value: Any) -> Any:
        return PortTypeRegistry.coerce_value(name, value)
