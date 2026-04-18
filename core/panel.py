"""PanelSpec — GUI-agnostic description of a plugin's side-panel.

Every GUI renders the same spec natively. The spec is JSON-serializable
(dataclasses → asdict), so it travels over the RPC surface unchanged.

Section kinds:
  - form           : static list of fields, one form
  - dynamic_form   : one form per item from `source_rpc` — used for
                     per-marker-group dataset config
  - status         : read-only display of values fetched from `source_rpc`
  - plot           : live line chart from `source_rpc`
  - buttons        : row of action buttons that fire RPC calls
  - custom         : escape hatch — GUI-specific rendering keyed on `custom_kind`

Frontend contract:
  * The frontend fetches specs via `get_panel_specs()` RPC.
  * Fields in form / dynamic_form sections are rendered with widgets that
    match `type` (str → text, int → int input, float → number input,
    bool → checkbox, choice → dropdown).
  * Status / plot sections poll `source_rpc` every `poll_ms` while visible.
  * Buttons invoke `rpc` with the current field values from every section
    listed in `collect` merged into params.

Custom sections are the pressure valve. If a panel needs a live loss plot
with a specific look, a serial monitor, or any renderer we haven't declared
here, add a custom section with a well-named `custom_kind`. GUIs that
recognize the kind render it; GUIs that don't show a "requires <name>
renderer" placeholder.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any


# ── Fields + actions ────────────────────────────────────────────────────────

@dataclass
class Field:
    """A single input in a form section."""
    id:        str
    type:      str                       # "str" | "int" | "float" | "bool" | "choice"
    label:     str = ""
    default:   Any = None
    choices:   list[str] = field(default_factory=list)
    hint:      str = ""
    min:       float | None = None
    max:       float | None = None
    step:      float | None = None


@dataclass
class Action:
    """A button in a buttons section."""
    id:      str
    label:   str
    rpc:     str                          # RPC method name to invoke on click
    collect: list[str] = field(default_factory=list)  # section ids whose fields to gather


# ── Sections ────────────────────────────────────────────────────────────────

@dataclass
class Section:
    """Base — every section has an id and a kind. Do not instantiate directly."""
    id:    str
    kind:  str
    label: str = ""


@dataclass
class FormSection(Section):
    kind:   str = "form"
    fields: list[Field] = field(default_factory=list)


@dataclass
class DynamicFormSection(Section):
    """One form per item returned by `source_rpc`.

    source_rpc must return `{items: [{key: str, label: str, **ctx}, ...]}`.
    `item_label_template` is a Python format string evaluated against each
    item dict. `fields` are rendered once per item with ids scoped by key.
    """
    kind:                str = "dynamic_form"
    source_rpc:          str = ""
    item_label_template: str = "{label}"
    fields:              list[Field] = field(default_factory=list)
    empty_hint:          str = ""        # shown when items list is empty


@dataclass
class StatusSection(Section):
    """Read-only display of key/value pairs.

    source_rpc returns a flat dict; `fields` names which keys to display and
    in what order. Field.type controls formatting (str, int, float, bool).
    """
    kind:       str = "status"
    source_rpc: str = ""
    fields:     list[Field] = field(default_factory=list)
    poll_ms:    int = 500


@dataclass
class PlotSection(Section):
    """Live line plot.

    source_rpc returns `{series: {name: [value, ...]}, x_values: [..]}` or
    `{series: {name: [(x, y), ...]}}`. The frontend picks whichever shape
    matches what it can render.
    """
    kind:       str = "plot"
    source_rpc: str = ""
    y_label:    str = "value"
    x_label:    str = "epoch"
    poll_ms:    int = 500


@dataclass
class ButtonsSection(Section):
    kind:    str = "buttons"
    actions: list[Action] = field(default_factory=list)


@dataclass
class CustomSection(Section):
    """Escape hatch for rendering that doesn't fit the core kinds.

    Every GUI must decide whether it has a renderer for `custom_kind`.
    Unknown kinds render as a placeholder.
    """
    kind:        str = "custom"
    custom_kind: str = ""
    params:      dict = field(default_factory=dict)


# ── Panel ───────────────────────────────────────────────────────────────────

@dataclass
class PanelSpec:
    """A plugin's side-panel, described once, rendered by every GUI."""
    label:    str
    sections: list[Section] = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-safe dict for RPC transport. `kind` discriminates sections."""
        return asdict(self)
