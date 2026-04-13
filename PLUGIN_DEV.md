# Plugin Development Guide

## Quick Start

1. Copy `plugins/_example/` to `plugins/your_domain/`
2. Edit `__init__.py` to register your port types, nodes, and panels
3. Add node files to `plugins/your_domain/nodes/`
4. Restart the app — your nodes appear in the palette

## Plugin Structure

```
plugins/your_domain/
├── __init__.py          ← register(ctx) — entry point
├── nodes/
│   ├── __init__.py      ← import your node classes here
│   └── my_node.py       ← one or more BaseNode subclasses per file
├── panel.py             ← optional sidebar panel builder
└── templates/           ← optional prebuilt graph templates
```

## Node Definition (5 required things)

```python
from core.node import BaseNode, PortType

class MyNode(BaseNode):
    type_name   = "mydomain_operation"    # 1. unique ID (prefix with your domain)
    label       = "My Operation"          # 2. display name
    category    = "My Domain"             # 3. palette group
    description = "Does something useful" # (optional but recommended)

    def _setup_ports(self):               # 4. declare inputs/outputs
        self.add_input("x", PortType.FLOAT, 0.0)
        self.add_output("result", PortType.FLOAT)

    def execute(self, inputs):            # 5. the computation
        return {"result": inputs["x"] * 2}
```

## Port Types

**Built-in (always available):**
| Type | Widget | Description |
|------|--------|-------------|
| `PortType.FLOAT` | Float spinner | editable |
| `PortType.INT` | Int spinner | editable |
| `PortType.BOOL` | Checkbox | editable |
| `PortType.STRING` | Text field | editable |
| `PortType.ANY` | Text label | connect only |

**Register your own** in your plugin's `register()`:
```python
ctx.register_port_type(
    "POINT_CLOUD",
    default=None,              # default value when unconnected
    color=(40, 255, 200, 255), # RGBA pin color
    pin_shape="triangle",      # circle, triangle, quad, + _filled variants
    editable=False,            # False = connect only, True = shows a widget
)
```

Then use it in nodes: `self.add_input("points", "POINT_CLOUD")`

## Export (Code Generation)

Optional but strongly recommended. Makes the Code panel useful.

```python
def export(self, iv, ov):
    # iv = {port_name: upstream_var_name or None}
    # ov = {port_name: variable_name to assign}
    x = self._val(iv, "x")  # helper: returns var name or repr(default)
    return (
        ["import my_lib"],                        # imports needed
        [f"{ov['result']} = my_lib.process({x})"] # code lines
    )
```

## Naming Conventions

- **type_name**: prefix with your domain abbreviation: `rob_trajectory`, `aud_waveform`, `bio_sequence`
- **category**: use your domain name: `"Robotics"`, `"Audio"`, `"Bioinformatics"`
- **port types**: UPPER_SNAKE_CASE: `POINT_CLOUD`, `WAVEFORM`, `DNA_SEQUENCE`

## Panel Registration

Add a sidebar tab next to Training:

```python
def build_my_panel(parent_tag, app):
    import dearpygui.dearpygui as dpg
    dpg.add_text("Status: Ready", parent=parent_tag)
    dpg.add_button(label="Run Simulation", parent=parent_tag,
                   callback=lambda: app._log("[Sim] Running..."))

ctx.register_panel("Simulation", build_my_panel)
```

## Testing

Nodes are headless — test without DPG:

```python
def test_my_node():
    from plugins.your_domain.nodes import MyNode
    result = MyNode().execute({"x": 5.0})
    assert result["result"] == 10.0
```

## Key Rules

1. **Nodes are stateless per-execute.** `execute()` receives fresh inputs each call. Use `self._layer` (cached `nn.Module`) only for trainable PyTorch layers.
2. **Handle None inputs gracefully.** Return sensible defaults, not crashes.
3. **One concept per node.** A node that does too much should be split.
4. **`type_name` must be globally unique.** Prefix with your domain to avoid collisions.
5. **The graph IS the model.** No hidden state, no side-channel data. Everything flows through ports.
6. **No GUI imports in nodes.** Nodes must work headlessly — no `dearpygui`, no `godot`. Both frontends render nodes from the same data.
7. **Port types must match.** Connections enforce type compatibility. INT cannot connect to DATAFRAME. Use correct types, or ANY for pass-through.
8. **Run the tests.** `python -m pytest tests/ -q` must pass before merging.

## Further Reading

See **CONTRIBUTING.md** for the full architecture overview, stability contract, RPC API reference, and plugin review checklist.
