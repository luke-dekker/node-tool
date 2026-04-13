# Contributing to Node Tool

## Architecture Overview

```
core/                    ← Layer 1: Pure Python, zero GUI dependency
  graph.py               Graph execution engine (topological sort)
  node.py                BaseNode ABC, Port, PortType
  port_types.py          PortTypeRegistry — extensible type system
  graph_module.py        Wrap graph as torch.nn.Module for training
  serializer.py          JSON save/load
  exporter.py            Export graph to runnable .py script
  plugins.py             Plugin loader

nodes/                   ← Layer 2: 215+ built-in nodes
  pytorch/               113 nodes (layers, ops, viz, training markers)
  numpy/, pandas/, sklearn/, scipy/
  io/, ai/, viz/, code/, subgraphs/

plugins/                 ← Layer 2b: Domain plugins (loaded at startup)
  robotics/              24 nodes (control, sensors, kinematics, filters)
  _example/              Template for new plugins

server.py                ← Layer 3a: WebSocket JSON-RPC server
gui/                     ← Layer 3b: DearPyGui frontend (legacy)
godot/                   ← Layer 3c: Godot 4 frontend (active development)

tests/                   269 tests, all headless (no GUI dependency)
templates/               Prebuilt graph templates
```

**The golden rule:** Layers 1 and 2 have zero dependency on Layer 3. Nodes never import GUI code. This is what makes dual-frontend (DPG + Godot) possible, and it must stay this way.

---

## Stability Contract

These interfaces are stable. Breaking them breaks every plugin.

### 1. BaseNode interface

```python
class MyNode(BaseNode):
    type_name   = "domain_name"      # globally unique, prefixed
    label       = "Display Name"     # shown in palette + canvas
    category    = "Domain"           # palette group
    description = "What it does"     # shown in inspector

    def _setup_ports(self):          # declare inputs and outputs
    def execute(self, inputs) -> dict:  # the computation
    def export(self, iv, ov):        # optional: code generation
```

**Will not change:** `type_name`, `label`, `category`, `_setup_ports()`, `execute()`, `export()` signatures. These are the contract.

### 2. Port type system

```python
self.add_input("name", PortType.FLOAT, default=0.0)
self.add_input("name", "MY_CUSTOM_TYPE", default=None)
self.add_output("name", PortType.TENSOR)
```

**Built-in types (always available):**

| Type | Editable | Pin | Connects to |
|------|----------|-----|-------------|
| `FLOAT` | Yes (spinner) | Green filled circle | FLOAT, ANY |
| `INT` | Yes (spinner) | Blue filled triangle | INT, ANY |
| `BOOL` | Yes (checkbox) | Red filled quad | BOOL, ANY |
| `STRING` | Yes (text/combo) | Orange filled circle | STRING, ANY |
| `ANY` | No | Grey hollow circle | Anything |
| `TENSOR` | No | Orange hollow circle | TENSOR, ANY |
| `MODULE` | No | Violet hollow triangle | MODULE, ANY |
| `NDARRAY` | No | Sky blue filled triangle | NDARRAY, ANY |
| `DATAFRAME` | No | Emerald filled quad | DATAFRAME, ANY |
| `SERIES` | No | Lime filled circle | SERIES, ANY |
| `SKLEARN_MODEL` | No | Amber hollow triangle | SKLEARN_MODEL, ANY |
| `IMAGE` | No | Pink hollow quad | IMAGE, ANY |
| `DATALOADER` | No | Teal hollow quad | DATALOADER, ANY |

### 3. Connection type enforcement

Connections are validated: `PortTypeRegistry.can_connect(from_type, to_type)` must return `True`. Current rules:

- **Exact match** always connects (TENSOR -> TENSOR)
- **ANY** connects to/from anything (ANY -> TENSOR, FLOAT -> ANY)
- **Everything else is rejected** (INT -> DATAFRAME will fail)

Plugin authors: if you need cross-type connections, register a new shared type or use ANY.

### 4. Server RPC API

All frontends (DPG, Godot, future) communicate through `server.py`:

| Method | Params | Returns | Description |
|--------|--------|---------|-------------|
| `get_registry` | `{}` | `{categories, category_order}` | Full node catalog |
| `add_node` | `{type_name}` | node dict | Instantiate and add to graph |
| `remove_node` | `{node_id}` | `{ok}` | Remove from graph |
| `connect` | `{from_node, from_port, to_node, to_port}` | `{ok}` | Wire two ports |
| `disconnect` | `{from_node, from_port, to_node, to_port}` | `{ok}` | Unwire |
| `set_input` | `{node_id, port_name, value}` | `{ok}` | Set config value |
| `get_node` | `{node_id}` | node dict | Read node state |
| `get_graph` | `{}` | `{nodes, connections}` | Full graph state |
| `execute` | `{}` | `{outputs, terminal}` | Run the graph |
| `clear` | `{}` | `{ok}` | Reset graph |
| `save_graph` | `{path, positions}` | `{ok, path, nodes}` | Save to JSON |
| `load_graph` | `{path}` | `{nodes, connections, positions}` | Load from JSON |
| `export_code` | `{}` | `{code}` | Generate Python script |

**Adding new RPC methods:** Add the method to `NodeToolServer`, then add its name to the `_METHODS` dispatch dict. That's it.

---

## Plugin Development Checklist

Use this checklist when developing or reviewing a plugin:

### Before you start
- [ ] Read this doc and `PLUGIN_DEV.md`
- [ ] Decide on a domain prefix (e.g., `rob_`, `aud_`, `bio_`)
- [ ] Copy `plugins/_example/` as your starting point

### Node quality
- [ ] Every `type_name` is globally unique and uses your domain prefix
- [ ] Every node handles `None` inputs gracefully (return `None`, don't crash)
- [ ] `execute()` never imports GUI code (no `dearpygui`, no `godot`)
- [ ] `execute()` uses `try/except` for operations that can fail (network, file I/O, GPU)
- [ ] Ports use correct types — don't use STRING for data that should be TENSOR
- [ ] `export()` implemented for code generation (or returns a comment explaining why not)
- [ ] Description is clear and fits in one line

### Port types
- [ ] Custom port types registered in `register()` with color + pin_shape
- [ ] `editable=True` only for types the user types in (scalars, short strings)
- [ ] `editable=False` for complex/reference types (connect only)
- [ ] Default values are sensible (not None for required numeric inputs)

### Plugin structure
- [ ] `__init__.py` has a `register(ctx)` function
- [ ] All node classes imported in your `nodes/__init__.py`
- [ ] No circular imports between your plugin and core/
- [ ] No global state that leaks between graph executions

### Testing
- [ ] At least one test per node (instantiate, execute, check output)
- [ ] None-input guard test for every node
- [ ] Tests run headlessly: `python -m pytest tests/ -q`
- [ ] Tests don't require external hardware/services (mock them)

### Compatibility
- [ ] Works with both frontends (DPG + Godot) — means no GUI code in nodes
- [ ] Doesn't modify NODE_REGISTRY directly — use `ctx.register_node(cls)`
- [ ] Doesn't monkey-patch core classes
- [ ] Port types don't shadow built-in names (don't register "FLOAT")

---

## How to Add a New RPC Method

For features that need frontend ↔ backend communication:

1. Add the method to `NodeToolServer` in `server.py`:
```python
def my_method(self, params: dict) -> dict:
    value = params["some_param"]
    # do work...
    return {"result": value}
```

2. Register in the dispatch table:
```python
_METHODS = {
    ...
    "my_method": "my_method",
}
```

3. Call from Godot:
```gdscript
_rpc("my_method", {"some_param": 42}, func(result: Dictionary):
    _log("Got: %s" % str(result))
)
```

4. Call from DPG (if needed):
```python
# Direct call — no RPC needed, same process
result = self._some_method(params)
```

---

## What NOT to Do

- **Don't put node logic in the GUI layer.** If you catch yourself importing `dpg` or writing GDScript in a node file, stop. The node should compute; the frontend should display.
- **Don't add new port types without colors.** Every type needs a color and pin shape or the UI breaks.
- **Don't use `PortType.ANY` as a lazy default.** It defeats type checking. Use the specific type.
- **Don't run torch operations in parallel on Windows.** CUDA deadlocks. The training controller handles threading — don't spawn your own.
- **Don't modify `core/graph.py` unless you're fixing a bug.** The execution engine is the foundation — changes there affect everything.
- **Don't break the test suite.** Run `python -m pytest tests/ -q` before pushing. 269 tests should pass, 6 skip.

---

## Godot Frontend Notes

The Godot frontend (`godot/`) communicates exclusively through `server.py`. It never touches Python nodes directly.

- **GraphEdit** provides native zoom/pan — no custom zoom code needed
- **GraphNode** slots map to data ports only. Config ports live in the inspector.
- **WebSocket buffer** is set to 1MB (`ws.inbound_buffer_size`). If the registry grows past this, increase it.
- **JSON id caveat:** Godot's JSON parser returns all numbers as floats. The client casts `id` to `int` before callback lookup.
- **Project targets Godot 4.4+** but 4.6+ recommended for robotics (better Jolt physics, IK).

---

## Version History

| Date | Change | Impact |
|------|--------|--------|
| 2026-04-12 | Node consolidation: 23 nodes merged to 10 | Old type_names deprecated, shim aliases provided |
| 2026-04-12 | Training panel moved to bottom panel | Layout change only, no API change |
| 2026-04-12 | Config ports hidden from canvas | Nodes are slimmer, inspector is the editing surface |
| 2026-04-13 | Port type enforcement added | Invalid connections now rejected at core level |
| 2026-04-13 | Godot frontend scaffolded | New frontend option, DPG preserved on master |
| 2026-04-13 | Save/load RPC added | Graphs can be saved/loaded from any frontend |
