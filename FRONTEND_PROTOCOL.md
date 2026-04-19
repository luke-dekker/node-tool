# Frontend Protocol

This is the contract between the node-tool backend (`server.py`) and any
frontend. Every GUI — React, Godot, a future native desktop build — uses
this JSON-RPC 2.0 surface. Keep this document in sync with
`server.py:_METHODS`; it is the source of truth for what the backend promises.

The DearPyGui frontend is embedded in-process today and calls `core/` directly
instead of going through this RPC. That's allowed, but new features should be
added to the RPC surface first and consumed by DPG through the same API, so
all three frontends reach parity for free.

## Transport

- **Protocol:** JSON-RPC 2.0 over WebSocket
- **Default URL:** `ws://127.0.0.1:9800`
- **Inbound buffer:** 1 MB (Godot / React must match — the default 64 KB
  truncates large graphs)
- **Request:** `{"jsonrpc": "2.0", "id": N, "method": "<name>", "params": {...}}`
- **Response:** `{"jsonrpc": "2.0", "id": N, "result": {...}}` or
  `{"jsonrpc": "2.0", "id": N, "error": {"code": int, "message": str}}`

## Error codes

| Code    | Meaning               |
|---------|-----------------------|
| -32700  | Parse error (bad JSON)|
| -32601  | Method not found      |
| -32602  | Invalid params        |
| -32603  | Internal error        |

## Methods

### Graph state

#### `get_registry()`
Returns the full palette of available nodes, grouped by category.
- **Params:** none
- **Returns:** `{categories: {name: [{type_name, label, category, subcategory, description, inputs, outputs}]}, category_order: [str]}`

#### `get_graph()`
Snapshot of every node and connection currently in the graph.
- **Params:** none
- **Returns:** `{nodes: {id: node_dict}, connections: [{from_node, from_port, to_node, to_port}]}`

#### `clear()`
Wipe the graph.
- **Returns:** `{ok: true}`

### Node manipulation

#### `add_node(type_name)`
Instantiate a node and add it to the graph.
- **Params:** `type_name: str` (from the palette)
- **Returns:** full serialized node dict (id, type_name, label, category, subcategory, description, inputs, outputs)

#### `remove_node(node_id)`
Delete a node and all its connections.
- **Params:** `node_id: str`
- **Returns:** `{ok: true}`

#### `get_node(node_id)`
Fetch a single node's current state.
- **Params:** `node_id: str`
- **Returns:** serialized node dict

#### `set_input(node_id, port_name, value)`
Set a default value on an input port.
- **Params:** `node_id: str`, `port_name: str`, `value: any`
- **Returns:** `{ok: true}`

### Connections

#### `connect(from_node, from_port, to_node, to_port)`
Wire two ports together. Validates port-type compatibility and rejects cycles.
- **Returns:** `{ok: true}`
- **Errors:** type mismatch, cycle, unknown port

#### `disconnect(from_node, from_port, to_node, to_port)`
Remove a specific edge.
- **Returns:** `{ok: true}`

### Execution

#### `execute()`
Run the graph. Wrapped in `torch.no_grad()` when torch is available so live
previews never train. Stores the outputs server-side for value-summary
read-back.
- **Returns:** `{outputs: {node_id: {port: summary_str}}, terminal: [str], errors: {}}`

#### `export_code()`
Generate a standalone Python script equivalent to the current graph.
- **Returns:** `{code: str}`

### Persistence

#### `save_graph(path, positions={})`
Write the graph to a JSON file.
- **Params:** `path: str`, `positions: {node_id: [x, y]}` — positions come from
  the frontend since the backend doesn't track layout.
- **Returns:** `{ok: true, path: str, nodes: int}`

#### `load_graph(path)`
Replace the graph from a JSON file.
- **Returns:** `{nodes, connections, positions}`

#### `serialize_graph(positions={})`
In-memory serialize — lets a browser frontend export without file I/O.
- **Returns:** `{version: 1, nodes: [{id, type_name, pos, inputs}], connections}`

#### `deserialize_graph(data={})`
Replace the graph from an in-memory dict. Unknown node types are skipped.
- **Returns:** `{nodes, connections, positions}`

### Templates

#### `get_templates()`
List shipped templates.
- **Returns:** `{templates: [{label, description}]}`

#### `load_template(label)`
Replace the graph with a template.
- **Returns:** `{nodes, connections, positions}`

### Plugins & panels

#### `get_plugin_panels()`
Panels registered by plugins — specs + legacy merged. Spec-driven panels
take precedence for the same label.
- **Returns:** `{panels: [str]}`

#### `get_panel_specs()`
Every plugin's PanelSpec as a serialized dict. Frontends render these
natively — see `core/panel.py` for the schema (sections: form / dynamic_form
/ status / plot / buttons / custom). No frontend should hardcode per-plugin
panel UI; read the spec.
- **Returns:** `{panels: {label: PanelSpec, ...}}`

#### `get_marker_groups()`
All training-marker node groups and their modalities. Backed by
`BaseNode.marker_role` (see `core.node.MarkerRole`), not type-name strings.
Used as the `source_rpc` for the training panel's datasets dynamic_form.
- **Returns:** `{groups: {name: {modalities: [str], has_output: bool}}}`

### Training (pytorch plugin)

All training methods live in `plugins/pytorch/training_orchestrator.py` and
are passthrough handlers on the server. DPG calls them in-process through
`NodeApp.dispatch_rpc`; other frontends call them over WebSocket.

#### `train_start(params)`
`params` is collected from the Training panel:
`{epochs, lr, optimizer, loss, device, datasets: {group: {path, batch_size, split, seq_len, chunk_size}}}`.
- **Returns:** `{ok: bool, error?: str, task_names?: [str], n_params?: int}`

#### `train_pause()`, `train_resume()`, `train_stop()`
- **Returns:** `{status: str}`

#### `train_save_model(path)`
- **Returns:** `{ok: bool, path?, n_params?, error?}`

#### `get_training_state()`
Live snapshot used by the status section.
- **Returns:** `{status, epoch_str, best_loss, last_loss, error}`

#### `get_training_losses()`
Series data for the loss plot.
- **Returns:** `{series: {train: [float], val: [float]}}`

#### `drain_training_logs()`
Pop accumulated log lines since last call.
- **Returns:** `{lines: [str]}`

### Robotics (robotics plugin)

Stubbed today — serial operations log to the controller's line buffer.
Real hardware wiring goes in `plugins/robotics/robotics_controller.py`
without touching any GUI.

- `robotics_list_ports()` → `{ports: [str], error?}`
- `robotics_connect(port, baud)` → `{ok, port?, baud?, error?}`
- `robotics_disconnect()` → `{ok}`
- `robotics_send(cmd)` → `{ok, error?}`
- `get_robotics_state()` → `{connected, port, baud, log_tail}`
- `get_robotics_log()` → `{lines: [str]}`

## Shared types

These shapes appear in multiple responses.

### `NodeDict`
```
{
  id:          str,
  type_name:   str,
  label:       str,
  category:    str,
  subcategory: str,
  description: str,
  inputs:      {port_name: PortDict},
  outputs:     {port_name: PortDict}
}
```

### `PortDict`
```
{
  name:          str,
  port_type:     str,   # see PortTypeRegistry — color/pin_shape derived
  is_input:      bool,
  default_value: any,
  description:   str,
  choices:       [str]  # non-empty → render as dropdown
}
```

### `ConnectionDict`
```
{ from_node: str, from_port: str, to_node: str, to_port: str }
```

## Stability guarantees

- Method names and param shapes in this doc are **stable** — renaming one is
  a breaking change that requires bumping all frontend clients.
- Adding a new method or a new optional param is **non-breaking**.
- New port types (e.g. pytorch plugin's `TENSOR`) appear dynamically via
  `get_registry`. Frontends should read color and pin shape from the registry,
  never hardcode per-type styling.
- Category names, node type_names, and port type names are NOT stable long-term
  — they come from plugins. Frontends should always re-fetch the registry
  rather than caching it across sessions.

## Adding a new method

1. Add a handler method on `NodeToolServer` in `server.py`.
2. Register its name in `_METHODS` (line ~437).
3. Document it here — method, params, returns, errors.
4. Add a tiny test in `tests/test_server_rpc.py` (or create that file if needed).
5. Ship client support in every frontend that cares, against this doc.
