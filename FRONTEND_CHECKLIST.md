# Frontend Feature Checklist

Every node-tool frontend implements the same set of surfaces. New frontends
(native desktop, mobile, web variants) work against this list instead of
re-deriving it from the DPG version each time.

The layout is intentionally loose — the features are the contract, not the
pixel layout. DPG uses a fixed 3-column arrangement; React uses the same
conceptually with React Flow; Godot uses native GraphEdit. All must cover the
features below.

## Mandatory surfaces

### 1. Palette
- Lists every node returned by `get_registry()`.
- Grouped by `category` and `subcategory`.
- Searchable by label and description.
- Category ordering respects `category_order` from the registry.
- Click/drag → `add_node(type_name)` RPC.

### 2. Node editor
- Canvas with pan and zoom.
- Nodes have a title bar (label), input ports on the left, output ports on the right.
- Port color/shape read from the port type registry (the RPC returns the type
  name; GUI looks up styling). Never hardcode per-type colors.
- Config ports (editable = true on the port type, e.g. FLOAT, INT, BOOL,
  STRING) render as an inline widget in the node body. Data ports (TENSOR,
  NDARRAY, etc.) render as a labeled pin only.
- Drag output → input creates an edge via `connect(...)`. Failures raise a
  toast/status-bar message (type mismatch, cycle).

### 3. Inspector
- Per-selected-node property editor.
- Editable config ports render as input widgets (spinner, combo, text field,
  checkbox) matching the port type.
- Values written to the backend via `set_input(node_id, port, value)`.
- Optional per-node extra UI: see `BaseNode.inspector_ui` (currently a DPG
  hook, will become a declarative schema in the "inspector-schema" initiative
  so every frontend renders it).

### 4. Terminal / output log
- Appends lines from `execute()` responses (the `terminal` field).
- Appends training lines from the training panel's event drain.
- Monospace text, scrollback.

### 5. File menu
- New (`clear`)
- Open (`load_graph`)
- Save / Save As (`save_graph` with node positions)
- Templates submenu (`get_templates` → `load_template`)
- Export Python (`export_code` → file download / save dialog)
- Optional: Pack / Expand subgraph (DPG-only for now)

### 6. Code view
- Tab that shows the current `export_code()` output.
- Re-fetches on graph change or on tab focus.

### 7. Training panel
- Driven entirely by `get_marker_groups()` — no hardcoding of pytorch node
  type names. One row of widgets per group (path, batch_size, split, seq_len,
  chunk_size).
- Hyperparameters: epochs, lr, optimizer (choices from
  `plugins.pytorch._factories.OPTIMIZER_CHOICES`), loss
  (`LOSS_CHOICES`), device.
- Start / Pause / Resume / Stop controls.
- Live loss plot (at minimum a line chart over epochs).
- Status label (Idle / Running / Paused / Done / Error).

### 8. Theming
- Dark and light themes shipped.
- Colors sourced from a single palette (`gui/theme.py` on DPG, CSS variables
  on React, theme resource on Godot).
- Accent, bg, text, and semantic (ok/warn/err) colors must exist.

### 9. Keyboard shortcuts
- Delete — remove selected nodes / edges
- Ctrl+Z / Ctrl+Y — undo / redo
- Ctrl+C / Ctrl+V — copy / paste nodes
- Ctrl+S — save
- Ctrl+O — open
- Ctrl+N — new
- Ctrl+E — export code

## Optional surfaces

- Minimap
- Node search in canvas (Ctrl+F)
- Recent files list
- Hot reload on templates / custom nodes
- Plugin-registered panels (Training is the one everyone implements; Robotics
  is pytorch-plugin-specific and optional)

## Protocol adherence

Every feature maps to one or more methods in `FRONTEND_PROTOCOL.md`. A
frontend that only reads `get_graph` + `execute` already gets a viewer for
free; a full editor needs the mutation methods (add/remove/connect/set_input).

## Feature matrix (current)

Update this table when a frontend catches up.

| Feature                 | DPG | React | Godot |
|-------------------------|-----|-------|-------|
| Palette                 | ✓   | ✓     | ✓     |
| Node editor             | ✓   | ✓     | ✓     |
| Inspector               | ✓   | ✓     | ✓     |
| Terminal                | ✓   | ✓     | ✓     |
| File menu               | ✓   | ✓     | ✓     |
| Code view               | ✓   | ✓     | ✓     |
| Training panel          | ✓   | skel  | skel  |
| Theme: dark             | ✓   | ✓     | —     |
| Keyboard shortcuts      | ✓   | partial| partial|
| Minimap                 | ✓   | ✓     | ✓     |
| Inspector custom UI     | partial (DPG hook) | — | — |

## See also
- `FRONTEND_PROTOCOL.md` — the RPC surface
- `ARCHITECTURE.md` — how plugins, core, and frontend fit together
- `PLUGIN_DEV.md` — writing a plugin that adds nodes / panels
