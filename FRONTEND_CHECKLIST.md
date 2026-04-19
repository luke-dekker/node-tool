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
Rendered entirely from the PanelSpec returned by `get_panel_specs()` for
the `"Training"` label. No per-frontend training UI code exists or should
be written — see the SpecRenderer section below. The spec declares the
dataset dynamic_form, hyperparameters form, Start/Pause/Stop buttons, live
status section, and the `loss_plot` custom section. Change the layout in
`plugins/pytorch/_panel_training.py` and every frontend updates.

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

## Plugin panels — the spec contract

Every plugin side panel (Training, Robotics, anything new) is described
**once** in Python as a `core.panel.PanelSpec` and rendered **natively** by
every GUI. A GUI has one generic renderer: DPG's lives in
`gui/panel_renderer.py`, React's in `web/src/panels/SpecRenderer.tsx`.

### What a GUI implements

1. Fetch `get_panel_specs()` on connect and keep the result in store.
2. For each plugin tab, call `render(spec)` with your GUI's renderer.
3. Support all section kinds from `core/panel.py`:
   - `form` — typed inputs (str/int/float/bool/choice)
   - `dynamic_form` — one form per item returned by `source_rpc`
   - `status` — polled key/value readout
   - `plot` — polled line chart
   - `buttons` — action buttons that call `rpc` with `collect`ed params
   - `custom` — dispatched to a renderer registered by `custom_kind`
4. Poll status / plot / dynamic_form sections at `poll_ms` (default 500 ms).
5. On a button click, gather values from the sections in `action.collect`
   (static forms flatten to top-level keys; dynamic forms nest under the
   section id), call the action's `rpc`, surface any `error` in the terminal.

### The custom section escape hatch

Any panel piece that doesn't fit the core kinds becomes a `CustomSection`
with a named `custom_kind`. Current kinds in use:

- `loss_plot` — live line chart (`series: [train, val]` by default)
- `log_tail` — scrolling read-only log readout

A GUI that recognizes a `custom_kind` renders it; a GUI that doesn't shows
a "[no renderer for kind X]" placeholder. Adding a new kind = add a renderer
in each GUI's spec renderer (2-3 files changed total, one per frontend).

### What you do NOT do

- Do not hardcode the Training panel layout in your frontend. Change
  `plugins/pytorch/_panel_training.py` instead.
- Do not hardcode plugin-specific RPC calls in your frontend. The spec
  tells you which RPC each button and polled section uses.
- Do not special-case known plugin names. If a plugin registers a panel,
  render it; if not, the tab doesn't appear.

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
| Training panel (spec)   | ✓   | ✓     | —     |
| Robotics panel (spec)   | ✓   | ✓     | —     |
| SpecRenderer            | ✓   | ✓     | —     |
| Theme: dark             | ✓   | ✓     | —     |
| Keyboard shortcuts      | ✓   | partial| partial|
| Minimap                 | ✓   | ✓     | ✓     |
| Inspector custom UI     | partial (DPG hook) | — | — |

## See also
- `FRONTEND_PROTOCOL.md` — the RPC surface
- `ARCHITECTURE.md` — how plugins, core, and frontend fit together
- `PLUGIN_DEV.md` — writing a plugin that adds nodes / panels
