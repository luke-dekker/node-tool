# node-tool

A visual node-based tool for building, training, and deploying machine-learning
and robotics workflows. The graph IS the program — every layer, dataset,
training loop, agent, and device connection is a node you wire together.

```
┌──────────────────────────────────────────────────────────────────┐
│                              core/                               │
│   Graph + BaseNode + PortType registry + PanelSpec contract.     │
│   Pure Python, no domain knowledge, no GUI imports.              │
└──────────────────────────────────────────────────────────────────┘
                                 ▲
                                 │ (only public API)
                                 │
┌────────────────────────┬───────┴────────┬─────────────────────────┐
│      plugins/          │     nodes/     │      templates/         │
│  pytorch, agents,      │ one .py per    │ shipped example graphs  │
│  robotics, numpy,      │ node, owned    │ for each plugin combo   │
│  pandas, sklearn,      │ by its plugin  │                         │
│  scipy, io, python     │                │                         │
│                        │                │                         │
│  Each plugin owns:     │                │                         │
│   • port types         │                │                         │
│   • node classes       │                │                         │
│   • orchestrator       │                │                         │
│   • PanelSpec          │                │                         │
│  No GUI imports. No    │                │                         │
│  cross-plugin imports. │                │                         │
└────────────────────────┴────────────────┴─────────────────────────┘
                                 ▲
                                 │ (RPC + PanelSpec only)
                                 │
┌────────────┬─────────────────────────────────┬────────────────────┐
│   gui/     │            web/                 │      godot/        │
│  DearPy    │      React + Vite               │   Godot scenes     │
│  desktop   │  (talks to server.py via        │  (in progress —    │
│  in-proc   │   WebSocket JSON-RPC)           │   themes scaffold) │
└────────────┴─────────────────────────────────┴────────────────────┘
```

## Architectural rules

These exist so plugins are interchangeable and GUIs are interchangeable.
Future contributors are expected to honor them.

1. **Plugins are GUI-clean.** A plugin in `plugins/<name>/` and its node files
   in `nodes/<name>/` must NOT import from `gui/`, `web/`, or `godot/`. The
   plugin describes its panel once via `PanelSpec` (a GUI-agnostic dataclass);
   every frontend renders the same spec natively.
2. **GUIs are plugin-agnostic.** Frontends auto-discover plugins via the
   `get_panel_specs` RPC. They do not maintain hardcoded plugin lists. Adding
   a new plugin requires zero edits to any frontend — it just appears.
3. **Cross-plugin contact only via public RPC.** A plugin must not `import`
   from another plugin. If `plugins/agents/` needs training data from
   `plugins/pytorch/`, it goes through the orchestrator's RPC surface, same
   as any external client.
4. **Heavy imports deferred.** Plugins must `register()` cleanly with their
   heavy deps (`torch`, `ollama`, `qdrant_client`, …) absent. Imports happen
   inside node `execute()` paths, not at module top.
5. **One node class per file.** `nodes/<plugin>/<name>.py` contains exactly
   one node class — no helpers, no factories. Shared logic lives in
   `plugins/<plugin>/_*.py` (underscore-prefixed, not auto-discovered).

The contract for adding a panel kind, an RPC method, or a port-type is in
[FRONTEND_PROTOCOL.md](FRONTEND_PROTOCOL.md) and [FRONTEND_CHECKLIST.md](FRONTEND_CHECKLIST.md).

## Writing a plugin

Minimum viable plugin:

```python
# plugins/myplugin/__init__.py
from core.plugins import PluginContext

def register(ctx: PluginContext) -> None:
    from plugins.myplugin.port_types import register_all
    register_all()

    import nodes.myplugin as my_pkg
    ctx.discover_nodes(my_pkg)

    ctx.add_categories(["MyPlugin"])

    from plugins.myplugin._panel import build_panel_spec
    ctx.register_panel_spec("MyPlugin", build_panel_spec())
```

Drop the package into `plugins/`, restart, and:

- All discovered nodes appear in the palette under their declared categories.
- The panel spec auto-renders as a tab in DearPyGui (`gui/`) and React (`web/`)
  with no per-frontend code. Godot will inherit the same once its panel
  rendering lands.

If a panel needs rendering beyond the six standard `PanelSpec` section kinds
(`form`, `dynamic_form`, `status`, `plot`, `buttons`, `custom`), the frontend
provides an optional override hook (e.g. React's `web/src/panels/index.ts`
`PANEL_BUILDERS` registry). Use sparingly — if more than one frontend would
benefit, extend the spec instead.

Reference plugin: `plugins/pytorch/`. Lean reference for non-domain plugins:
`plugins/_example/`.

## Available plugins

| Plugin | Domain | Notes |
|---|---|---|
| `pytorch` | ML / deep learning | Layers, losses, optimizers, training loop, dataset loader. |
| `agents` | LLM-driven agents | Ollama + OpenAI-compat backends; Qdrant memory (Phase B); autoresearch (Phase C). See `plugins/agents/DESIGN.md`. |
| `robotics` | LeRobot, serial control | Teleop, episode recording, controller orchestration. |
| `numpy` / `pandas` / `scipy` / `sklearn` | Classical data science | Pure-Python tabular and array ops. |
| `io` | Filesystem, web, dataset I/O | CSV, JSON, image, dataset loaders. |
| `python` | Generic compute | `PythonNode` for inline expressions. |

## Frontends

| Frontend | Status | How to run |
|---|---|---|
| `gui/` (DearPyGui desktop) | Primary today | `python main.py` |
| `web/` (React + Vite) | Working | `python launch_web.py` then `cd web && npm run dev` |
| `godot/` (Godot) | Scaffold (panel rendering pending) | n/a |

All three speak the same backend (`server.py` JSON-RPC over WebSocket) and the
same `PanelSpec` schema. Adding a new GUI means rendering the six section
kinds; nothing else.

## Why this matters

The point of this architecture is that you can:

- Build a plugin once and have it work in every GUI we ship now and every GUI
  we (or anyone else) ship later.
- Build a new GUI once and have every plugin work in it.
- Mix and match — run the React frontend for one user while a coworker uses
  DearPyGui against the same `server.py`.

The dream: external contributors drop a plugin into `plugins/` and it Just
Works everywhere. Don't break that.
