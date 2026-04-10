"""Plugin discovery and registration system.

Each plugin is a Python package in the `plugins/` directory (or anywhere on
sys.path) with a `register()` function. At app startup, the plugin loader
scans the directory, imports each package, and calls `register(app_context)`.

The register function receives a PluginContext with methods to:
  - Register port types (forwarded to PortTypeRegistry)
  - Register node classes (added to NODE_REGISTRY)
  - Register panel builders (UI tabs in the right sidebar)
  - Register categories + their display order
  - Register templates

Plugin directory layout:
    plugins/
    ├── pytorch/              ← domain: ML / deep learning
    │   ├── __init__.py       ← def register(ctx): ...
    │   ├── port_types.py     ← optional, called by register()
    │   └── nodes/            ← node .py files
    ├── robotics/             ← domain: robotics
    │   ├── __init__.py
    │   └── nodes/
    └── audio/                ← domain: audio processing
        ├── __init__.py
        └── nodes/

A plugin's __init__.py:
    from core.plugins import PluginContext

    def register(ctx: PluginContext):
        # Register domain-specific port types
        ctx.register_port_type("TRAJECTORY", default=None,
                               color=(255, 200, 40, 255))

        # Register nodes (auto-discovered from a module)
        from plugins.robotics import nodes
        ctx.discover_nodes(nodes)

        # Register a sidebar panel
        ctx.register_panel("Simulation", build_simulation_panel)

        # Register categories for the palette
        ctx.add_categories(["Sensors", "Actuators", "Planning"])
"""
from __future__ import annotations
import importlib
import sys
from pathlib import Path
from typing import Any, Callable, Type

from core.port_types import PortTypeRegistry


# The plugins directory — sits alongside core/, gui/, nodes/
PLUGINS_DIR = Path(__file__).parent.parent / "plugins"


class PluginContext:
    """API surface that plugins use to register their functionality.

    Passed to each plugin's `register()` function. Wraps the global registries
    so plugins don't need to import them directly — cleaner dependency inversion
    and easier to mock in tests.
    """

    def __init__(self):
        self._panels: list[tuple[str, Callable]] = []
        self._categories: list[str] = []
        self._node_classes: list[Type] = []

    # ── Port types ──────────────────────────────────────────────────────────

    def register_port_type(self, name: str, **kwargs) -> str:
        """Register a domain-specific port type. See PortTypeRegistry.register."""
        return PortTypeRegistry.register(name, **kwargs)

    # ── Nodes ───────────────────────────────────────────────────────────────

    def register_node(self, cls: Type) -> None:
        """Register a single node class."""
        from core.node import BaseNode
        if hasattr(cls, "type_name") and cls.type_name != "base":
            self._node_classes.append(cls)

    def discover_nodes(self, module) -> int:
        """Auto-discover BaseNode subclasses in a module (same as nodes/__init__._discover)."""
        import inspect
        from core.node import BaseNode
        count = 0
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (issubclass(obj, BaseNode)
                    and obj is not BaseNode
                    and hasattr(obj, "type_name")
                    and obj.type_name != "base"):
                self._node_classes.append(obj)
                count += 1
        return count

    # ── Panels ──────────────────────────────────────────────────────────────

    def register_panel(self, label: str, builder: Callable) -> None:
        """Register a sidebar panel tab. The builder receives the DPG parent tag
        and the app instance: builder(parent_tag, app)."""
        self._panels.append((label, builder))

    # ── Categories ──────────────────────────────────────────────────────────

    def add_categories(self, names: list[str]) -> None:
        """Add palette category names (in display order)."""
        self._categories.extend(names)

    # ── Accessors (used by the app after all plugins are loaded) ────────────

    @property
    def panels(self) -> list[tuple[str, Callable]]:
        return list(self._panels)

    @property
    def categories(self) -> list[str]:
        return list(self._categories)

    @property
    def node_classes(self) -> list[Type]:
        return list(self._node_classes)


def discover_plugins() -> list[tuple[str, Any]]:
    """Scan PLUGINS_DIR for Python packages with a register() function.

    Returns a list of (plugin_name, module) tuples for successfully imported
    plugins. Errors are printed but don't stop the app.
    """
    PLUGINS_DIR.mkdir(parents=True, exist_ok=True)
    plugins: list[tuple[str, Any]] = []

    for path in sorted(PLUGINS_DIR.iterdir()):
        if not path.is_dir():
            continue
        init_file = path / "__init__.py"
        if not init_file.exists():
            continue

        mod_name = f"plugins.{path.name}"
        try:
            # Ensure the plugins parent is on sys.path
            plugins_parent = str(PLUGINS_DIR.parent)
            if plugins_parent not in sys.path:
                sys.path.insert(0, plugins_parent)

            module = importlib.import_module(mod_name)
            if hasattr(module, "register") and callable(module.register):
                plugins.append((path.name, module))
            else:
                print(f"[plugins] {path.name}: no register() function, skipped")
        except Exception as exc:
            print(f"[plugins] {path.name}: import failed: {exc}")

    return plugins


def load_plugins() -> PluginContext:
    """Discover and register all plugins. Returns the merged PluginContext.

    Call this once at app startup, after the core is initialized but before
    the GUI is built. The returned context contains all registered nodes,
    panels, and categories from all plugins.
    """
    ctx = PluginContext()
    discovered = discover_plugins()

    for name, module in discovered:
        try:
            module.register(ctx)
            n_nodes = len(ctx._node_classes)
            n_panels = len(ctx._panels)
            print(f"[plugins] {name}: registered ({n_nodes} nodes, "
                  f"{n_panels} panels, "
                  f"{len(PortTypeRegistry.all_types())} port types)")
        except Exception as exc:
            print(f"[plugins] {name}: register() failed: {exc}")

    return ctx
