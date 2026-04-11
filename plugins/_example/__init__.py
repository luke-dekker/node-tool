"""Example plugin — copy this folder to create your own domain plugin.

A plugin is a Python package in the `plugins/` directory with a `register()`
function. The app discovers it at startup and calls register(ctx) where ctx
is a PluginContext with methods to add port types, nodes, panels, and
palette categories.

To create your own plugin:
    1. Copy this `_example/` folder to `plugins/your_domain/`
    2. Edit __init__.py to register your port types + discover your nodes
    3. Add node files to `plugins/your_domain/nodes/`
    4. Restart the app — your nodes appear in the palette automatically

This example is prefixed with `_` so it's skipped by the plugin loader
(underscore-prefixed dirs are ignored). Remove the underscore to activate.
"""
from core.plugins import PluginContext


def register(ctx: PluginContext) -> None:
    """Called once at app startup. Register everything your domain needs."""

    # ── 1. Port types ──────────────────────────────────────────────────────
    # Register domain-specific data types that flow between nodes.
    # editable=True → shows a widget on the canvas (like FLOAT, INT)
    # editable=False (default) → shows a text label, must be connected

    ctx.register_port_type(
        "EXAMPLE_DATA",
        default=None,
        color=(100, 255, 180, 255),   # pin color in the node editor
        pin_shape="triangle",         # circle, triangle, quad, + _filled variants
        description="Example custom data type",
        editable=False,               # complex type → connect only, no widget
    )

    # ── 2. Nodes ───────────────────────────────────────────────────────────
    # Discover all BaseNode subclasses in your nodes module.
    from plugins._example import nodes as my_nodes
    ctx.discover_nodes(my_nodes)

    # ── 3. Palette categories ──────────────────────────────────────────────
    # Add categories for the left-side palette. Nodes declare their own
    # category in the class body; this just sets the display order.
    ctx.add_categories(["Example"])

    # ── 4. Panels (optional) ──────────────────────────────────────────────
    # Register a sidebar panel tab. The builder receives the DPG parent tag
    # and the app instance. Uncomment to add one:
    #
    # def build_my_panel(parent_tag, app):
    #     import dearpygui.dearpygui as dpg
    #     dpg.add_text("Hello from my plugin!", parent=parent_tag)
    #     dpg.add_button(label="Do something", parent=parent_tag,
    #                    callback=lambda: app._log("[MyPlugin] Button clicked"))
    #
    # ctx.register_panel("My Domain", build_my_panel)
