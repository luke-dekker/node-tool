"""PyTorch plugin — ML / deep learning nodes, training panel, port types.

This is the default domain plugin that ships with node-tool. It registers:
  - Port types: TENSOR, MODULE, DATALOADER, OPTIMIZER, LOSS_FN, SCHEDULER,
    DATASET, TRANSFORM
  - ~100 node classes: layers, losses, optimizers, datasets, training, viz
  - Training panel (sidebar tab with epochs/lr/optimizer/loss/device)
  - Palette categories: Datasets, Layers, Models, Training, Analyze

Other domains (robotics, audio, etc.) follow the same pattern: a plugins/
directory with register(ctx) that adds port types, nodes, and panels.
"""
from __future__ import annotations
from core.plugins import PluginContext


def register(ctx: PluginContext) -> None:
    """Register all PyTorch functionality."""

    # Port types first — nodes reference them during class-level setup.
    from plugins.pytorch.port_types import register_all as register_port_types
    register_port_types()

    # Discover all node classes from the existing nodes/pytorch/ package
    import nodes.pytorch as pt_pkg
    ctx.discover_nodes(pt_pkg)

    # Also discover from the sub-shim modules that pt.__init__ imports
    for attr_name in dir(pt_pkg):
        obj = getattr(pt_pkg, attr_name)
        if isinstance(obj, type):
            ctx.register_node(obj)

    # Categories this plugin adds to the palette
    ctx.add_categories([
        "Datasets", "Layers", "Models", "Training", "AI", "Analyze",
    ])

    # Training panel — declared once as a PanelSpec, rendered by every GUI.
    from plugins.pytorch._panel_training import build_training_panel_spec
    ctx.register_panel_spec("Training", build_training_panel_spec())
