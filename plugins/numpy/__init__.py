"""Plugin for numpy nodes."""
from core.plugins import PluginContext

def register(ctx: PluginContext) -> None:
    import nodes.numpy as mod
    ctx.discover_nodes(mod)
    ctx.add_categories(["NumPy"])
