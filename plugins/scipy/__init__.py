"""Plugin for scipy nodes."""
from core.plugins import PluginContext

def register(ctx: PluginContext) -> None:
    import nodes.scipy as mod
    ctx.discover_nodes(mod)
    ctx.add_categories(["SciPy"])
