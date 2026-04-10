"""Plugin for pandas nodes."""
from core.plugins import PluginContext

def register(ctx: PluginContext) -> None:
    import nodes.pandas as mod
    ctx.discover_nodes(mod)
    ctx.add_categories(["Pandas"])
