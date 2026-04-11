"""Plugin for sklearn nodes."""
from core.plugins import PluginContext

def register(ctx: PluginContext) -> None:
    import nodes.sklearn as mod
    ctx.discover_nodes(mod)
