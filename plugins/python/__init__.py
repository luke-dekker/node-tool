"""Plugin for core Python nodes (math, logic, string, code, data)."""
from core.plugins import PluginContext

def register(ctx: PluginContext) -> None:
    from nodes import math, logic, string, code, data
    for mod in [math, logic, string, code, data]:
        ctx.discover_nodes(mod)
    ctx.add_categories(["Python"])
