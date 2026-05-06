"""Plugin for IO nodes."""
from core.plugins import PluginContext

def register(ctx: PluginContext) -> None:
    # nodes.io's __init__ already imports every IO node class, so a single
    # package-level discover pass picks them all up.
    from nodes import io as io_mod
    ctx.discover_nodes(io_mod)
    ctx.add_categories(["IO"])
