"""Plugin for IO nodes."""
from core.plugins import PluginContext

def register(ctx: PluginContext) -> None:
    from nodes import io as io_mod
    ctx.discover_nodes(io_mod)
    from nodes.io import serial_nodes, network_nodes, file_nodes
    ctx.discover_nodes(serial_nodes)
    ctx.discover_nodes(network_nodes)
    ctx.discover_nodes(file_nodes)
    ctx.add_categories(["IO"])
