"""Example nodes — demonstrates the node definition pattern.

Every node file exports one or more BaseNode subclasses. The plugin's
register() calls ctx.discover_nodes(this_module) to find them automatically.
"""
from plugins._example.nodes.example_node import ExampleNode, ExampleProcessorNode
