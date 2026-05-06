"""CSV Quick Look — the 'what's in this file' workflow.

Loads a CSV and runs the standard inspection chain: shape, info, describe, head,
plus pulls a single column out as a Series. Five-second exploration of any new
dataset before you commit to a pipeline.
"""
from __future__ import annotations
from core.graph import Graph
from templates._helpers import grid

LABEL = "CSV Quick Look"
DESCRIPTION = "Load a CSV and show shape, info, describe, head. The 'what's in this file' workflow."


def build(graph: Graph) -> dict[str, tuple[int, int]]:
    from nodes.pandas import (
        PdSourceNode, PdShapeNode, PdInfoNode, PdGetColumnNode,
    )

    pos = grid(step_x=240)
    positions: dict[str, tuple[int, int]] = {}

    csv = PdSourceNode()
    csv.inputs["kind"].default_value = "csv"
    csv.inputs["path"].default_value = "data.csv"
    graph.add_node(csv); positions[csv.id] = pos(col=0, row=1)

    shape = PdShapeNode()
    graph.add_node(shape); positions[shape.id] = pos(col=1, row=0)

    info = PdInfoNode()
    info.inputs["op"].default_value = "info"
    graph.add_node(info); positions[info.id] = pos(col=1, row=1)

    desc = PdInfoNode()
    desc.inputs["op"].default_value = "describe"
    graph.add_node(desc); positions[desc.id] = pos(col=1, row=2)

    head = PdInfoNode()
    head.inputs["op"].default_value = "head"
    head.inputs["n"].default_value  = 10
    graph.add_node(head); positions[head.id] = pos(col=1, row=3)

    col = PdGetColumnNode()
    col.inputs["column"].default_value = "col_0"
    graph.add_node(col); positions[col.id] = pos(col=2, row=1)

    graph.add_connection(csv.id, "df", shape.id, "df")
    graph.add_connection(csv.id, "df", info.id,  "df")
    graph.add_connection(csv.id, "df", desc.id,  "df")
    graph.add_connection(csv.id, "df", head.id,  "df")
    graph.add_connection(csv.id, "df", col.id,   "df")
    return positions
