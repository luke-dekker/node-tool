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
    from nodes.pandas.pd_from_csv    import PdFromCsvNode
    from nodes.pandas.pd_shape       import PdShapeNode
    from nodes.pandas.pd_info        import PdInfoNode
    from nodes.pandas.pd_describe    import PdDescribeNode
    from nodes.pandas.pd_head        import PdHeadNode
    from nodes.pandas.pd_get_column  import PdGetColumnNode

    pos = grid(step_x=240)
    positions: dict[str, tuple[int, int]] = {}

    csv = PdFromCsvNode()
    csv.inputs["path"].default_value = "data.csv"
    graph.add_node(csv); positions[csv.id] = pos(col=0, row=1)

    shape = PdShapeNode()
    graph.add_node(shape); positions[shape.id] = pos(col=1, row=0)

    info = PdInfoNode()
    graph.add_node(info); positions[info.id] = pos(col=1, row=1)

    desc = PdDescribeNode()
    graph.add_node(desc); positions[desc.id] = pos(col=1, row=2)

    head = PdHeadNode()
    head.inputs["n"].default_value = 10
    graph.add_node(head); positions[head.id] = pos(col=1, row=3)

    col = PdGetColumnNode()
    col.inputs["column"].default_value = "col_0"
    graph.add_node(col); positions[col.id] = pos(col=2, row=1)

    # Fan out the loaded DataFrame to every inspection node
    graph.add_connection(csv.id, "df", shape.id, "df")
    graph.add_connection(csv.id, "df", info.id,  "df")
    graph.add_connection(csv.id, "df", desc.id,  "df")
    graph.add_connection(csv.id, "df", head.id,  "df")
    graph.add_connection(csv.id, "df", col.id,   "df")
    return positions
