"""Two-table join + aggregate template.

Loads two CSVs, joins them on a shared key, filters rows by a column, then
groups and aggregates. Demonstrates the merge / filter / groupby trio that
covers ~80 percent of analytics workflows.
"""
from __future__ import annotations
from core.graph import Graph
from templates._helpers import grid


def build(graph: Graph) -> dict[str, tuple[int, int]]:
    from nodes.pandas.pd_from_csv     import PdFromCsvNode
    from nodes.pandas.pd_merge        import PdMergeNode
    from nodes.pandas.pd_filter_rows  import PdFilterRowsNode
    from nodes.pandas.pd_groupby      import PdGroupByNode
    from nodes.pandas.pd_describe     import PdDescribeNode

    pos = grid(step_x=240)
    positions: dict[str, tuple[int, int]] = {}

    left = PdFromCsvNode()
    left.inputs["path"].default_value = "left.csv"
    graph.add_node(left); positions[left.id] = pos(col=0, row=0)

    right = PdFromCsvNode()
    right.inputs["path"].default_value = "right.csv"
    graph.add_node(right); positions[right.id] = pos(col=0, row=1)

    merge = PdMergeNode()
    merge.inputs["on"].default_value  = "id"
    merge.inputs["how"].default_value = "inner"
    graph.add_node(merge); positions[merge.id] = pos(col=1, row=0)

    filt = PdFilterRowsNode()
    filt.inputs["column"].default_value = "value"
    filt.inputs["op"].default_value     = ">"
    filt.inputs["value"].default_value  = 0.0
    graph.add_node(filt); positions[filt.id] = pos(col=2, row=0)

    grp = PdGroupByNode()
    grp.inputs["by"].default_value  = "category"
    grp.inputs["agg"].default_value = "mean"
    graph.add_node(grp); positions[grp.id] = pos(col=3, row=0)

    desc = PdDescribeNode()
    graph.add_node(desc); positions[desc.id] = pos(col=4, row=0)

    graph.add_connection(left.id,  "df",     merge.id, "left")
    graph.add_connection(right.id, "df",     merge.id, "right")
    graph.add_connection(merge.id, "result", filt.id,  "df")
    graph.add_connection(filt.id,  "result", grp.id,   "df")
    graph.add_connection(grp.id,   "result", desc.id,  "df")
    return positions
