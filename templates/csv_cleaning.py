"""CSV cleaning ETL pipeline template.

Realistic data-cleaning chain: load -> drop NA rows -> fill remaining NA with 0
-> normalize columns -> sort by a column -> select a subset of columns. Each
node represents one explicit transform; you can rewire or insert new steps
without touching anything else.
"""
from __future__ import annotations
from core.graph import Graph
from templates._helpers import grid


def build(graph: Graph) -> dict[str, tuple[int, int]]:
    from nodes.pandas.pd_from_csv     import PdFromCsvNode
    from nodes.pandas.pd_drop_na      import PdDropNaNode
    from nodes.pandas.pd_fill_na      import PdFillNaNode
    from nodes.pandas.pd_normalize    import PdNormalizeNode
    from nodes.pandas.pd_sort         import PdSortNode
    from nodes.pandas.pd_select_cols  import PdSelectColsNode
    from nodes.pandas.pd_describe     import PdDescribeNode

    pos = grid(step_x=220)
    positions: dict[str, tuple[int, int]] = {}

    csv = PdFromCsvNode()
    csv.inputs["path"].default_value = "data.csv"
    graph.add_node(csv); positions[csv.id] = pos()

    drop_na = PdDropNaNode()
    graph.add_node(drop_na); positions[drop_na.id] = pos()

    fill_na = PdFillNaNode()
    fill_na.inputs["value"].default_value = 0.0
    graph.add_node(fill_na); positions[fill_na.id] = pos()

    norm = PdNormalizeNode()
    graph.add_node(norm); positions[norm.id] = pos()

    sort = PdSortNode()
    sort.inputs["by"].default_value = "col_0"
    sort.inputs["ascending"].default_value = True
    graph.add_node(sort); positions[sort.id] = pos()

    select = PdSelectColsNode()
    select.inputs["columns"].default_value = "col_0,col_1,col_2"
    graph.add_node(select); positions[select.id] = pos()

    desc = PdDescribeNode()
    graph.add_node(desc); positions[desc.id] = pos()

    graph.add_connection(csv.id,     "df",     drop_na.id, "df")
    graph.add_connection(drop_na.id, "result", fill_na.id, "df")
    graph.add_connection(fill_na.id, "result", norm.id,    "df")
    graph.add_connection(norm.id,    "result", sort.id,    "df")
    graph.add_connection(sort.id,    "result", select.id,  "df")
    graph.add_connection(select.id,  "result", desc.id,    "df")
    return positions
