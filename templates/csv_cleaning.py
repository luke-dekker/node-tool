"""CSV cleaning ETL pipeline template.

Realistic data-cleaning chain: load -> drop NA rows -> fill remaining NA with 0
-> normalize columns -> sort by a column -> select a subset of columns. Each
node represents one explicit transform; you can rewire or insert new steps
without touching anything else.
"""
from __future__ import annotations
from core.graph import Graph
from templates._helpers import grid

LABEL = "CSV Cleaning Pipeline"
DESCRIPTION = "Real ETL: load -> drop NA -> fill NA -> normalize -> sort -> select cols."


def build(graph: Graph) -> dict[str, tuple[int, int]]:
    from nodes.pandas import PdSourceNode, PdTransformNode, PdInfoNode

    pos = grid(step_x=220)
    positions: dict[str, tuple[int, int]] = {}

    csv = PdSourceNode()
    csv.inputs["kind"].default_value = "csv"
    csv.inputs["path"].default_value = "data.csv"
    graph.add_node(csv); positions[csv.id] = pos()

    drop_na = PdTransformNode()
    drop_na.inputs["op"].default_value = "dropna"
    graph.add_node(drop_na); positions[drop_na.id] = pos()

    fill_na = PdTransformNode()
    fill_na.inputs["op"].default_value    = "fillna"
    fill_na.inputs["value"].default_value = 0.0
    graph.add_node(fill_na); positions[fill_na.id] = pos()

    norm = PdTransformNode()
    norm.inputs["op"].default_value = "normalize"
    graph.add_node(norm); positions[norm.id] = pos()

    sort = PdTransformNode()
    sort.inputs["op"].default_value        = "sort"
    sort.inputs["by"].default_value        = "col_0"
    sort.inputs["ascending"].default_value = True
    graph.add_node(sort); positions[sort.id] = pos()

    select = PdTransformNode()
    select.inputs["op"].default_value      = "select_cols"
    select.inputs["columns"].default_value = "col_0,col_1,col_2"
    graph.add_node(select); positions[select.id] = pos()

    desc = PdInfoNode()
    desc.inputs["op"].default_value = "describe"
    graph.add_node(desc); positions[desc.id] = pos()

    graph.add_connection(csv.id,     "df",     drop_na.id, "df")
    graph.add_connection(drop_na.id, "result", fill_na.id, "df")
    graph.add_connection(fill_na.id, "result", norm.id,    "df")
    graph.add_connection(norm.id,    "result", sort.id,    "df")
    graph.add_connection(sort.id,    "result", select.id,  "df")
    graph.add_connection(select.id,  "result", desc.id,    "df")
    return positions
