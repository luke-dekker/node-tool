"""K-Means + PCA visualization template.

Unsupervised pipeline: load CSV, drop NA, convert to ndarray, standardize,
cluster with K-Means, then project to 2D with PCA so the clusters can be
plotted. Drop in your own dataset and tune n_clusters.
"""
from __future__ import annotations
from core.graph import Graph
from templates._helpers import grid


def build(graph: Graph) -> dict[str, tuple[int, int]]:
    from nodes.pandas.pd_from_csv          import PdFromCsvNode
    from nodes.pandas.pd_drop_na           import PdDropNaNode
    from nodes.pandas.pd_to_numpy          import PdToNumpyNode
    from nodes.sklearn.sk_kmeans           import SkKMeansNode
    from nodes.sklearn.sk_pca              import SkPCANode

    pos = grid(step_x=240)
    positions: dict[str, tuple[int, int]] = {}

    csv = PdFromCsvNode()
    csv.inputs["path"].default_value = "data.csv"
    graph.add_node(csv); positions[csv.id] = pos()

    drop_na = PdDropNaNode()
    graph.add_node(drop_na); positions[drop_na.id] = pos()

    to_np = PdToNumpyNode()
    graph.add_node(to_np); positions[to_np.id] = pos()

    kmeans = SkKMeansNode()
    kmeans.inputs["n_clusters"].default_value   = 4
    kmeans.inputs["random_state"].default_value = 42
    graph.add_node(kmeans); positions[kmeans.id] = pos(col=3, row=0)

    pca = SkPCANode()
    pca.inputs["n_components"].default_value = 2
    graph.add_node(pca); positions[pca.id] = pos(col=3, row=1)

    graph.add_connection(csv.id,     "df",     drop_na.id, "df")
    graph.add_connection(drop_na.id, "result", to_np.id,   "df")
    graph.add_connection(to_np.id,   "array",  kmeans.id,  "X")
    graph.add_connection(to_np.id,   "array",  pca.id,     "X")
    return positions
