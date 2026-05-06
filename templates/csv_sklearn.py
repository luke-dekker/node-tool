"""CSV -> sklearn linear regression template.

End-to-end tabular ML in nine nodes: load CSV, drop missing rows, split into
features/target, train/test split, scale features, fit linear regression on
train, predict on test, score with R^2.
"""
from __future__ import annotations
from core.graph import Graph
from templates._helpers import grid

LABEL = "CSV -> Sklearn Regression"
DESCRIPTION = "End-to-end tabular ML. Load -> clean -> split -> scale -> fit -> predict -> R2."


def build(graph: Graph) -> dict[str, tuple[int, int]]:
    from nodes.pandas import PdSourceNode, PdTransformNode, PdXYSplitNode
    from nodes.sklearn.sk_split               import SkSplitNode
    from nodes.sklearn.sk_preprocessor        import SkScalerNode
    from nodes.sklearn.sk_model               import SkModelNode
    from nodes.sklearn.sk_predict             import SkPredictNode
    from nodes.sklearn.sk_metrics             import SkMetricsNode

    pos = grid(step_x=220)
    positions: dict[str, tuple[int, int]] = {}

    csv = PdSourceNode()
    csv.inputs["kind"].default_value = "csv"
    csv.inputs["path"].default_value = "data.csv"
    graph.add_node(csv); positions[csv.id] = pos()

    drop_na = PdTransformNode()
    drop_na.inputs["op"].default_value = "dropna"
    graph.add_node(drop_na); positions[drop_na.id] = pos()

    xy = PdXYSplitNode()
    xy.inputs["op"].default_value        = "xy_split"
    xy.inputs["label_col"].default_value = "target"
    graph.add_node(xy); positions[xy.id] = pos()

    split = SkSplitNode()
    split.inputs["mode"].default_value         = "train_test"
    split.inputs["test_size"].default_value    = 0.2
    split.inputs["random_state"].default_value = 42
    graph.add_node(split); positions[split.id] = pos()

    scaler = SkScalerNode()
    graph.add_node(scaler); positions[scaler.id] = pos()

    lr = SkModelNode()
    lr.inputs["algorithm"].default_value = "linear_regression"
    graph.add_node(lr); positions[lr.id] = pos()

    pred = SkPredictNode()
    graph.add_node(pred); positions[pred.id] = pos()

    r2 = SkMetricsNode()
    r2.inputs["metric"].default_value = "r2"
    graph.add_node(r2); positions[r2.id] = pos()

    graph.add_connection(csv.id,     "df",             drop_na.id, "df")
    graph.add_connection(drop_na.id, "result",         xy.id,      "df")
    # PdXYSplitNode aliases PdTransformNode: X→`result`, y→`series`.
    graph.add_connection(xy.id,      "result",         split.id,   "X")
    graph.add_connection(xy.id,      "series",         split.id,   "y")
    graph.add_connection(split.id,   "X_train",        scaler.id,  "X_train")
    graph.add_connection(split.id,   "X_test",         scaler.id,  "X_test")
    graph.add_connection(scaler.id,  "X_train_scaled", lr.id,      "X_train")
    graph.add_connection(split.id,   "y_train",        lr.id,      "y_train")
    graph.add_connection(lr.id,      "model",          pred.id,    "model")
    graph.add_connection(scaler.id,  "X_test_scaled",  pred.id,    "X")
    graph.add_connection(split.id,   "y_test",         r2.id,      "y_true")
    graph.add_connection(pred.id,    "predictions",    r2.id,      "y_pred")
    return positions
