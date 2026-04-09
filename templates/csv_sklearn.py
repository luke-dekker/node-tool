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
    from nodes.pandas.pd_from_csv             import PdFromCsvNode
    from nodes.pandas.pd_drop_na              import PdDropNaNode
    from nodes.pandas.pd_xy_split             import PdXYSplitNode
    from nodes.sklearn.sk_train_test_split    import SkTrainTestSplitNode
    from nodes.sklearn.sk_standard_scaler     import SkStandardScalerNode
    from nodes.sklearn.sk_linear_regression   import SkLinearRegressionNode
    from nodes.sklearn.sk_predict             import SkPredictNode
    from nodes.sklearn.sk_r2_score            import SkR2ScoreNode

    pos = grid(step_x=220)
    positions: dict[str, tuple[int, int]] = {}

    csv = PdFromCsvNode()
    csv.inputs["path"].default_value = "data.csv"
    graph.add_node(csv); positions[csv.id] = pos()

    drop_na = PdDropNaNode()
    graph.add_node(drop_na); positions[drop_na.id] = pos()

    xy = PdXYSplitNode()
    xy.inputs["label_col"].default_value = "target"
    graph.add_node(xy); positions[xy.id] = pos()

    split = SkTrainTestSplitNode()
    split.inputs["test_size"].default_value    = 0.2
    split.inputs["random_state"].default_value = 42
    graph.add_node(split); positions[split.id] = pos()

    scaler = SkStandardScalerNode()
    graph.add_node(scaler); positions[scaler.id] = pos()

    lr = SkLinearRegressionNode()
    graph.add_node(lr); positions[lr.id] = pos()

    pred = SkPredictNode()
    graph.add_node(pred); positions[pred.id] = pos()

    r2 = SkR2ScoreNode()
    graph.add_node(r2); positions[r2.id] = pos()

    graph.add_connection(csv.id,     "df",             drop_na.id, "df")
    graph.add_connection(drop_na.id, "result",         xy.id,      "df")
    graph.add_connection(xy.id,      "X",              split.id,   "X")
    graph.add_connection(xy.id,      "y",              split.id,   "y")
    graph.add_connection(split.id,   "X_train",        scaler.id,  "X_train")
    graph.add_connection(split.id,   "X_test",         scaler.id,  "X_test")
    graph.add_connection(scaler.id,  "X_train_scaled", lr.id,      "X_train")
    graph.add_connection(split.id,   "y_train",        lr.id,      "y_train")
    graph.add_connection(lr.id,      "model",          pred.id,    "model")
    graph.add_connection(scaler.id,  "X_test_scaled",  pred.id,    "X")
    graph.add_connection(split.id,   "y_test",         r2.id,      "y_true")
    graph.add_connection(pred.id,    "predictions",    r2.id,      "y_pred")
    return positions
