"""Sklearn nodes — consolidated to 5 mega-nodes.

- SkModelNode        — algorithm: logistic_regression | random_forest | svc |
                                   gradient_boosting | linear_regression |
                                   kmeans | pca
- SkPreprocessorNode — split into SkScalerNode + SkEncoderNode (kept; different ports)
- SkPredictNode      — mode: predict | proba
- SkSplitNode        — mode: train_test | cross_val
- SkMetricsNode      — metric: accuracy | r2 | confusion_matrix

Old class names alias to the new consolidated nodes. Caller sets
`algorithm`/`mode`/`metric` on the instance to recover specific behavior.
"""
from nodes.sklearn.sk_model        import SkModelNode
from nodes.sklearn.sk_preprocessor import SkScalerNode, SkEncoderNode
from nodes.sklearn.sk_predict      import SkPredictNode
from nodes.sklearn.sk_split        import SkSplitNode
from nodes.sklearn.sk_metrics      import SkMetricsNode

# Back-compat — SkModelNode algorithms
SkClassifierNode       = SkModelNode    # algorithm="logistic_regression" (default)
SkLinearRegressionNode = SkModelNode    # algorithm="linear_regression"
SkKMeansNode           = SkModelNode    # algorithm="kmeans"
SkPCANode              = SkModelNode    # algorithm="pca"

# Back-compat — SkPredictNode modes
SkPredictProbaNode = SkPredictNode      # mode="proba"

# Back-compat — SkSplitNode modes
SkTrainTestSplitNode = SkSplitNode      # mode="train_test" (default)
SkCrossValScoreNode  = SkSplitNode      # mode="cross_val"

# Back-compat — SkMetricsNode metrics (set elsewhere)
SkAccuracyNode        = SkMetricsNode
SkR2ScoreNode         = SkMetricsNode
SkConfusionMatrixNode = SkMetricsNode

__all__ = [
    "SkModelNode", "SkScalerNode", "SkEncoderNode",
    "SkPredictNode", "SkSplitNode", "SkMetricsNode",
    "SkClassifierNode", "SkLinearRegressionNode", "SkKMeansNode", "SkPCANode",
    "SkPredictProbaNode",
    "SkTrainTestSplitNode", "SkCrossValScoreNode",
    "SkAccuracyNode", "SkR2ScoreNode", "SkConfusionMatrixNode",
]
