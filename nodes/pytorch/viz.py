"""Re-export shim — TensorVizNode is the source of truth for tensor viz.

Old per-mode class names kept as aliases for backward compatibility.
"""
from nodes.pytorch.tensor_viz import TensorVizNode
from nodes.pytorch.viz_training_curve import PlotTrainingCurveNode
from nodes.pytorch.viz_weight_hist import WeightHistogramNode

# Backward-compat aliases
PlotTensorNode = TensorVizNode
TensorHistogramNode = TensorVizNode
TensorScatterNode = TensorVizNode
ShowImageNode = TensorVizNode

__all__ = [
    "TensorVizNode", "PlotTrainingCurveNode", "WeightHistogramNode",
    "PlotTensorNode", "TensorHistogramNode", "TensorScatterNode", "ShowImageNode",
]
