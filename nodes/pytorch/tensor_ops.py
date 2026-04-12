"""Re-export shim — consolidated tensor nodes are the source of truth.

Old per-op class names kept as aliases for backward compatibility.
"""
from nodes.pytorch.tensor_cat import TensorCatNode
from nodes.pytorch.tensor_stack import TensorStackNode
from nodes.pytorch.tensor_split import TensorSplitNode
from nodes.pytorch.tensor_shape_op import TensorShapeOpNode
from nodes.pytorch.tensor_create import TensorCreateNode
from nodes.pytorch.tensor_transpose import TensorTransposeNode
from nodes.pytorch.tensor_permute import TensorPermuteNode
from nodes.pytorch.tensor_einsum import TensorEinsumNode

# Backward-compat aliases
TensorReshapeNode = TensorShapeOpNode
TensorUnsqueezeNode = TensorShapeOpNode
TensorSqueezeNode = TensorShapeOpNode
RandTensorNode = TensorCreateNode
ZerosTensorNode = TensorCreateNode
OnesTensorNode = TensorCreateNode

__all__ = [
    "TensorCatNode", "TensorStackNode", "TensorSplitNode",
    "TensorShapeOpNode", "TensorCreateNode",
    "TensorTransposeNode", "TensorPermuteNode", "TensorEinsumNode",
    "TensorReshapeNode", "TensorUnsqueezeNode", "TensorSqueezeNode",
    "RandTensorNode", "ZerosTensorNode", "OnesTensorNode",
]
