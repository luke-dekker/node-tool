"""Re-export shim — consolidated tensor nodes are the source of truth.

The 13 per-op tensor nodes collapsed into 5 (with PrintTensor as separate
side-effect logger):
  TensorCreateNode  — fill: zeros|ones|rand|from_list
  TensorOpNode      — op: add|sub|mul|div|argmax|softmax|einsum|mux
  TensorReshapeNode — op: cat|stack|split|reshape|squeeze|unsqueeze|permute|transpose
  TensorInfoNode    — mode: shape|full
  PrintTensorNode   — terminal logger (kept; side effect)

Old per-op class names below alias to the new consolidated class. Caller
sets `op`/`fill` on the instance to recover specific behavior.
"""
from nodes.pytorch.tensor_create  import TensorCreateNode
from nodes.pytorch.tensor_op      import TensorOpNode
from nodes.pytorch.tensor_reshape import TensorReshapeNode
from nodes.pytorch.tensor_info    import TensorInfoNode
from nodes.pytorch.print_tensor   import PrintTensorNode

# Back-compat aliases — TensorOpNode
TensorBinaryOpNode  = TensorOpNode
TensorAddNode       = TensorOpNode
TensorMulNode       = TensorOpNode
ArgmaxNode          = TensorOpNode
SoftmaxOpNode       = TensorOpNode
TensorEinsumNode    = TensorOpNode
TensorMuxNode       = TensorOpNode

# Back-compat aliases — TensorReshapeNode
TensorCatNode       = TensorReshapeNode
TensorStackNode     = TensorReshapeNode
TensorSplitNode     = TensorReshapeNode
TensorShapeOpNode   = TensorReshapeNode
TensorUnsqueezeNode = TensorReshapeNode
TensorSqueezeNode   = TensorReshapeNode
TensorTransposeNode = TensorReshapeNode
TensorPermuteNode   = TensorReshapeNode

# Back-compat aliases — TensorCreateNode
TensorFromListNode  = TensorCreateNode
RandTensorNode      = TensorCreateNode
ZerosTensorNode     = TensorCreateNode
OnesTensorNode      = TensorCreateNode

__all__ = [
    "TensorCreateNode", "TensorOpNode", "TensorReshapeNode",
    "TensorInfoNode", "PrintTensorNode",
    "TensorBinaryOpNode", "TensorAddNode", "TensorMulNode",
    "ArgmaxNode", "SoftmaxOpNode", "TensorEinsumNode", "TensorMuxNode",
    "TensorCatNode", "TensorStackNode", "TensorSplitNode",
    "TensorShapeOpNode", "TensorUnsqueezeNode", "TensorSqueezeNode",
    "TensorTransposeNode", "TensorPermuteNode",
    "TensorFromListNode", "RandTensorNode", "ZerosTensorNode", "OnesTensorNode",
]
