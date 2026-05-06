"""Re-export shim — LayerNode is the consolidated source of truth.

LayerNode absorbs 13 single-in single-out layer types via `kind` dropdown
(linear, conv2d, batchnorm{1d,2d}, layernorm, dropout, embedding, activation,
positional_encoding, transformer_encoder, max_pool2d, avg_pool2d,
adaptive_avg_pool2d). Old class names below are aliases — same class, set
`kind` on the instance to recover specific behavior.

FlattenNode stays standalone — its semantics aren't an nn.Module wrapper.
"""
from nodes.pytorch.flatten import FlattenNode
from nodes.pytorch.layer   import LayerNode

# Back-compat aliases — same class. Caller sets `kind` to dispatch.
LinearNode                  = LayerNode
Conv2dNode                  = LayerNode
DropoutNode                 = LayerNode
BatchNorm1dNode             = LayerNode
BatchNorm2dNode             = LayerNode
LayerNormNode               = LayerNode
EmbeddingNode               = LayerNode
ActivationNode              = LayerNode
PositionalEncodingNode      = LayerNode
TransformerEncoderLayerNode = LayerNode
Pool2dNode                  = LayerNode
MaxPool2dNode               = LayerNode
AvgPool2dNode               = LayerNode
AdaptiveAvgPool2dNode       = LayerNode

__all__ = [
    "FlattenNode", "LayerNode",
    "LinearNode", "Conv2dNode", "DropoutNode",
    "BatchNorm1dNode", "BatchNorm2dNode", "LayerNormNode",
    "EmbeddingNode", "ActivationNode",
    "PositionalEncodingNode", "TransformerEncoderLayerNode",
    "Pool2dNode", "MaxPool2dNode", "AvgPool2dNode", "AdaptiveAvgPool2dNode",
]
