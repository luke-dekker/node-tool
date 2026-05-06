"""Re-export shim — RecurrentLayerNode is the consolidated source of truth.

RNN/LSTM/GRU all share the same input/output shape pattern (varying only
in whether `cell` is populated), so they collapse into one mode-dispatched
node. Old class names below are aliases — set `kind` on the instance.
"""
from nodes.pytorch.recurrent_layer import RecurrentLayerNode
from nodes.pytorch.pack_sequence   import PackSequenceNode
from nodes.pytorch.unpack_sequence import UnpackSequenceNode

# Back-compat aliases — same class.
RNNNode         = RecurrentLayerNode
LSTMNode        = RecurrentLayerNode
GRUNode         = RecurrentLayerNode
RNNLayerNode    = RecurrentLayerNode
GRULayerNode    = RecurrentLayerNode
LSTMLayerNode   = RecurrentLayerNode
RNNForwardNode  = RecurrentLayerNode
LSTMForwardNode = RecurrentLayerNode

__all__ = [
    "RecurrentLayerNode",
    "RNNNode", "LSTMNode", "GRUNode",
    "RNNLayerNode", "GRULayerNode", "LSTMLayerNode",
    "RNNForwardNode", "LSTMForwardNode",
    "PackSequenceNode", "UnpackSequenceNode",
]
