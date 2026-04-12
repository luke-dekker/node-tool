"""Re-export shim — RNN/LSTM/GRU merged nodes are the source of truth.

Old per-layer class names kept as aliases for backward compatibility.
"""
from nodes.pytorch.rnn import RNNNode
from nodes.pytorch.lstm import LSTMNode
from nodes.pytorch.gru import GRUNode
from nodes.pytorch.pack_sequence import PackSequenceNode
from nodes.pytorch.unpack_sequence import UnpackSequenceNode

# Backward-compat aliases
RNNLayerNode = RNNNode
GRULayerNode = GRUNode
LSTMLayerNode = LSTMNode
RNNForwardNode = RNNNode
LSTMForwardNode = LSTMNode

__all__ = [
    "RNNNode", "LSTMNode", "GRUNode",
    "RNNLayerNode", "GRULayerNode", "LSTMLayerNode",
    "RNNForwardNode", "LSTMForwardNode",
    "PackSequenceNode", "UnpackSequenceNode",
]
