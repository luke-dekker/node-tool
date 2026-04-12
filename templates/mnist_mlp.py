"""MNIST MLP classifier — the hello-world of deep learning.

Marker-based architecture: the graph contains zero data-loading nodes. An A
marker injects the image batch at training time, the MLP runs, and a B
marker marks the logits as the training target. All dataset config
(mnist/cifar/whatever path, batch_size, shuffle) lives in the Training
Panel.

    Data In (A:x) → Flatten → Linear+ReLU → Linear → Data Out (B:logits)
"""
from __future__ import annotations
from core.graph import Graph
from templates._helpers import grid

LABEL = "MNIST Classifier (MLP)"
DESCRIPTION = "Hello-world MLP on MNIST. Marker-based — dataset lives in the panel."


def build(graph: Graph) -> dict[str, tuple[int, int]]:
    from nodes.pytorch.input_marker   import InputMarkerNode
    from nodes.pytorch.flatten        import FlattenNode
    from nodes.pytorch.linear         import LinearNode
    from nodes.pytorch.train_marker   import TrainMarkerNode

    pos = grid(step_x=240)
    positions: dict[str, tuple[int, int]] = {}

    data_in = InputMarkerNode()
    data_in.inputs["modality"].default_value = "x"
    graph.add_node(data_in); positions[data_in.id] = pos()

    flat = FlattenNode()
    graph.add_node(flat); positions[flat.id] = pos()

    h1 = LinearNode()
    h1.inputs["in_features"].default_value  = 784
    h1.inputs["out_features"].default_value = 128
    h1.inputs["activation"].default_value   = "relu"
    graph.add_node(h1); positions[h1.id] = pos()

    h2 = LinearNode()
    h2.inputs["in_features"].default_value  = 128
    h2.inputs["out_features"].default_value = 10
    h2.inputs["activation"].default_value   = "none"
    graph.add_node(h2); positions[h2.id] = pos()

    data_out = TrainMarkerNode()
    data_out.inputs["kind"].default_value = "logits"
    data_out.inputs["target"].default_value = "label"
    graph.add_node(data_out); positions[data_out.id] = pos()

    graph.add_connection(data_in.id, "tensor",     flat.id,     "tensor_in")
    graph.add_connection(flat.id,    "tensor_out", h1.id,       "tensor_in")
    graph.add_connection(h1.id,      "tensor_out", h2.id,       "tensor_in")
    graph.add_connection(h2.id,      "tensor_out", data_out.id, "tensor_in")
    return positions
