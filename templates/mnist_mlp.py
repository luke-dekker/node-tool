"""MNIST MLP classifier — the hello-world of deep learning.

Loads MNIST, flattens 28x28 images to 784 vectors, runs a 2-layer MLP, and
trains with cross-entropy loss. Set epochs/lr/optimizer in the Training Panel
(right sidebar) and click Start.

New architecture: 4 nodes, zero legacy adapters, zero cross-canvas wires.

    MNIST Dataset ── x ──→ Flatten → Linear+ReLU → Linear ──→ Train Output
                  └─ label  (auto-discovered by panel for loss computation)
                  └─ dataloader  (auto-discovered by panel for batch iteration)
"""
from __future__ import annotations
from core.graph import Graph
from templates._helpers import grid

LABEL = "MNIST Classifier (MLP)"
DESCRIPTION = "Hello-world MLP on MNIST. 4 nodes, no legacy adapters."


def build(graph: Graph) -> dict[str, tuple[int, int]]:
    from nodes.pytorch.mnist_dataset  import MNISTDatasetNode
    from nodes.pytorch.flatten        import FlattenNode
    from nodes.pytorch.linear         import LinearNode
    from nodes.pytorch.train_output   import TrainOutputNode

    pos = grid(step_x=240)
    positions: dict[str, tuple[int, int]] = {}

    mnist = MNISTDatasetNode()
    mnist.inputs["batch_size"].default_value = 64
    graph.add_node(mnist); positions[mnist.id] = pos()

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

    target = TrainOutputNode()
    graph.add_node(target); positions[target.id] = pos()

    # Wire: dataset.x → flatten → linear → linear → train output
    graph.add_connection(mnist.id, "x",          flat.id,   "tensor_in")
    graph.add_connection(flat.id,  "tensor_out", h1.id,     "tensor_in")
    graph.add_connection(h1.id,    "tensor_out", h2.id,     "tensor_in")
    graph.add_connection(h2.id,    "tensor_out", target.id, "tensor_in")
    return positions
