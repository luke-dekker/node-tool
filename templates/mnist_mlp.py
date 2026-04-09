"""MNIST MLP classifier template.

The hello-world of deep learning. Loads MNIST, flattens 28x28 images to 784
vectors, runs a 2-layer MLP, and trains with cross-entropy loss.

Open the Training panel and click Start to train.
"""
from __future__ import annotations
from core.graph import Graph
from templates._helpers import grid

LABEL = "MNIST Classifier (MLP)"
DESCRIPTION = "Hello-world MLP on MNIST. Flatten -> Linear+ReLU -> Linear -> CrossEntropy."


def build(graph: Graph) -> dict[str, tuple[int, int]]:
    from nodes.pytorch.mnist_dataset    import MNISTDatasetNode
    from nodes.pytorch.batch_input      import BatchInputNode
    from nodes.pytorch.flatten          import FlattenNode
    from nodes.pytorch.linear           import LinearNode
    from nodes.pytorch.training_config  import TrainingConfigNode

    pos = grid(step_x=240)
    positions: dict[str, tuple[int, int]] = {}

    mnist = MNISTDatasetNode()
    mnist.inputs["batch_size"].default_value = 64
    mnist.inputs["train"].default_value      = True
    graph.add_node(mnist); positions[mnist.id] = pos()

    batch = BatchInputNode()
    graph.add_node(batch); positions[batch.id] = pos()

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

    cfg = TrainingConfigNode()
    cfg.inputs["epochs"].default_value    = 5
    cfg.inputs["device"].default_value    = "cpu"
    cfg.inputs["optimizer"].default_value = "adam"
    cfg.inputs["lr"].default_value        = 0.001
    cfg.inputs["loss"].default_value      = "crossentropy"
    graph.add_node(cfg); positions[cfg.id] = pos()

    # Wire the model: dataloader -> batch input -> flatten -> linear -> linear -> cfg
    # MNIST batches arrive as (x, y) tuples, so the BatchInput's `x` port is the entry point.
    graph.add_connection(mnist.id, "dataloader", batch.id, "dataloader")
    graph.add_connection(mnist.id, "dataloader", cfg.id,   "dataloader")
    graph.add_connection(batch.id, "x",          flat.id,  "tensor_in")
    graph.add_connection(flat.id,  "tensor_out", h1.id,    "tensor_in")
    graph.add_connection(h1.id,    "tensor_out", h2.id,    "tensor_in")
    graph.add_connection(h2.id,    "tensor_out", cfg.id,   "tensor_in")
    return positions
