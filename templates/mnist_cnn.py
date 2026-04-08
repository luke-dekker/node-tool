"""MNIST CNN classifier template.

A small convolutional pipeline: two Conv2d+ReLU+MaxPool blocks, flatten, dense head.
Demonstrates the conv->pool->flatten->linear pattern that any image classifier follows.
"""
from __future__ import annotations
from core.graph import Graph
from templates._helpers import grid


def build(graph: Graph) -> dict[str, tuple[int, int]]:
    from nodes.pytorch.mnist_dataset    import MNISTDatasetNode
    from nodes.pytorch.batch_input      import BatchInputNode
    from nodes.pytorch.conv2d           import Conv2dNode
    from nodes.pytorch.maxpool2d        import MaxPool2dNode
    from nodes.pytorch.flatten          import FlattenNode
    from nodes.pytorch.linear           import LinearNode
    from nodes.pytorch.training_config  import TrainingConfigNode

    pos = grid(step_x=220)
    positions: dict[str, tuple[int, int]] = {}

    mnist = MNISTDatasetNode()
    mnist.inputs["batch_size"].default_value = 64
    graph.add_node(mnist); positions[mnist.id] = pos()

    batch = BatchInputNode()
    graph.add_node(batch); positions[batch.id] = pos()

    # Block 1: 1 -> 16 channels
    conv1 = Conv2dNode()
    conv1.inputs["in_ch"].default_value      = 1
    conv1.inputs["out_ch"].default_value     = 16
    conv1.inputs["kernel"].default_value     = 3
    conv1.inputs["padding"].default_value    = 1
    conv1.inputs["activation"].default_value = "relu"
    graph.add_node(conv1); positions[conv1.id] = pos()

    pool1 = MaxPool2dNode()
    pool1.inputs["kernel"].default_value = 2
    pool1.inputs["stride"].default_value = 2
    graph.add_node(pool1); positions[pool1.id] = pos()

    # Block 2: 16 -> 32 channels
    conv2 = Conv2dNode()
    conv2.inputs["in_ch"].default_value      = 16
    conv2.inputs["out_ch"].default_value     = 32
    conv2.inputs["kernel"].default_value     = 3
    conv2.inputs["padding"].default_value    = 1
    conv2.inputs["activation"].default_value = "relu"
    graph.add_node(conv2); positions[conv2.id] = pos()

    pool2 = MaxPool2dNode()
    pool2.inputs["kernel"].default_value = 2
    pool2.inputs["stride"].default_value = 2
    graph.add_node(pool2); positions[pool2.id] = pos()

    flat = FlattenNode()
    graph.add_node(flat); positions[flat.id] = pos()

    # 32 channels * 7 * 7 = 1568 (28 -> 14 -> 7)
    head = LinearNode()
    head.inputs["in_features"].default_value  = 1568
    head.inputs["out_features"].default_value = 10
    head.inputs["activation"].default_value   = "none"
    graph.add_node(head); positions[head.id] = pos()

    cfg = TrainingConfigNode()
    cfg.inputs["epochs"].default_value    = 3
    cfg.inputs["lr"].default_value        = 0.001
    cfg.inputs["loss"].default_value      = "crossentropy"
    cfg.inputs["optimizer"].default_value = "adam"
    graph.add_node(cfg); positions[cfg.id] = pos()

    graph.add_connection(mnist.id, "dataloader", batch.id, "dataloader")
    graph.add_connection(mnist.id, "dataloader", cfg.id,   "dataloader")
    graph.add_connection(batch.id, "x",          conv1.id, "tensor_in")
    graph.add_connection(conv1.id, "tensor_out", pool1.id, "tensor_in")
    graph.add_connection(pool1.id, "tensor_out", conv2.id, "tensor_in")
    graph.add_connection(conv2.id, "tensor_out", pool2.id, "tensor_in")
    graph.add_connection(pool2.id, "tensor_out", flat.id,  "tensor_in")
    graph.add_connection(flat.id,  "tensor_out", head.id,  "tensor_in")
    graph.add_connection(head.id,  "tensor_out", cfg.id,   "tensor_in")
    return positions
