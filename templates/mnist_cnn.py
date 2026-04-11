"""MNIST CNN — convolutional pipeline. Conv → Pool → Conv → Pool → Flatten → Linear."""
from __future__ import annotations
from core.graph import Graph
from templates._helpers import grid

LABEL = "MNIST Classifier (CNN)"
DESCRIPTION = "Convolutional pipeline on MNIST. Conv -> Pool -> Conv -> Pool -> Linear."

def build(graph: Graph) -> dict[str, tuple[int, int]]:
    from nodes.pytorch.dataset       import DatasetNode
    from nodes.pytorch.conv2d        import Conv2dNode
    from nodes.pytorch.maxpool2d     import MaxPool2dNode
    from nodes.pytorch.flatten       import FlattenNode
    from nodes.pytorch.linear        import LinearNode
    from nodes.pytorch.train_output  import TrainOutputNode

    pos = grid(step_x=200); positions = {}
    mnist = DatasetNode(); mnist.inputs["path"].default_value = "mnist"; mnist.inputs["batch_size"].default_value = 64
    graph.add_node(mnist); positions[mnist.id] = pos()
    c1 = Conv2dNode(); c1.inputs["in_ch"].default_value=1; c1.inputs["out_ch"].default_value=16; c1.inputs["kernel"].default_value=3; c1.inputs["padding"].default_value=1; c1.inputs["activation"].default_value="relu"
    graph.add_node(c1); positions[c1.id] = pos()
    p1 = MaxPool2dNode(); graph.add_node(p1); positions[p1.id] = pos()
    c2 = Conv2dNode(); c2.inputs["in_ch"].default_value=16; c2.inputs["out_ch"].default_value=32; c2.inputs["kernel"].default_value=3; c2.inputs["padding"].default_value=1; c2.inputs["activation"].default_value="relu"
    graph.add_node(c2); positions[c2.id] = pos()
    p2 = MaxPool2dNode(); graph.add_node(p2); positions[p2.id] = pos()
    flat = FlattenNode(); graph.add_node(flat); positions[flat.id] = pos()
    head = LinearNode(); head.inputs["in_features"].default_value=1568; head.inputs["out_features"].default_value=10
    graph.add_node(head); positions[head.id] = pos()
    target = TrainOutputNode(); graph.add_node(target); positions[target.id] = pos()
    graph.add_connection(mnist.id,"x",c1.id,"tensor_in"); graph.add_connection(c1.id,"tensor_out",p1.id,"tensor_in")
    graph.add_connection(p1.id,"tensor_out",c2.id,"tensor_in"); graph.add_connection(c2.id,"tensor_out",p2.id,"tensor_in")
    graph.add_connection(p2.id,"tensor_out",flat.id,"tensor_in"); graph.add_connection(flat.id,"tensor_out",head.id,"tensor_in")
    graph.add_connection(head.id,"tensor_out",target.id,"tensor_in")
    return positions
