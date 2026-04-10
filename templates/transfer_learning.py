"""Transfer Learning — pretrained ResNet18 with frozen backbone on CIFAR10."""
from __future__ import annotations
from core.graph import Graph
from templates._helpers import grid

LABEL = "Transfer Learning (ResNet18)"
DESCRIPTION = "Pretrained ResNet18, frozen backbone, new classification head on CIFAR-10."

def build(graph: Graph) -> dict[str, tuple[int, int]]:
    from nodes.pytorch.cifar10_dataset       import CIFAR10DatasetNode
    from nodes.pytorch.resnet18              import ResNet18Node
    from nodes.pytorch.freeze_named_layers   import FreezeNamedLayersNode
    from nodes.pytorch.apply_module          import ApplyModuleNode
    from nodes.pytorch.train_output          import TrainOutputNode

    pos = grid(step_x=240); positions = {}

    ds = CIFAR10DatasetNode(); ds.inputs["batch_size"].default_value = 32
    graph.add_node(ds); positions[ds.id] = pos(col=0, row=1)

    backbone = ResNet18Node(); backbone.inputs["pretrained"].default_value=True; backbone.inputs["num_classes"].default_value=10
    graph.add_node(backbone); positions[backbone.id] = pos(col=1, row=0)

    freeze = FreezeNamedLayersNode()
    freeze.inputs["names"].default_value = "conv1,bn1,layer1,layer2,layer3,layer4"
    freeze.inputs["freeze"].default_value = True
    graph.add_node(freeze); positions[freeze.id] = pos(col=2, row=0)

    apply_node = ApplyModuleNode(); graph.add_node(apply_node); positions[apply_node.id] = pos(col=2, row=1)
    target = TrainOutputNode(); graph.add_node(target); positions[target.id] = pos(col=3, row=1)

    graph.add_connection(backbone.id,"model",freeze.id,"model")
    graph.add_connection(freeze.id,"model",apply_node.id,"model")
    graph.add_connection(ds.id,"x",apply_node.id,"tensor_in")
    graph.add_connection(apply_node.id,"tensor_out",target.id,"tensor_in")
    return positions
