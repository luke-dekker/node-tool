"""Transfer Learning template — pretrained ResNet18 with frozen backbone.

Loads ResNet18 with ImageNet weights, freezes the backbone, then runs CIFAR-10
images through it via an Apply Module bridge into the training config. The
freeze node outputs a copy of the model with requires_grad=False on every
parameter except the new fc head, so only the head trains.

The Apply Module node is the missing piece: backbones output a MODULE, not a
tensor flow, so they need a bridge to slot into the tensor_in/tensor_out chain
that TrainingConfigNode expects.
"""
from __future__ import annotations
from core.graph import Graph
from templates._helpers import grid

LABEL = "Transfer Learning (ResNet18)"
DESCRIPTION = "Pretrained ResNet18, frozen backbone, new classification head on CIFAR-10."


def build(graph: Graph) -> dict[str, tuple[int, int]]:
    from nodes.pytorch.cifar10_dataset       import CIFAR10DatasetNode
    from nodes.pytorch.batch_input           import BatchInputNode
    from nodes.pytorch.resnet18              import ResNet18Node
    from nodes.pytorch.freeze_named_layers   import FreezeNamedLayersNode
    from nodes.pytorch.apply_module          import ApplyModuleNode
    from nodes.pytorch.training_config       import TrainingConfigNode

    pos = grid(step_x=240)
    positions: dict[str, tuple[int, int]] = {}

    ds = CIFAR10DatasetNode()
    ds.inputs["batch_size"].default_value = 32
    graph.add_node(ds); positions[ds.id] = pos(col=0, row=1)

    batch = BatchInputNode()
    graph.add_node(batch); positions[batch.id] = pos(col=1, row=1)

    backbone = ResNet18Node()
    backbone.inputs["pretrained"].default_value  = True
    backbone.inputs["num_classes"].default_value = 10  # CIFAR-10
    graph.add_node(backbone); positions[backbone.id] = pos(col=2, row=0)

    # Freeze the backbone but leave the new fc head trainable. ResNet's
    # children are: conv1, bn1, relu, maxpool, layer1..layer4, avgpool, fc.
    # Freezing everything except `fc` is the canonical transfer-learning pattern.
    freeze = FreezeNamedLayersNode()
    freeze.inputs["names"].default_value  = "conv1,bn1,layer1,layer2,layer3,layer4"
    freeze.inputs["freeze"].default_value = True
    graph.add_node(freeze); positions[freeze.id] = pos(col=3, row=0)

    apply_node = ApplyModuleNode()
    graph.add_node(apply_node); positions[apply_node.id] = pos(col=3, row=1)

    cfg = TrainingConfigNode()
    cfg.inputs["epochs"].default_value    = 5
    cfg.inputs["lr"].default_value        = 0.001
    cfg.inputs["loss"].default_value      = "crossentropy"
    cfg.inputs["optimizer"].default_value = "adam"
    graph.add_node(cfg); positions[cfg.id] = pos(col=4, row=0)

    # Wire it: dataloader -> batch input -> backbone -> apply -> training config
    graph.add_connection(ds.id,         "dataloader", batch.id,      "dataloader")
    graph.add_connection(ds.id,         "dataloader", cfg.id,        "dataloader")
    graph.add_connection(backbone.id,   "model",      freeze.id,     "model")
    graph.add_connection(freeze.id,     "model",      apply_node.id, "model")
    graph.add_connection(batch.id,      "x",          apply_node.id, "tensor_in")
    graph.add_connection(apply_node.id, "tensor_out", cfg.id,        "tensor_in")
    return positions
