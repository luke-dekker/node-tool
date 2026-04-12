"""Transfer Learning — pretrained ResNet18 with frozen backbone on CIFAR10.

Marker-based architecture: the graph contains zero data-loading nodes. An A
marker injects the image batch at training time and a B marker marks the
logits as the training target. All dataset config lives in the Training Panel.

    Data In (A:x) → ApplyModule(ResNet18/frozen) → Data Out (B:logits)
"""
from __future__ import annotations
from core.graph import Graph
from templates._helpers import grid

LABEL = "Transfer Learning (ResNet18)"
DESCRIPTION = "Pretrained ResNet18, frozen backbone, new classification head. Marker-based — dataset lives in the panel."

def build(graph: Graph) -> dict[str, tuple[int, int]]:
    from nodes.pytorch.input_marker          import InputMarkerNode
    from nodes.pytorch.pretrained_backbone   import PretrainedBackboneNode
    from nodes.pytorch.freeze_named_layers   import FreezeNamedLayersNode
    from nodes.pytorch.apply_module          import ApplyModuleNode
    from nodes.pytorch.train_marker          import TrainMarkerNode

    pos = grid(step_x=240); positions = {}

    data_in = InputMarkerNode(); data_in.inputs["modality"].default_value = "x"
    graph.add_node(data_in); positions[data_in.id] = pos(col=0, row=1)

    backbone = PretrainedBackboneNode(); backbone.inputs["architecture"].default_value="resnet18"; backbone.inputs["pretrained"].default_value=True; backbone.inputs["num_classes"].default_value=10
    graph.add_node(backbone); positions[backbone.id] = pos(col=1, row=0)

    freeze = FreezeNamedLayersNode()
    freeze.inputs["names"].default_value = "conv1,bn1,layer1,layer2,layer3,layer4"
    freeze.inputs["freeze"].default_value = True
    graph.add_node(freeze); positions[freeze.id] = pos(col=2, row=0)

    apply_node = ApplyModuleNode(); graph.add_node(apply_node); positions[apply_node.id] = pos(col=2, row=1)
    data_out = TrainMarkerNode(); data_out.inputs["kind"].default_value="logits"; data_out.inputs["target"].default_value="label"
    graph.add_node(data_out); positions[data_out.id] = pos(col=3, row=1)

    graph.add_connection(backbone.id,"model",freeze.id,"model")
    graph.add_connection(freeze.id,"model",apply_node.id,"model")
    graph.add_connection(data_in.id,"tensor",apply_node.id,"tensor_in")
    graph.add_connection(apply_node.id,"tensor_out",data_out.id,"tensor_in")
    return positions
