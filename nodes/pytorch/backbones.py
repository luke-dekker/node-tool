"""Re-export shim — PretrainedBackboneNode is the source of truth.

Old per-model class names kept as aliases for backward compatibility
with templates and tests that import them.
"""
from nodes.pytorch.pretrained_backbone import PretrainedBackboneNode
from nodes.pytorch.freeze_backbone import FreezeBackboneNode
from nodes.pytorch.model_info_persist import ModelInfoNode
from nodes.pytorch.freeze_named_layers import FreezeNamedLayersNode

# Backward-compat aliases — these are NOT registered as separate nodes,
# they just let old `from nodes.pytorch.backbones import ResNet18Node` work.
ResNet18Node = PretrainedBackboneNode
ResNet50Node = PretrainedBackboneNode
MobileNetV3Node = PretrainedBackboneNode
EfficientNetB0Node = PretrainedBackboneNode

__all__ = [
    "PretrainedBackboneNode",
    "ResNet18Node", "ResNet50Node", "MobileNetV3Node", "EfficientNetB0Node",
    "FreezeBackboneNode", "ModelInfoNode", "FreezeNamedLayersNode",
]
