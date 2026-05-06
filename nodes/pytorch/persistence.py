"""Re-export shim — ModelIONode is the consolidated source of truth.

Save / Load weights / Load full model / Export ONNX all dispatch through
mode on a single ModelIONode. Old class names alias to it.
"""
from nodes.pytorch.model_io import ModelIONode
from nodes.pytorch.model_info_persist import ModelInfoNode

# Back-compat aliases — same class. Caller sets `mode` to dispatch.
SaveWeightsNode     = ModelIONode    # mode="save_weights" (default)
SaveCheckpointNode  = ModelIONode    # mode="save_checkpoint"
SaveFullModelNode   = ModelIONode    # mode="save_full"
ExportONNXNode      = ModelIONode    # mode="export_onnx"
LoadWeightsNode     = ModelIONode    # mode="load_into"
LoadCheckpointNode  = ModelIONode    # mode="load_checkpoint"
LoadModelNode       = ModelIONode    # mode="load_full"
PretrainedBlockNode = ModelIONode    # mode="load_full" + eval_mode=True

__all__ = [
    "ModelIONode", "ModelInfoNode",
    "SaveWeightsNode", "SaveCheckpointNode", "SaveFullModelNode", "ExportONNXNode",
    "LoadWeightsNode", "LoadCheckpointNode", "LoadModelNode", "PretrainedBlockNode",
]
