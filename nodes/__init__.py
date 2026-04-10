"""Node registry — auto-discovers all node classes from the nodes package."""

from __future__ import annotations
import inspect
from typing import Type
from core.node import BaseNode

# Import all node subpackages to trigger registration
from nodes import math, logic, string, data, numpy, pandas, sklearn, scipy, viz, code
from nodes import pytorch, io, ai

# Import individual pytorch node modules (shims forward to individual files)
from nodes.pytorch import (
    layers, losses, optimizers, schedulers,
    dataset_transforms,
    tensor_ops, recurrent, backbones, persistence,
    viz as pt_viz, architecture,
)

# Import individual IO and AI modules (shims forward to individual files)
from nodes.io import serial_nodes, network_nodes, file_nodes
from nodes.ai import ollama_nodes, hf_nodes

NODE_REGISTRY: dict[str, Type[BaseNode]] = {}

def _discover(module) -> None:
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if (issubclass(obj, BaseNode)
                and obj is not BaseNode
                and hasattr(obj, "type_name")
                and obj.type_name != "base"):
            NODE_REGISTRY[obj.type_name] = obj

_discover(math)
_discover(logic)
_discover(string)
_discover(data)
_discover(numpy)
_discover(pandas)
_discover(sklearn)
_discover(scipy)
_discover(viz)
_discover(layers)
_discover(losses)
_discover(optimizers)
_discover(schedulers)
_discover(dataset_transforms)
_discover(tensor_ops)
_discover(recurrent)
_discover(backbones)
_discover(persistence)
_discover(pt_viz)
_discover(architecture)
_discover(pytorch)   # catches any individual files not covered above
_discover(serial_nodes)
_discover(network_nodes)
_discover(file_nodes)
_discover(io)
_discover(ollama_nodes)
_discover(hf_nodes)
_discover(ai)
_discover(code)

# Subgraphs are auto-generated dynamic classes — discover them last so the
# inner-graph nodes they reference are already in NODE_REGISTRY.
from nodes import subgraphs
_discover(subgraphs)

# Remove obsolete nodes from the palette. Their .py files stay on disk so
# old saved graphs can still load, but they don't clutter the palette.
# These are all replaced by the new architecture:
#   - Individual dataset nodes → universal DatasetNode
#   - BatchInput → dataset x/label ports
#   - MultiDataset, MultimodalModel → per-layer wiring
#   - TrainingConfig, MultimodalTrainingConfig → panel + TrainOutput
#   - DataLoader, DataLoaderInfo, DatasetInfo, SampleBatch → universal DatasetNode
#   - GaussianNoise → GateNode (noise mode)
#   - Legacy single-type const/cast nodes → ConstNode, CastNode
# Keep: MNIST, CIFAR10, TextDataset, HFDataset (auto-download benchmarks).
# Keep: TrainOutput, Dataset (universal), Gate, LossCompute, ApplyModule (new arch).
# Keep: FreezeLayersNode, FreezeNamedLayersNode (useful experiment controls).
# Keep: ConstNode, CastNode, PreviewNode, ImageInputNode (consolidated v5).
# Remove: everything replaced by the new architecture.
_OBSOLETE = {
    # Legacy training adapters — replaced by panel + TrainOutput
    "batch_input", "pt_training_config", "pt_multimodal_training_config",
    # Monolithic multi-modal — replaced by per-layer wiring
    "pt_multi_dataset", "pt_multimodal_model",
    # Folder-based dataset nodes — replaced by universal DatasetNode
    "pt_csv_dataset", "pt_numpy_dataset", "pt_image_folder_dataset",
    "pt_audio_folder_dataset", "pt_folder_multimodal_dataset",
    # Utility nodes absorbed into universal DatasetNode or GateNode
    "pt_dataloader", "pt_dataloader_info", "pt_dataset_info", "pt_sample_batch",
    "pt_gaussian_noise",
    # Legacy single-type const/cast — replaced by ConstNode/CastNode
    "float_const", "int_const", "bool_const", "string_const",
    "to_float", "to_int", "to_bool", "to_string",
    # Old forward pass node
    "pt_forward_pass",
}
for _tn in _OBSOLETE:
    NODE_REGISTRY.pop(_tn, None)

# ── Plugin system ────────────────────────────────────────────────────────────
# Discover and load plugins from the plugins/ directory. Each plugin calls
# register(ctx) which can add port types, node classes, panel builders, and
# categories. Plugins are loaded AFTER the built-in nodes so they can
# reference existing node types in their subgraphs/templates.
try:
    from core.plugins import load_plugins
    _plugin_ctx = load_plugins()
    # Merge plugin-registered nodes into NODE_REGISTRY
    for _cls in _plugin_ctx.node_classes:
        if hasattr(_cls, "type_name") and _cls.type_name not in NODE_REGISTRY:
            NODE_REGISTRY[_cls.type_name] = _cls
except Exception as _exc:
    print(f"[plugins] Plugin loading failed: {_exc}")
    _plugin_ctx = None

# Grouped by category for the palette
CATEGORY_ORDER = [
    # ML workflow
    "Datasets", "Layers", "Models", "Training", "AI", "Analyze",
    # General compute
    "Python", "NumPy", "Pandas", "Sklearn", "SciPy",
    # Reusable building blocks
    "Subgraphs",
    # Output
    "IO",
]

def get_nodes_by_category() -> dict[str, list[Type[BaseNode]]]:
    result: dict[str, list[Type[BaseNode]]] = {}
    for cls in NODE_REGISTRY.values():
        cat = cls.category
        result.setdefault(cat, [])
        result[cat].append(cls)
    # Sort each category alphabetically by label
    for cat in result:
        result[cat].sort(key=lambda c: c.label)
    return result
