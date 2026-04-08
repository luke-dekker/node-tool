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
    training, dataset_sources, dataset_transforms, dataset_ops,
    tensor_ops, recurrent, backbones, autoencoder, persistence,
    viz as pt_viz, architecture,
)
from nodes.pytorch import data as pt_data

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
_discover(pt_data)
_discover(training)
_discover(dataset_sources)
_discover(dataset_transforms)
_discover(dataset_ops)
_discover(tensor_ops)
_discover(recurrent)
_discover(backbones)
_discover(autoencoder)
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
