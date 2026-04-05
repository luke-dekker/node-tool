"""Node registry — auto-discovers all node classes from the nodes package."""

from __future__ import annotations
import inspect
from typing import Type
from core.node import BaseNode

# Import all node subpackages to trigger registration
from nodes import math, logic, string, data, numpy, pandas, sklearn, scipy, viz, code
from nodes import pytorch
# also import pytorch submodules so _discover finds all classes
from nodes.pytorch import layers, losses, optimizers, schedulers
from nodes.pytorch import data as pt_data, training
from nodes.pytorch import dataset_sources, dataset_transforms, dataset_ops
from nodes.pytorch import tensor_ops, recurrent, backbones
from nodes.pytorch import autoencoder
from nodes.pytorch import persistence
from nodes.pytorch import viz as pt_viz
# pipeline.py is now a stub — nodes unified into layers.py
from nodes import io
from nodes.io import serial_nodes, network_nodes, file_nodes
from nodes import ai
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
_discover(serial_nodes)
_discover(network_nodes)
_discover(file_nodes)
_discover(ollama_nodes)
_discover(hf_nodes)
_discover(code)

# Grouped by category for the palette
CATEGORY_ORDER = [
    # ML workflow
    "Datasets", "Layers", "Models", "Training", "AI", "Analyze",
    # General compute
    "Data", "NumPy", "Pandas", "Math", "Logic", "String", "Sklearn", "SciPy",
    # Output
    "IO", "Code",
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
