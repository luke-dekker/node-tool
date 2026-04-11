"""Tests for dataset pipeline nodes — no DPG, no network required."""
import pytest
import numpy as np


def test_to_tensor_transform():
    torchvision = pytest.importorskip("torchvision")
    from nodes.pytorch.dataset_transforms import ToTensorTransformNode
    result = ToTensorTransformNode().execute({})
    from torchvision.transforms import ToTensor
    assert isinstance(result["transform"], ToTensor)


def test_resize_transform():
    pytest.importorskip("torchvision")
    from nodes.pytorch.dataset_transforms import ResizeTransformNode
    result = ResizeTransformNode().execute({"height": 64, "width": 64})
    assert result["transform"] is not None


def test_normalize_transform():
    pytest.importorskip("torchvision")
    from nodes.pytorch.dataset_transforms import NormalizeTransformNode
    result = NormalizeTransformNode().execute({"mean": "0.5,0.5,0.5", "std": "0.5,0.5,0.5"})
    from torchvision.transforms import Normalize
    assert isinstance(result["transform"], Normalize)


def test_compose_transforms():
    pytest.importorskip("torchvision")
    from nodes.pytorch.dataset_transforms import ComposeTransformsNode, ToTensorTransformNode, ResizeTransformNode
    t1 = ResizeTransformNode().execute({"height": 32, "width": 32})["transform"]
    t2 = ToTensorTransformNode().execute({})["transform"]
    result = ComposeTransformsNode().execute({"t1": t1, "t2": t2, "t3": None, "t4": None, "t5": None, "t6": None})
    from torchvision.transforms import Compose
    assert isinstance(result["transform"], Compose)


def test_random_hflip_transform():
    pytest.importorskip("torchvision")
    from nodes.pytorch.dataset_transforms import RandomHFlipTransformNode
    result = RandomHFlipTransformNode().execute({"p": 0.5})
    assert result["transform"] is not None


def test_grayscale_transform():
    pytest.importorskip("torchvision")
    from nodes.pytorch.dataset_transforms import GrayscaleTransformNode
    result = GrayscaleTransformNode().execute({"num_channels": 1})
    assert result["transform"] is not None


def test_registry_has_dataset_nodes():
    from nodes import NODE_REGISTRY
    # Universal DatasetNode replaces all specialized dataset nodes
    expected = ["pt_dataset", "pt_train_val_split", "pt_train_val_test_split",
                "pt_compose_transforms", "pt_to_tensor_transform",
                "pt_resize_transform", "pt_normalize_transform",
                "pt_apply_transform"]
    for tn in expected:
        assert tn in NODE_REGISTRY, f"Missing from registry: {tn}"
