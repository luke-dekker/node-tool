"""Tests for PyTorch tensor visualization nodes."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
import torch


def _has_pil():
    try:
        from PIL import Image  # noqa
        return True
    except ImportError:
        return False


skip_pil = pytest.mark.skipif(not _has_pil(), reason="PIL not installed")


# ── TensorVizNode — heatmap mode ──────────────────────────────────────────────

@skip_pil
def test_plot_tensor_2d():
    from nodes.pytorch.tensor_viz import TensorVizNode
    t = torch.rand(8, 8)
    result = TensorVizNode().execute({"mode": "heatmap", "tensor": t, "title": "Test", "cmap": "viridis"})
    assert result["image"] is not None
    assert isinstance(result["image"], np.ndarray)
    assert result["image"].ndim == 3  # H,W,C


@skip_pil
def test_plot_tensor_higher_rank_sliced():
    from nodes.pytorch.tensor_viz import TensorVizNode
    t = torch.rand(3, 8, 8)  # should take first slice → 8x8
    result = TensorVizNode().execute({"mode": "heatmap", "tensor": t, "title": "3D", "cmap": "plasma"})
    assert result["image"] is not None


def test_plot_tensor_none():
    from nodes.pytorch.tensor_viz import TensorVizNode
    result = TensorVizNode().execute({"mode": "heatmap", "tensor": None, "title": "", "cmap": "viridis"})
    assert result["image"] is None


# ── PlotTrainingCurveNode ──────────────────────────────────────────────────────

@skip_pil
def test_training_curve_train_only():
    from nodes.pytorch.viz import PlotTrainingCurveNode
    losses = [1.0, 0.8, 0.6, 0.4]
    result = PlotTrainingCurveNode().execute({
        "train_losses": losses, "val_losses": None, "title": "Curve"
    })
    assert result["image"] is not None


@skip_pil
def test_training_curve_with_val():
    from nodes.pytorch.viz import PlotTrainingCurveNode
    result = PlotTrainingCurveNode().execute({
        "train_losses": [1.0, 0.8, 0.6],
        "val_losses":   [1.1, 0.9, 0.7],
        "title": "Train+Val"
    })
    assert result["image"] is not None


def test_training_curve_none():
    from nodes.pytorch.viz import PlotTrainingCurveNode
    result = PlotTrainingCurveNode().execute({"train_losses": None, "val_losses": None, "title": ""})
    assert result["image"] is None


# ── TensorVizNode — histogram mode ────────────────────────────────────────────

@skip_pil
def test_tensor_histogram():
    from nodes.pytorch.tensor_viz import TensorVizNode
    t = torch.randn(100)
    result = TensorVizNode().execute({"mode": "histogram", "tensor": t, "bins": 20, "title": "Hist", "color": "cyan"})
    assert result["image"] is not None


def test_tensor_histogram_none():
    from nodes.pytorch.tensor_viz import TensorVizNode
    result = TensorVizNode().execute({"mode": "histogram", "tensor": None, "bins": 20, "title": "", "color": "cyan"})
    assert result["image"] is None


# ── TensorVizNode — scatter mode ──────────────────────────────────────────────

@skip_pil
def test_tensor_scatter_points():
    from nodes.pytorch.tensor_viz import TensorVizNode
    pts = torch.rand(50, 2)
    result = TensorVizNode().execute({
        "mode": "scatter", "points": pts, "x": None, "y": None, "labels": None,
        "title": "Scatter", "alpha": 0.8
    })
    assert result["image"] is not None


@skip_pil
def test_tensor_scatter_xy():
    from nodes.pytorch.tensor_viz import TensorVizNode
    x = torch.rand(30)
    y = torch.rand(30)
    result = TensorVizNode().execute({
        "mode": "scatter", "points": None, "x": x, "y": y, "labels": None,
        "title": "XY", "alpha": 0.7
    })
    assert result["image"] is not None


@skip_pil
def test_tensor_scatter_with_labels():
    from nodes.pytorch.tensor_viz import TensorVizNode
    pts = torch.rand(40, 2)
    labels = torch.randint(0, 3, (40,))
    result = TensorVizNode().execute({
        "mode": "scatter", "points": pts, "x": None, "y": None, "labels": labels,
        "title": "Coloured", "alpha": 0.7
    })
    assert result["image"] is not None


def test_tensor_scatter_none():
    from nodes.pytorch.tensor_viz import TensorVizNode
    result = TensorVizNode().execute({
        "mode": "scatter", "points": None, "x": None, "y": None, "labels": None,
        "title": "", "alpha": 0.7
    })
    assert result["image"] is None


# ── TensorVizNode — image mode ────────────────────────────────────────────────

@skip_pil
def test_show_image_chw():
    from nodes.pytorch.tensor_viz import TensorVizNode
    t = torch.rand(3, 32, 32)  # CHW colour
    result = TensorVizNode().execute({"mode": "image", "tensor": t, "title": "RGB"})
    assert result["image"] is not None


@skip_pil
def test_show_image_grayscale():
    from nodes.pytorch.tensor_viz import TensorVizNode
    t = torch.rand(1, 28, 28)  # 1-channel CHW
    result = TensorVizNode().execute({"mode": "image", "tensor": t, "title": "Gray"})
    assert result["image"] is not None


@skip_pil
def test_show_image_2d():
    from nodes.pytorch.tensor_viz import TensorVizNode
    t = torch.rand(28, 28)  # bare 2D
    result = TensorVizNode().execute({"mode": "image", "tensor": t, "title": "2D"})
    assert result["image"] is not None


def test_show_image_none():
    from nodes.pytorch.tensor_viz import TensorVizNode
    result = TensorVizNode().execute({"mode": "image", "tensor": None, "title": ""})
    assert result["image"] is None


# ── WeightHistogramNode ────────────────────────────────────────────────────────

@skip_pil
def test_weight_histogram():
    import torch.nn as nn
    from nodes.pytorch.viz import WeightHistogramNode
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    result = WeightHistogramNode().execute({"model": model, "bins": 30, "title": "Weights"})
    assert result["image"] is not None
    assert result["model"] is model


def test_weight_histogram_none():
    from nodes.pytorch.viz import WeightHistogramNode
    result = WeightHistogramNode().execute({"model": None, "bins": 30, "title": ""})
    assert result["image"] is None
    assert result["model"] is None


# ── Registry check ─────────────────────────────────────────────────────────────

def test_pt_viz_nodes_registered():
    from nodes import NODE_REGISTRY
    expected = [
        "pt_tensor_viz", "pt_viz_training_curve",
        "pt_viz_weight_hist",
    ]
    for tn in expected:
        assert tn in NODE_REGISTRY, f"{tn} not in registry"
