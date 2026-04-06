"""Tests for save/load persistence nodes."""
import sys, os, tempfile, pathlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import torch.nn as nn


# ── helpers ───────────────────────────────────────────────────────────────────

def _simple_model():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


# ── SaveWeightsNode ───────────────────────────────────────────────────────────

def test_save_weights_creates_file():
    from nodes.pytorch.persistence import SaveWeightsNode
    model = _simple_model()
    with tempfile.TemporaryDirectory() as tmp:
        path = str(pathlib.Path(tmp) / "weights.pt")
        result = SaveWeightsNode().execute({"model": model, "path": path})
        assert pathlib.Path(path).exists()
        assert result["model"] is model      # pass-through
        assert result["path"] == path


def test_save_weights_none_model():
    from nodes.pytorch.persistence import SaveWeightsNode
    with tempfile.TemporaryDirectory() as tmp:
        path = str(pathlib.Path(tmp) / "weights.pt")
        result = SaveWeightsNode().execute({"model": None, "path": path})
        assert not pathlib.Path(path).exists()   # nothing saved
        assert result["model"] is None


# ── LoadWeightsNode ───────────────────────────────────────────────────────────

def test_load_weights_roundtrip():
    from nodes.pytorch.persistence import SaveWeightsNode, LoadWeightsNode
    model_a = _simple_model()
    model_b = _simple_model()   # same architecture, different random weights

    with tempfile.TemporaryDirectory() as tmp:
        path = str(pathlib.Path(tmp) / "weights.pt")
        SaveWeightsNode().execute({"model": model_a, "path": path})

        result = LoadWeightsNode().execute({"model": model_b, "path": path, "device": "cpu"})
        loaded = result["model"]

        # weights should now be equal
        for pa, pb in zip(model_a.parameters(), loaded.parameters()):
            assert torch.allclose(pa, pb)


def test_load_weights_none_model():
    from nodes.pytorch.persistence import LoadWeightsNode
    # Should not crash when model is None (nothing to load into)
    result = LoadWeightsNode().execute({"model": None, "path": "nonexistent.pt", "device": "cpu"})
    assert result["model"] is None


# ── SaveCheckpointNode ────────────────────────────────────────────────────────

def test_save_checkpoint_creates_file():
    from nodes.pytorch.persistence import SaveCheckpointNode
    model = _simple_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    with tempfile.TemporaryDirectory() as tmp:
        path = str(pathlib.Path(tmp) / "ckpt.pt")
        result = SaveCheckpointNode().execute({
            "model": model, "optimizer": optimizer,
            "epoch": 3, "loss": 0.42, "path": path,
        })
        assert pathlib.Path(path).exists()
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert ckpt["epoch"] == 3
        assert abs(ckpt["loss"] - 0.42) < 1e-6
        assert "model_state_dict" in ckpt
        assert "optimizer_state_dict" in ckpt


def test_save_checkpoint_without_optimizer():
    from nodes.pytorch.persistence import SaveCheckpointNode
    model = _simple_model()
    with tempfile.TemporaryDirectory() as tmp:
        path = str(pathlib.Path(tmp) / "ckpt.pt")
        SaveCheckpointNode().execute({
            "model": model, "optimizer": None,
            "epoch": 0, "loss": 0.0, "path": path,
        })
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        assert ckpt["optimizer_state_dict"] is None


# ── LoadCheckpointNode ────────────────────────────────────────────────────────

def test_load_checkpoint_roundtrip():
    from nodes.pytorch.persistence import SaveCheckpointNode, LoadCheckpointNode
    model_a   = _simple_model()
    optimizer_a = torch.optim.Adam(model_a.parameters(), lr=1e-3)
    model_b   = _simple_model()
    optimizer_b = torch.optim.Adam(model_b.parameters(), lr=1e-3)

    with tempfile.TemporaryDirectory() as tmp:
        path = str(pathlib.Path(tmp) / "ckpt.pt")
        SaveCheckpointNode().execute({
            "model": model_a, "optimizer": optimizer_a,
            "epoch": 7, "loss": 0.123, "path": path,
        })
        result = LoadCheckpointNode().execute({
            "model": model_b, "optimizer": optimizer_b,
            "path": path, "device": "cpu",
        })
        assert result["epoch"] == 7
        assert abs(result["loss"] - 0.123) < 1e-5
        for pa, pb in zip(model_a.parameters(), result["model"].parameters()):
            assert torch.allclose(pa, pb)


def test_load_checkpoint_none_guards():
    from nodes.pytorch.persistence import LoadCheckpointNode
    # No crash when model/optimizer are None
    with tempfile.TemporaryDirectory() as tmp:
        # Save a minimal checkpoint first
        import torch
        path = str(pathlib.Path(tmp) / "ckpt.pt")
        torch.save({"epoch": 2, "loss": 0.5}, path)
        result = LoadCheckpointNode().execute({
            "model": None, "optimizer": None,
            "path": path, "device": "cpu",
        })
        assert result["epoch"] == 2
        assert result["model"] is None
        assert result["optimizer"] is None


# ── ExportONNXNode ────────────────────────────────────────────────────────────

def test_export_onnx_runs_without_crash():
    """ONNX export may fail if onnxscript is not installed — that's fine,
    but the node must not raise and must return a meaningful info string."""
    from nodes.pytorch.persistence import ExportONNXNode
    model = _simple_model()
    with tempfile.TemporaryDirectory() as tmp:
        path = str(pathlib.Path(tmp) / "model.onnx")
        result = ExportONNXNode().execute({
            "model": model, "input_shape": "1,4", "path": path, "opset": 17,
        })
        # Either succeeded or returned a descriptive failure message
        assert result["info"] != ""
        assert result["path"] == path


def test_export_onnx_none_model():
    from nodes.pytorch.persistence import ExportONNXNode
    result = ExportONNXNode().execute({
        "model": None, "input_shape": "1,4", "path": "x.onnx", "opset": 17,
    })
    assert "No model" in result["info"]


def test_export_onnx_bad_shape():
    from nodes.pytorch.persistence import ExportONNXNode
    model = _simple_model()
    with tempfile.TemporaryDirectory() as tmp:
        path = str(pathlib.Path(tmp) / "model.onnx")
        result = ExportONNXNode().execute({
            "model": model, "input_shape": "1,999",  # wrong input size
            "path": path, "opset": 17,
        })
        assert "failed" in result["info"].lower()


# ── SaveFullModelNode ──────────────────────────────────────────────────────────

def test_save_full_model_creates_file():
    from nodes.pytorch.persistence import SaveFullModelNode
    model = _simple_model()
    with tempfile.TemporaryDirectory() as tmp:
        path = str(pathlib.Path(tmp) / "full.pt")
        result = SaveFullModelNode().execute({"model": model, "path": path})
        assert pathlib.Path(path).exists()
        assert result["model"] is model
        assert "Saved" in result["info"]

def test_save_full_model_none():
    from nodes.pytorch.persistence import SaveFullModelNode
    result = SaveFullModelNode().execute({"model": None, "path": "x.pt"})
    assert result["model"] is None
    assert "No model" in result["info"]


# ── PretrainedBlockNode ────────────────────────────────────────────────────────

def test_pretrained_block_roundtrip():
    from nodes.pytorch.persistence import SaveFullModelNode, PretrainedBlockNode
    model = _simple_model()
    with tempfile.TemporaryDirectory() as tmp:
        path = str(pathlib.Path(tmp) / "full.pt")
        SaveFullModelNode().execute({"model": model, "path": path})
        result = PretrainedBlockNode().execute({
            "path": path, "device": "cpu",
            "freeze_all": False, "trainable_layers": 0, "eval_mode": False,
        })
        assert result["model"] is not None
        assert "Sequential" in result["info"]
        assert "Total:" in result["info"]

def test_pretrained_block_freeze_all():
    from nodes.pytorch.persistence import SaveFullModelNode, PretrainedBlockNode
    model = _simple_model()
    with tempfile.TemporaryDirectory() as tmp:
        path = str(pathlib.Path(tmp) / "full.pt")
        SaveFullModelNode().execute({"model": model, "path": path})
        result = PretrainedBlockNode().execute({
            "path": path, "device": "cpu",
            "freeze_all": True, "trainable_layers": 0, "eval_mode": False,
        })
        loaded = result["model"]
        assert all(not p.requires_grad for p in loaded.parameters())
        assert "Trainable: 0" in result["info"]

def test_pretrained_block_freeze_except_last():
    from nodes.pytorch.persistence import SaveFullModelNode, PretrainedBlockNode
    model = _simple_model()   # 3 children: Linear, ReLU, Linear
    with tempfile.TemporaryDirectory() as tmp:
        path = str(pathlib.Path(tmp) / "full.pt")
        SaveFullModelNode().execute({"model": model, "path": path})
        result = PretrainedBlockNode().execute({
            "path": path, "device": "cpu",
            "freeze_all": True, "trainable_layers": 1, "eval_mode": False,
        })
        loaded = result["model"]
        trainable = sum(p.numel() for p in loaded.parameters() if p.requires_grad)
        assert trainable > 0   # last layer is trainable

def test_pretrained_block_eval_mode():
    from nodes.pytorch.persistence import SaveFullModelNode, PretrainedBlockNode
    model = _simple_model()
    with tempfile.TemporaryDirectory() as tmp:
        path = str(pathlib.Path(tmp) / "full.pt")
        SaveFullModelNode().execute({"model": model, "path": path})
        result = PretrainedBlockNode().execute({
            "path": path, "device": "cpu",
            "freeze_all": False, "trainable_layers": 0, "eval_mode": True,
        })
        assert not result["model"].training
        assert "eval" in result["info"]

def test_pretrained_block_bad_path():
    from nodes.pytorch.persistence import PretrainedBlockNode
    result = PretrainedBlockNode().execute({
        "path": "/nonexistent/path/model.pt", "device": "cpu",
        "freeze_all": False, "trainable_layers": 0, "eval_mode": False,
    })
    assert result["model"] is None
    assert "Load failed" in result["info"]

def test_pretrained_block_forward_runs():
    from nodes.pytorch.persistence import SaveFullModelNode, PretrainedBlockNode
    model = _simple_model()
    with tempfile.TemporaryDirectory() as tmp:
        path = str(pathlib.Path(tmp) / "full.pt")
        SaveFullModelNode().execute({"model": model, "path": path})
        loaded = PretrainedBlockNode().execute({
            "path": path, "device": "cpu",
            "freeze_all": False, "trainable_layers": 0, "eval_mode": True,
        })["model"]
        import torch
        x = torch.randn(2, 4)
        out = loaded(x)
        assert out.shape == (2, 2)


# ── ModelInfoNode ─────────────────────────────────────────────────────────────

def test_model_info_shows_params():
    from nodes.pytorch.persistence import ModelInfoNode
    model = _simple_model()
    result = ModelInfoNode().execute({"model": model})
    assert result["model"] is model
    assert "Total params" in result["info"]
    assert "Trainable" in result["info"]


def test_model_info_frozen_params():
    from nodes.pytorch.persistence import ModelInfoNode
    model = _simple_model()
    for p in model[0].parameters():   # freeze first layer
        p.requires_grad = False
    result = ModelInfoNode().execute({"model": model})
    assert "Frozen" in result["info"]


def test_model_info_none():
    from nodes.pytorch.persistence import ModelInfoNode
    result = ModelInfoNode().execute({"model": None})
    assert "No model" in result["info"]


# ── Exporter integration ──────────────────────────────────────────────────────

def test_exporter_persistence_nodes():
    from core.graph import Graph
    from core.exporter import GraphExporter
    from nodes.pytorch.persistence import (
        SaveWeightsNode, LoadWeightsNode,
        SaveCheckpointNode, LoadCheckpointNode,
        ExportONNXNode, ModelInfoNode,
    )
    exporter = GraphExporter()
    for NodeCls in [SaveWeightsNode, LoadWeightsNode,
                    SaveCheckpointNode, LoadCheckpointNode,
                    ExportONNXNode, ModelInfoNode]:
        g = Graph()
        g.add_node(NodeCls())
        script = exporter.export(g)
        assert "export not" not in script.lower(), f"{NodeCls.__name__} missing exporter"


def test_save_weights_exporter_contains_torch_save():
    from core.graph import Graph
    from core.exporter import GraphExporter
    from nodes.pytorch.persistence import SaveWeightsNode
    g = Graph()
    g.add_node(SaveWeightsNode())
    script = GraphExporter().export(g)
    assert "torch.save" in script


def test_load_weights_exporter_contains_load():
    from core.graph import Graph
    from core.exporter import GraphExporter
    from nodes.pytorch.persistence import LoadWeightsNode
    g = Graph()
    g.add_node(LoadWeightsNode())
    script = GraphExporter().export(g)
    assert "torch.load" in script


def test_checkpoint_exporter_has_epoch_and_loss():
    from core.graph import Graph
    from core.exporter import GraphExporter
    from nodes.pytorch.persistence import SaveCheckpointNode
    g = Graph()
    g.add_node(SaveCheckpointNode())
    script = GraphExporter().export(g)
    assert "epoch" in script
    assert "loss" in script
