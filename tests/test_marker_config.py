"""A/B markers carry training config as input ports — training params
flow from markers first, with `params` dict as an override layer. This
is the foundation for letting the autoresearch agent wire `control`
directly into a marker port (e.g. `B.lr`) to make it agent-tunable.

Tests cover:
  - Marker defaults drive training when `params` is empty.
  - `params` overrides marker defaults field-by-field.
  - Single-B graphs auto-promote to primary; multi-B requires exactly one.
  - Multiple primaries or zero primaries (with >1 B) raise a clear error.
"""
from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


def _tiny_loader(n: int = 40, batch_size: int = 8) -> DataLoader:
    x = torch.randn(n, 4)
    y = torch.randint(0, 3, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


@pytest.fixture(autouse=True)
def _register_port_types():
    from plugins.agents.port_types import register_all
    register_all()


@pytest.fixture
def stub_dataset(monkeypatch):
    """Replace DatasetNode.execute so tests stay hermetic (no torchvision
    download). Returns a loader whose batch_size matches what's passed in."""
    from nodes.pytorch._dataset_loader import DatasetNode
    def _fake(self, inputs):
        bs = int(inputs.get("batch_size", 8))
        return {"dataloader": _tiny_loader(n=40, batch_size=bs), "info": "stub"}
    monkeypatch.setattr(DatasetNode, "execute", _fake)


# ── Marker-driven training config ─────────────────────────────────────────

class _FakeCtrl:
    status = "idle"
    last_model = None
    def __init__(self): self.cfg = {}
    def start(self, cfg): self.cfg = cfg
    def poll(self): return []


def _make_orch(graph):
    from plugins.pytorch.training_orchestrator import TrainingOrchestrator
    orch = TrainingOrchestrator(graph)
    orch._ctrl = _FakeCtrl()
    return orch


def test_marker_defaults_drive_training_without_params_overrides(stub_dataset):
    """Markers alone specify everything; params dict carries nothing but
    device. Training should still start and use marker values."""
    from core.graph import Graph
    from core.node import MarkerRole
    from templates.mnist_mlp import build
    g = Graph()
    build(g)
    a = next(n for n in g.nodes_by_role(MarkerRole.INPUT))
    a.inputs["path"].default_value = "stub"
    a.inputs["batch_size"].default_value = 16
    a.inputs["val_fraction"].default_value = 0.2
    b = next(n for n in g.nodes_by_role(MarkerRole.TRAIN_TARGET))
    b.inputs["lr"].default_value = 0.002
    b.inputs["optimizer"].default_value = "adam"
    b.inputs["loss"].default_value = "crossentropy"
    b.inputs["epochs"].default_value = 5

    orch = _make_orch(g)
    r = orch.start({"device": "cpu"})
    assert r.get("ok"), r
    cfg = orch._ctrl.cfg
    assert cfg["epochs"] == 5
    # val_fraction=0.2 on 40-sample stub dataset → 8 val samples, loader exists
    assert cfg["val_dataloader"] is not None
    # Optimizer reflects the marker's lr
    found_lr = [g["lr"] for g in cfg["optimizer"].param_groups]
    assert all(abs(lr - 0.002) < 1e-9 for lr in found_lr)


def test_params_override_marker_defaults_field_by_field(stub_dataset):
    """The panel still submits `params` today — its values must win over
    marker defaults for backward compat during the phase-1 transition."""
    from core.graph import Graph
    from core.node import MarkerRole
    from templates.mnist_mlp import build
    g = Graph()
    build(g)
    a = next(n for n in g.nodes_by_role(MarkerRole.INPUT))
    a.inputs["path"].default_value = "stub-marker"
    a.inputs["batch_size"].default_value = 16
    b = next(n for n in g.nodes_by_role(MarkerRole.TRAIN_TARGET))
    b.inputs["lr"].default_value = 0.002
    b.inputs["epochs"].default_value = 5

    orch = _make_orch(g)
    r = orch.start({
        "device": "cpu",
        "lr": 0.05,           # override marker's 0.002
        "epochs": 2,          # override marker's 5
        "datasets": {"task_1": {"batch_size": 64}},  # override marker's 16
    })
    assert r.get("ok"), r
    cfg = orch._ctrl.cfg
    assert cfg["epochs"] == 2
    found_lr = [g["lr"] for g in cfg["optimizer"].param_groups]
    assert all(abs(lr - 0.05) < 1e-9 for lr in found_lr)
    # batch_size override flowed → dataset loader's batch_size is 64
    assert cfg["dataloader"].batch_size == 64


def test_params_blank_string_falls_through_to_marker_default(stub_dataset):
    """Panel fields submitted as empty strings (common when user left the
    field blank) should not stomp the marker default."""
    from core.graph import Graph
    from core.node import MarkerRole
    from templates.mnist_mlp import build
    g = Graph()
    build(g)
    a = next(n for n in g.nodes_by_role(MarkerRole.INPUT))
    a.inputs["path"].default_value = "stub-marker"
    a.inputs["batch_size"].default_value = 16
    b = next(n for n in g.nodes_by_role(MarkerRole.TRAIN_TARGET))
    b.inputs["optimizer"].default_value = "sgd"

    orch = _make_orch(g)
    r = orch.start({
        "device": "cpu",
        "optimizer": "",   # blank — should NOT override to empty
        "datasets": {"task_1": {"path": ""}},
    })
    assert r.get("ok"), r
    # SGD was selected from the marker default, not overridden to empty.
    assert type(orch._ctrl.cfg["optimizer"]).__name__.lower() == "sgd"


# ── Primary B validation ────────────────────────────────────────────────

def test_single_b_auto_promotes_to_primary(stub_dataset):
    """Most graphs have one B marker — user shouldn't have to flip primary."""
    from core.graph import Graph
    from core.node import MarkerRole
    from templates.mnist_mlp import build
    g = Graph()
    build(g)
    a = next(n for n in g.nodes_by_role(MarkerRole.INPUT))
    a.inputs["path"].default_value = "stub"
    b = next(n for n in g.nodes_by_role(MarkerRole.TRAIN_TARGET))
    # b.primary is False by default — single-B case must still train.
    assert b.inputs["primary"].default_value is False

    orch = _make_orch(g)
    r = orch.start({"device": "cpu"})
    assert r.get("ok"), r


def test_multi_b_without_primary_errors_clearly():
    """Multi-task graphs need an explicit primary flag."""
    from core.graph import Graph
    from core.node import MarkerRole
    from nodes.pytorch.input_marker import InputMarkerNode
    from nodes.pytorch.train_marker import TrainMarkerNode

    g = Graph()
    a = InputMarkerNode()
    a.inputs["group"].default_value = "task_a"
    a.inputs["path"].default_value = "stub"
    g.add_node(a)
    b1 = TrainMarkerNode()
    b1.inputs["group"].default_value = "task_a"
    g.add_node(b1)
    # Second group + second B marker — neither flagged primary.
    a2 = InputMarkerNode()
    a2.inputs["group"].default_value = "task_b"
    a2.inputs["path"].default_value = "stub"
    g.add_node(a2)
    b2 = TrainMarkerNode()
    b2.inputs["group"].default_value = "task_b"
    g.add_node(b2)
    g.add_connection(a.id, "tensor", b1.id, "tensor_in")
    g.add_connection(a2.id, "tensor", b2.id, "tensor_in")

    orch = _make_orch(g)
    r = orch.start({"device": "cpu"})
    assert r.get("ok") is False
    assert "primary=True" in r["error"]


def test_multi_b_with_two_primaries_errors_clearly():
    from core.graph import Graph
    from nodes.pytorch.input_marker import InputMarkerNode
    from nodes.pytorch.train_marker import TrainMarkerNode

    g = Graph()
    a1 = InputMarkerNode(); a1.inputs["group"].default_value = "task_a"
    a1.inputs["path"].default_value = "stub"
    b1 = TrainMarkerNode(); b1.inputs["group"].default_value = "task_a"
    b1.inputs["primary"].default_value = True
    a2 = InputMarkerNode(); a2.inputs["group"].default_value = "task_b"
    a2.inputs["path"].default_value = "stub"
    b2 = TrainMarkerNode(); b2.inputs["group"].default_value = "task_b"
    b2.inputs["primary"].default_value = True
    for n in (a1, b1, a2, b2):
        g.add_node(n)
    g.add_connection(a1.id, "tensor", b1.id, "tensor_in")
    g.add_connection(a2.id, "tensor", b2.id, "tensor_in")

    orch = _make_orch(g)
    r = orch.start({"device": "cpu"})
    assert r.get("ok") is False
    assert "exactly one" in r["error"]
