"""Training panel's per-group val split — unit test on `_split_train_val`
plus an integration test that val_fraction forwarded through `train_start`
produces a val_dataloader on the built task list.

Avoids downloading real MNIST: feeds a tiny in-memory TensorDataset straight
into the orchestrator path by patching DatasetNode.execute.
"""
from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from plugins.pytorch.training_orchestrator import _split_train_val


# ── Unit test: the split helper ────────────────────────────────────────────

def _tiny_loader(n: int = 100, batch_size: int = 10) -> DataLoader:
    x = torch.randn(n, 4)
    y = torch.randint(0, 3, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)


def test_split_train_val_honors_fraction_and_preserves_disjoint():
    dl = _tiny_loader(n=100, batch_size=16)
    train_dl, val_dl, info = _split_train_val(dl, batch_size=16, val_fraction=0.1)
    assert train_dl is not None and val_dl is not None
    assert len(train_dl.dataset) == 90
    assert len(val_dl.dataset) == 10
    assert "90 train / 10 val" in info
    # Subsets come from the same underlying dataset — indices must be disjoint.
    train_idx = set(train_dl.dataset.indices)
    val_idx = set(val_dl.dataset.indices)
    assert train_idx.isdisjoint(val_idx)
    assert train_idx | val_idx == set(range(100))


def test_split_train_val_seed_is_stable_across_calls():
    """Two splits of the same dataset with the same fraction produce the
    same partition — so val set doesn't drift across restarts."""
    dl_a = _tiny_loader(n=100)
    dl_b = _tiny_loader(n=100)   # different shuffle order, same dataset size
    _, val_a, _ = _split_train_val(dl_a, batch_size=16, val_fraction=0.2)
    _, val_b, _ = _split_train_val(dl_b, batch_size=16, val_fraction=0.2)
    assert list(val_a.dataset.indices) == list(val_b.dataset.indices)


def test_split_train_val_rejects_tiny_datasets():
    dl = _tiny_loader(n=1)
    train_dl, val_dl, info = _split_train_val(dl, batch_size=1, val_fraction=0.2)
    assert train_dl is None and val_dl is None
    assert "too small" in info


def test_split_train_val_rejects_iterable_datasets():
    class _NoLen:
        # No __len__ — mimics an IterableDataset / streaming source.
        def __iter__(self):
            return iter([])

    class _FakeDL:
        dataset = _NoLen()
        collate_fn = None

    train_dl, val_dl, info = _split_train_val(_FakeDL(), batch_size=1, val_fraction=0.2)
    assert train_dl is None and val_dl is None
    assert "no len" in info


def test_split_train_val_preserves_collate_fn():
    """Custom collate_fn should carry into both derived loaders so
    multimodal/structured datasets don't silently regress to default
    stacking."""
    sentinel_calls: list[int] = []

    def _collate(batch):
        sentinel_calls.append(len(batch))
        xs = torch.stack([b[0] for b in batch])
        ys = torch.stack([b[1] for b in batch])
        return xs, ys

    x = torch.randn(50, 4)
    y = torch.randint(0, 3, (50,))
    dl = DataLoader(TensorDataset(x, y), batch_size=8, collate_fn=_collate)
    train_dl, val_dl, _ = _split_train_val(dl, batch_size=8, val_fraction=0.2)
    assert train_dl.collate_fn is _collate
    assert val_dl.collate_fn is _collate
    # Iterating fires the collate function.
    next(iter(train_dl))
    next(iter(val_dl))
    assert len(sentinel_calls) >= 2


# ── Integration: val_fraction threaded through _start_marker_path ──────────

@pytest.fixture(autouse=True)
def _register_port_types():
    from plugins.agents.port_types import register_all
    register_all()


def test_val_fraction_routes_val_dataloader_into_task(monkeypatch):
    """Calling train_start with val_fraction>0 must produce a task whose
    `val_dataloader` is the held-out partition, not None."""
    from core.graph import Graph
    from templates.mnist_mlp import build
    from plugins.pytorch.training_orchestrator import TrainingOrchestrator
    from nodes.pytorch._dataset_loader import DatasetNode

    # Stub DatasetNode.execute to return a tiny in-memory loader — keeps the
    # test hermetic (no torchvision download, no disk I/O).
    def _fake_execute(self, inputs):
        dl = _tiny_loader(n=40, batch_size=inputs.get("batch_size", 8))
        return {"dataloader": dl, "info": "stubbed 40-sample loader"}

    monkeypatch.setattr(DatasetNode, "execute", _fake_execute)

    # Stub out the background TrainingController so we inspect the config
    # handed in instead of actually training.
    captured: dict = {}

    class _FakeCtrl:
        status = "idle"
        last_model = None
        train_losses: list[float] = []
        val_losses: list[float] = []
        best_loss = float("inf")
        last_batch_loss = None
        current_epoch = 0
        total_epochs = 0
        current_batch = 0
        error_message = None

        def start(self, cfg):
            captured.update(cfg)

        def poll(self):
            return []

    g = Graph()
    build(g)
    orch = TrainingOrchestrator(g)
    orch._ctrl = _FakeCtrl()   # bypass real training

    r = orch.start({
        "epochs": 1, "lr": 0.001, "optimizer": "adam",
        "loss": "crossentropy", "device": "cpu",
        "datasets": {"task_1": {"path": "stubbed", "batch_size": 8,
                                  "val_fraction": 0.25}},
    })
    assert r.get("ok"), r
    assert captured["val_dataloader"] is not None
    val_ds = captured["val_dataloader"].dataset
    assert len(val_ds) == 10       # 25% of 40
    # And the task list carries the per-group val loader for future
    # multi-group validation.
    assert captured["tasks"][0]["val_dataloader"] is not None


def test_successful_train_start_caches_params_for_autoresearch(monkeypatch):
    """Autoresearch can't refire training without the user's original
    dataset/loss/optimizer config — start() caches it on success so the
    `get_training_last_params` RPC returns something."""
    from core.graph import Graph
    from templates.mnist_mlp import build
    from plugins.pytorch.training_orchestrator import TrainingOrchestrator
    from nodes.pytorch._dataset_loader import DatasetNode

    monkeypatch.setattr(DatasetNode, "execute", lambda self, inputs: {
        "dataloader": _tiny_loader(n=20, batch_size=4), "info": "stub",
    })

    class _FakeCtrl:
        status = "idle"
        last_model = None
        def start(self, cfg): pass
        def poll(self): return []
    g = Graph()
    build(g)
    orch = TrainingOrchestrator(g)
    orch._ctrl = _FakeCtrl()

    # Before any successful start, no cached params.
    assert orch.handle_rpc("get_training_last_params", {})["params"] is None

    params = {
        "epochs": 2, "lr": 0.001, "optimizer": "adam",
        "loss": "crossentropy", "device": "cpu",
        "datasets": {"task_1": {"path": "stub", "batch_size": 4,
                                  "val_fraction": 0.25}},
    }
    r = orch.start(params)
    assert r.get("ok"), r

    cached = orch.handle_rpc("get_training_last_params", {})["params"]
    assert cached is not None
    assert cached["epochs"] == 2
    assert cached["datasets"]["task_1"]["val_fraction"] == 0.25
    # Deep-copied — mutating the cache doesn't bleed back.
    cached["epochs"] = 999
    cached_again = orch.handle_rpc("get_training_last_params", {})["params"]
    assert cached_again["epochs"] == 2


def test_failed_train_start_does_not_cache_params(monkeypatch):
    """A failed start (bad dataset etc.) must not poison the cache."""
    from core.graph import Graph
    from templates.mnist_mlp import build
    from plugins.pytorch.training_orchestrator import TrainingOrchestrator

    g = Graph()
    build(g)
    orch = TrainingOrchestrator(g)

    # No datasets → _start_marker_path returns ok=False.
    r = orch.start({"epochs": 1, "datasets": {}})
    assert r.get("ok") is False
    assert orch.handle_rpc("get_training_last_params", {})["params"] is None


def test_val_fraction_zero_leaves_val_dataloader_none(monkeypatch):
    from core.graph import Graph
    from templates.mnist_mlp import build
    from plugins.pytorch.training_orchestrator import TrainingOrchestrator
    from nodes.pytorch._dataset_loader import DatasetNode

    def _fake_execute(self, inputs):
        dl = _tiny_loader(n=40, batch_size=inputs.get("batch_size", 8))
        return {"dataloader": dl, "info": ""}

    monkeypatch.setattr(DatasetNode, "execute", _fake_execute)

    captured: dict = {}

    class _FakeCtrl:
        status = "idle"
        last_model = None
        def start(self, cfg): captured.update(cfg)
        def poll(self): return []

    g = Graph()
    build(g)
    orch = TrainingOrchestrator(g)
    orch._ctrl = _FakeCtrl()

    r = orch.start({
        "epochs": 1, "lr": 0.001, "optimizer": "adam",
        "loss": "crossentropy", "device": "cpu",
        "datasets": {"task_1": {"path": "stubbed", "batch_size": 8,
                                  "val_fraction": 0.0}},
    })
    assert r.get("ok"), r
    assert captured["val_dataloader"] is None
    assert captured["tasks"][0]["val_dataloader"] is None
