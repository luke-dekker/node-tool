"""Per-template one-step training smoke test.

Every shipped template with markers gets built, injected with a tiny
synthetic dataset (mimicking what the panel will do), wrapped as a
GraphAsModule, and run for one forward + backward pass. Catches broken
imports, stale APIs, shape mismatches, and GraphAsModule wiring breaks.

Marker-based: no DatasetNode in the graph. The test builds a DataLoader
directly from synthetic fixtures and primes input markers, exactly
mirroring the panel's future flow.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")


# ── Generic tiny-dataset builder ───────────────────────────────────────────

def _make_loader(samples_fn, n_samples, batch_size, collate_fn=None):
    """Build a DataLoader from a sample-factory function."""
    from torch.utils.data import DataLoader

    class _DS:
        def __init__(self):
            self.items = [samples_fn(i) for i in range(n_samples)]
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx):
            return self.items[idx]

    def _default_collate(items):
        keys = items[0].keys()
        return {k: torch.stack([s[k] for s in items]) for k in keys}

    return DataLoader(_DS(), batch_size=batch_size, shuffle=False,
                      drop_last=True, collate_fn=collate_fn or _default_collate)


# ── Per-template loader builders ───────────────────────────────────────────

def _build_mnist_loader(tmp_path, batch_size=4):
    def sample(i):
        return {
            "x":     torch.from_numpy(np.random.rand(1, 28, 28).astype(np.float32)),
            "label": torch.tensor(i % 4, dtype=torch.long),
        }
    return _make_loader(sample, 16, batch_size)


def _build_cifar_loader(tmp_path, batch_size=4):
    def sample(i):
        return {
            "x":     torch.from_numpy(np.random.rand(3, 32, 32).astype(np.float32)),
            "label": torch.tensor(i % 4, dtype=torch.long),
        }
    return _make_loader(sample, 16, batch_size)


def _build_text_loader(tmp_path, seq_len=64, batch_size=4):
    def sample(i):
        return {
            "x":     torch.randint(0, 256, (seq_len,)),
            "label": torch.randint(0, 256, (seq_len,)),
        }
    return _make_loader(sample, 16, batch_size)


def _build_text_loader_short(tmp_path, batch_size=4):
    return _build_text_loader(tmp_path, seq_len=32, batch_size=batch_size)


def _build_multimodal_loader(tmp_path, batch_size=4):
    def sample(i):
        return {
            "audio": torch.from_numpy(np.random.randn(64).astype(np.float32)),
            "image": torch.from_numpy(np.random.rand(3, 8, 8).astype(np.float32)),
            "label": torch.tensor(i % 2, dtype=torch.long),
        }
    return _make_loader(sample, 16, batch_size)


def _build_so101_loader(tmp_path, batch_size=2):
    _CHUNK, _DOF = 16, 6
    def sample(i):
        return {
            "observation.state":      torch.randn(_DOF),
            "observation.images.top": torch.rand(3, 4, 4),
            "action":                 torch.randn(_CHUNK, _DOF),
        }
    return _make_loader(sample, 8, batch_size)


# ── Test parametrization ───────────────────────────────────────────────────

# (template_stem, loader_builder, loss_name_if_supervised_else_None)
MARKER_TRAIN_TEMPLATES = [
    ("mnist_mlp",                 _build_mnist_loader,      "cross_entropy"),
    ("mnist_cnn",                 _build_mnist_loader,      "cross_entropy"),
    ("mnist_vae",                 _build_mnist_loader,      None),   # kind="loss"
    ("transfer_learning",         _build_cifar_loader,      "cross_entropy"),
    ("char_lm",                   _build_text_loader,       None),   # kind="loss"
    ("time_series_lstm",          _build_text_loader_short, None),   # kind="loss"
    ("multimodal_classification", _build_multimodal_loader, None),   # kind="loss"
    ("so101_imitation",           _build_so101_loader,      None),   # kind="loss"
]


def _find_train_target(graph):
    """Find the training target node.

    Returns (node_id, is_loss_output, target_column).
    """
    for nid, n in graph.nodes.items():
        if n.type_name == "pt_train_marker":
            kind = str(n.inputs["kind"].default_value or "logits")
            target = str(n.inputs["target"].default_value or "label")
            return nid, kind == "loss", target
    for nid, n in graph.nodes.items():
        if n.type_name == "pt_train_output":
            lio = bool(n.inputs["loss_is_output"].default_value)
            return nid, lio, "label"
    return None, False, "label"


def _prime_input_markers(graph, batch):
    for n in graph.nodes.values():
        if n.type_name != "pt_input_marker":
            continue
        modality = str(n.inputs["modality"].default_value or "x")
        n._probe_tensor = batch.get(modality) if isinstance(batch, dict) else None


@pytest.mark.parametrize(
    "stem,loader_builder,loss_name",
    MARKER_TRAIN_TEMPLATES,
    ids=[t[0] for t in MARKER_TRAIN_TEMPLATES],
)
def test_template_trains_one_step(stem, loader_builder, loss_name, tmp_path):
    from core.graph import Graph
    from core.graph_module import GraphAsModule
    from gui.mixins.training import _build_loss, _build_optimizer
    from templates import reload_template

    entry = reload_template(stem)
    assert entry is not None, f"template {stem!r} failed to import"
    _label, _desc, builder = entry

    g = Graph()
    builder(g)

    train_id, lio, target_col = _find_train_target(g)
    assert train_id, f"{stem}: no train target in graph"

    loader = loader_builder(tmp_path)
    probe = next(iter(loader))
    _prime_input_markers(g, probe)

    model = GraphAsModule(g, output_node_id=train_id, output_port="tensor_in")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params > 0, f"{stem}: GraphAsModule exposes zero trainable params"

    out = model(probe)
    assert out is not None, f"{stem}: GraphAsModule forward returned None"

    if lio:
        loss = out
    else:
        loss_fn = _build_loss(loss_name or "cross_entropy")
        label = probe.get(target_col) if isinstance(probe, dict) else probe[1]
        assert label is not None, f"{stem}: batch has no '{target_col}'"
        loss = loss_fn(out, label)

    assert torch.is_tensor(loss), f"{stem}: not a tensor ({type(loss)})"
    assert loss.dim() == 0, f"{stem}: not scalar (shape={tuple(loss.shape)})"
    assert math.isfinite(loss.item()), f"{stem}: not finite ({loss.item()})"

    opt = _build_optimizer("adam", model, 1e-3, 0.0, 0.9)
    opt.zero_grad()
    loss.backward()
    opt.step()

    any_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0
                   for p in model.parameters() if p.requires_grad)
    assert any_grad, f"{stem}: no parameter received a non-zero gradient"


# ── Build-only test for every template (training or not) ──────────────────

def _all_template_stems():
    tdir = Path(__file__).parent.parent / "templates"
    return sorted(
        p.stem for p in tdir.glob("*.py")
        if not p.name.startswith("_") and p.name != "__init__.py"
    )


@pytest.mark.parametrize("stem", _all_template_stems())
def test_template_builds(stem):
    from core.graph import Graph
    from templates import reload_template

    entry = reload_template(stem)
    assert entry is not None, f"template {stem!r} failed to import"
    _label, _desc, builder = entry
    g = Graph()
    builder(g)
    assert len(g.nodes) > 0, f"template {stem!r} built an empty graph"
