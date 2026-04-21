"""Multi-A/B val pass — the executor must iterate every task's
`val_dataloader`, not just tasks[0]. A two-task graph with different
per-task val losses should surface both in the log and a batch-weighted
mean in `val_losses`.

Uses the TrainingController directly with a stubbed model so we can drive
multi-task forward without building real nn.Modules.
"""
from __future__ import annotations

import threading
import time

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from plugins.pytorch._training_executor import TrainingController


@pytest.fixture(autouse=True)
def _register_port_types():
    from plugins.agents.port_types import register_all
    register_all()


class _StubModel(torch.nn.Module):
    """Returns a tensor keyed on `output_node_id` so each task emits a
    distinguishable prediction. Has a single trainable param so
    `model.parameters()` is non-empty."""
    def __init__(self, target_to_bias: dict[str, float]):
        super().__init__()
        self._bias = torch.nn.Parameter(torch.zeros(1))
        self._targets = target_to_bias
        self.output_node_id: str | None = None
        self.output_port = "tensor_in"

    def forward(self, batch):
        if isinstance(batch, (list, tuple)) and len(batch) >= 1:
            x = batch[0]
        elif isinstance(batch, dict):
            x = batch.get("x") or next(iter(batch.values()))
        else:
            x = batch
        bias = self._targets.get(self.output_node_id, 0.0)
        return x + self._bias + bias


def _loader(n: int, batch_size: int, y_offset: float = 0.0) -> DataLoader:
    x = torch.zeros(n, 1)
    y = torch.full((n, 1), y_offset)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def _drain_until(ctrl: TrainingController, status: str, timeout: float = 5.0) -> list[str]:
    """Poll ctrl until it reaches `status`, returning every log string drained."""
    t0 = time.time()
    lines: list[str] = []
    while time.time() - t0 < timeout:
        lines.extend(ctrl.poll())
        if ctrl.status == status:
            lines.extend(ctrl.poll())
            return lines
        time.sleep(0.02)
    raise AssertionError(f"status never reached {status!r}; last={ctrl.status!r}")


def test_multi_group_val_iterates_every_task():
    """Two tasks, each with its own val_dataloader — executor should visit
    both and report their per-task val losses in the log stream, with the
    overall val_loss being the batch-weighted mean."""
    # Task A: predicts x+0; labels = 0 → val loss 0
    # Task B: predicts x+1; labels = 0 → val loss 1 (MSE against zero labels = 1)
    targets = {"t_a": 0.0, "t_b": 1.0}
    model = _StubModel(targets)
    loss_fn = torch.nn.MSELoss()

    tasks = [
        {"target_id": "t_a", "task_name": "task_a",
         "dataloader":     _loader(4, batch_size=2),
         "val_dataloader": _loader(4, batch_size=2),
         "loss_fn": loss_fn, "loss_is_output": False},
        {"target_id": "t_b", "task_name": "task_b",
         "dataloader":     _loader(4, batch_size=2),
         "val_dataloader": _loader(2, batch_size=1),
         "loss_fn": loss_fn, "loss_is_output": False},
    ]

    ctrl = TrainingController()
    ctrl.start({
        "model": model,
        "optimizer": torch.optim.SGD(model.parameters(), lr=0.0),  # no updates
        "tasks": tasks,
        "dataloader": tasks[0]["dataloader"],
        "loss_fn": loss_fn, "loss_is_output": False,
        "val_dataloader": None,  # no legacy loader — tasks carry their own
        "scheduler": None, "epochs": 1, "device": "cpu",
        "graph_module": False,
    })
    lines = _drain_until(ctrl, "done")

    # The val breakdown log fires when there's more than one task with val.
    breakdown = [l for l in lines if "val breakdown" in l]
    assert breakdown, f"expected a 'val breakdown' log; got {lines}"
    assert "task_a" in breakdown[0] and "task_b" in breakdown[0]

    # val_losses recorded exactly one per epoch.
    assert len(ctrl.val_losses) == 1
    # task_a val=0, task_b val=1. Task A has 2 batches, Task B has 2 batches.
    # Batch-weighted mean = (0*2 + 1*2) / 4 = 0.5.
    assert ctrl.val_losses[0] == pytest.approx(0.5, abs=0.05)


def test_legacy_val_dataloader_still_works_for_single_task():
    """Backward compat: a legacy config-level `val_dataloader` on a single-
    task setup continues to populate val_losses just like before."""
    model = _StubModel({"t_a": 0.5})
    loss_fn = torch.nn.MSELoss()

    tasks = [
        {"target_id": "t_a", "task_name": "task_a",
         "dataloader": _loader(4, batch_size=2),
         # No val_dataloader on the task — exercises the promotion path.
         "loss_fn": loss_fn, "loss_is_output": False},
    ]
    ctrl = TrainingController()
    ctrl.start({
        "model": model,
        "optimizer": torch.optim.SGD(model.parameters(), lr=0.0),
        "tasks": tasks,
        "dataloader": tasks[0]["dataloader"],
        "loss_fn": loss_fn, "loss_is_output": False,
        "val_dataloader": _loader(4, batch_size=2),  # legacy shape
        "scheduler": None, "epochs": 1, "device": "cpu",
        "graph_module": False,
    })
    _drain_until(ctrl, "done")
    assert len(ctrl.val_losses) == 1
    # bias=0.5, labels=0, MSE=(0.5)^2=0.25
    assert ctrl.val_losses[0] == pytest.approx(0.25, abs=0.05)
