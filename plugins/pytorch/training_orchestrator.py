"""TrainingOrchestrator — high-level, GUI-free training driver.

This is the business logic that used to live in gui/mixins/training.py,
extracted so every frontend talks to the same backend. DPG, React, Godot,
and the RPC server all drive the same orchestrator instance — no GUI
has to reinvent optimizer construction, marker priming, or epoch plumbing.

Responsibilities
  - Discover marker groups from the graph (for the dataset-config UI).
  - Build a DataLoader per group from panel config (path/batch/split/...).
  - Prime InputMarker nodes with batch tensors.
  - Discover TrainMarker targets and wrap the graph as an nn.Module.
  - Construct the optimizer + loss via the pytorch plugin factories.
  - Spawn the background TrainingController and drain its events.

What this class does NOT do
  - No widget tags, no dpg / Qt / React / Godot imports.
  - No decisions about layout — that's in core/panel PanelSpec.

RPC surface (call from any frontend — see `handle_rpc`):
  marker_groups()     → {groups: {...}}
  state()             → {status, epoch, total_epochs, best_loss, last_loss,
                         current_lr, error}
  losses()            → {series: {train: [...], val: [...]}}
  start(params)       → {ok, task_names} or {ok: False, error}
  pause()             → {status}
  resume()            → {status}
  stop()              → {status}
  save_model(path)    → {ok, path, n_params} or {ok: False, error}
  drain_logs()        → {lines: [...]}
"""
from __future__ import annotations
from typing import Any

from core.graph import Graph
from core.node import MarkerRole
from plugins.pytorch._factories import build_optimizer, build_loss
from plugins.pytorch._training_executor import TrainingController


class TrainingOrchestrator:
    """One orchestrator per app. Holds a reference to the live graph and a
    low-level TrainingController. Every frontend shares the same instance.
    """

    def __init__(self, graph: Graph):
        self.graph = graph
        self._ctrl = TrainingController()
        self._pending_logs: list[str] = []
        self._last_error: str | None = None

    # ── Accessors ────────────────────────────────────────────────────────

    def marker_groups(self) -> dict:
        """Group-keyed marker info for the dataset config panel."""
        groups: dict[str, dict] = {}
        for n in self.graph.nodes_by_role(MarkerRole.INPUT):
            g = str(n.inputs["group"].default_value or "task_1")
            m = str(n.inputs["modality"].default_value or "x")
            groups.setdefault(g, {"modalities": [], "has_output": False})
            if m not in groups[g]["modalities"]:
                groups[g]["modalities"].append(m)
        for n in self.graph.nodes_by_role(MarkerRole.TRAIN_TARGET):
            g = str(n.inputs["group"].default_value or "task_1")
            groups.setdefault(g, {"modalities": [], "has_output": False})
            groups[g]["has_output"] = True
        return groups

    def state(self) -> dict:
        """Live snapshot for the status section."""
        c = self._ctrl
        # Drain event queue into our log buffer so callers get up-to-date
        # status plus can read recent log lines via drain_logs().
        self._pending_logs.extend(c.poll())
        # Include in-epoch batch progress so the UI shows activity during
        # long epochs instead of staring at "Epoch 0 / 1" for minutes.
        if c.current_batch:
            epoch_str = f"{c.current_epoch} / {c.total_epochs}  step {c.current_batch}"
        else:
            epoch_str = f"{c.current_epoch} / {c.total_epochs}"
        best = f"{c.best_loss:.6f}" if c.best_loss != float("inf") else "—"
        if c.train_losses:
            last = f"{c.train_losses[-1]:.6f}"
        elif c.last_batch_loss is not None:
            last = f"{c.last_batch_loss:.6f}"
        else:
            last = "—"
        status_label = {
            "idle":     "Idle",
            "running":  "Running",
            "paused":   "Paused",
            "stopped":  "Stopped",
            "done":     "Done",
            "error":    "Error",
        }.get(c.status, c.status.capitalize())
        return {
            "status":       status_label,
            "epoch_str":    epoch_str,
            "best_loss":    best,
            "last_loss":    last,
            "error":        c.error_message or "",
        }

    def losses(self) -> dict:
        """Series data for the loss plot. Drain events so plot keeps up."""
        self._pending_logs.extend(self._ctrl.poll())
        return {
            "series": {
                "train": list(self._ctrl.train_losses),
                "val":   list(self._ctrl.val_losses),
            },
        }

    def drain_logs(self) -> dict:
        """Pop accumulated log lines — the frontend appends them to its terminal."""
        lines = self._pending_logs
        self._pending_logs = []
        return {"lines": lines}

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self, params: dict) -> dict:
        """Kick off training.

        `params` is the flat dict every GUI assembles from its panel:
          epochs, lr, optimizer (name), loss (name), device,
          datasets: {group: {path, batch_size, split, seq_len, chunk_size}}
        """
        if self._ctrl.status == "running":
            return {"ok": False, "error": "Already running"}
        try:
            return self._start_marker_path(params)
        except Exception as exc:
            self._last_error = str(exc)
            self._pending_logs.append(f"[Train] {exc}")
            return {"ok": False, "error": str(exc)}

    def pause(self, _params: dict | None = None) -> dict:
        self._ctrl.pause()
        return {"status": self._ctrl.status}

    def resume(self, _params: dict | None = None) -> dict:
        self._ctrl.resume()
        return {"status": self._ctrl.status}

    def stop(self, _params: dict | None = None) -> dict:
        self._ctrl.stop()
        return {"status": self._ctrl.status}

    def save_model(self, params: dict) -> dict:
        path = str(params.get("path", "")).strip()
        if not path:
            return {"ok": False, "error": "path is empty"}
        model = self._ctrl.last_model
        if model is None:
            return {"ok": False, "error": "No trained model to save"}
        try:
            import torch
            torch.save(model, path)
            n_params = sum(p.numel() for p in model.parameters())
            self._pending_logs.append(
                f"[Train] Model saved → {path}  ({n_params:,} params)"
            )
            return {"ok": True, "path": path, "n_params": n_params}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    # ── Uniform RPC entry point ─────────────────────────────────────────

    def handle_rpc(self, method: str, params: dict | None = None) -> Any:
        """Map method names from PanelSpec action ids / status source_rpc into
        orchestrator calls. Raises ValueError for unknown methods."""
        params = params or {}
        handlers = {
            "train_start":          self.start,
            "train_pause":          self.pause,
            "train_resume":         self.resume,
            "train_stop":           self.stop,
            "train_save_model":     self.save_model,
            "get_training_state":   lambda _p: self.state(),
            "get_training_losses":  lambda _p: self.losses(),
            "get_marker_groups":    lambda _p: {"groups": self.marker_groups()},
            "drain_training_logs":  lambda _p: self.drain_logs(),
        }
        handler = handlers.get(method)
        if handler is None:
            raise ValueError(f"Unknown training RPC method: {method}")
        return handler(params)

    # ── Private: actually start training ─────────────────────────────────

    def _start_marker_path(self, params: dict) -> dict:
        from plugins.pytorch.graph_module import GraphAsModule
        from nodes.pytorch._dataset_loader import DatasetNode
        import torch

        datasets = params.get("datasets", {}) or {}
        loss_fn  = build_loss(str(params.get("loss", "crossentropy")))
        groups   = sorted(self.marker_groups().keys())
        if not groups:
            return {"ok": False, "error": "No training markers in the graph"}

        # Build a DataLoader per group from the panel's per-group config.
        loaders: dict[str, Any] = {}
        batches: dict[str, Any] = {}
        for g in groups:
            cfg = datasets.get(g) or {}
            path = str(cfg.get("path", "")).strip()
            if not path:
                return {"ok": False, "error": f"Set a dataset path for group '{g}'"}
            tmp = DatasetNode()
            result = tmp.execute({
                "path":          path,
                "batch_size":    max(1, int(cfg.get("batch_size", 32))),
                "split":         str(cfg.get("split", "train")),
                "seq_len":       max(0, int(cfg.get("seq_len", 0))),
                "chunk_size":    max(1, int(cfg.get("chunk_size", 1))),
                "chunk_columns": "",
                "columns":       "",
                "shuffle":       True,
                "extra_outputs": "",
                "task_id":       g,
            })
            dl = result.get("dataloader")
            info = result.get("info", "") or ""
            if dl is None:
                return {"ok": False, "error": f"[{g}] {info or 'Failed to build dataloader'}"}
            self._pending_logs.append(f"[Train] [{g}] {info}")
            try:
                batch = next(iter(dl))
            except StopIteration:
                return {"ok": False, "error": f"[{g}] Dataset is empty"}
            loaders[g] = dl
            batches[g] = batch

        # Prime markers from their group's batch so a priming graph execute
        # has tensors to flow through downstream nodes.
        for n in self.graph.nodes_by_role(MarkerRole.INPUT):
            g = str(n.inputs["group"].default_value or "task_1")
            m = str(n.inputs["modality"].default_value or "x")
            b = batches.get(g, {})
            if isinstance(b, dict) and m in b:
                n._probe_tensor = b[m]

        try:
            with torch.no_grad():
                outputs, _, _ = self.graph.execute()
        except Exception as exc:
            return {"ok": False, "error": f"Graph error: {exc}"}

        # Build task list from TrainMarker targets
        tasks = []
        for target in self.graph.nodes_by_role(MarkerRole.TRAIN_TARGET):
            if target.id not in outputs:
                continue
            cfg = outputs[target.id].get("config")
            if cfg is None:
                continue
            kind = cfg.get("kind", "logits")
            lio  = kind == "loss"
            name = cfg.get("task_name", "task")
            group = cfg.get("group", "task_1")
            dl = loaders.get(group) or next(iter(loaders.values()), None)
            if dl is None:
                return {"ok": False, "error": f"No loader for group {group!r}"}
            tasks.append({
                "target_id":      target.id,
                "dataloader":     dl,
                "loss_is_output": lio,
                "loss_fn":        None if lio else loss_fn,
                "task_name":      name,
            })
        if not tasks:
            return {"ok": False, "error": "No Data Out (B) marker found"}

        first_target = tasks[0]["target_id"]
        try:
            model = GraphAsModule(self.graph, output_node_id=first_target,
                                  output_port="tensor_in")
        except Exception as exc:
            return {"ok": False, "error": f"Failed to wrap graph as module: {exc}"}
        n_params = sum(p.numel() for p in model.parameters())
        if n_params == 0:
            return {"ok": False, "error": "Graph has no trainable layers upstream of Data Out"}

        optimizer = build_optimizer(
            str(params.get("optimizer", "adam")), model,
            float(params.get("lr", 0.001)), 0.0, 0.9,
        )
        epochs = max(1, int(params.get("epochs", 10)))
        device = str(params.get("device", "cpu"))

        self._pending_logs.append(
            f"[Train] Starting: {epochs} epochs on {device}  "
            f"({n_params:,} params, tasks={[t['task_name'] for t in tasks]})"
        )

        self._ctrl.start({
            "model":          model,
            "optimizer":      optimizer,
            "tasks":          tasks,
            "loss_fn":        tasks[0].get("loss_fn"),
            "loss_is_output": tasks[0].get("loss_is_output", False),
            "dataloader":     tasks[0]["dataloader"],
            "val_dataloader": None,
            "scheduler":      None,
            "epochs":         epochs,
            "device":         device,
            "graph_module":   True,
        })
        return {
            "ok":         True,
            "task_names": [t["task_name"] for t in tasks],
            "n_params":   n_params,
        }
