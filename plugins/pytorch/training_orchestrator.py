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
        # Cached copy of the last successful `train_start` params — used by
        # autoresearch so it can re-fire training with the same dataset
        # config the user set up manually, substituting only its own epoch
        # budget and group. See `get_training_last_params` RPC.
        self._last_params: dict | None = None

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
            result = self._start_marker_path(params)
        except Exception as exc:
            self._last_error = str(exc)
            self._pending_logs.append(f"[Train] {exc}")
            return {"ok": False, "error": str(exc)}
        # Cache the params on success so autoresearch (or any future driver)
        # can re-fire training with the same dataset/optimizer/loss setup.
        if isinstance(result, dict) and result.get("ok"):
            import copy
            self._last_params = copy.deepcopy(params)
        return result

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
            "train_start":              self.start,
            "train_pause":              self.pause,
            "train_resume":             self.resume,
            "train_stop":               self.stop,
            "train_save_model":         self.save_model,
            "get_training_state":       lambda _p: self.state(),
            "get_training_losses":      lambda _p: self.losses(),
            "get_marker_groups":        lambda _p: {"groups": self.marker_groups()},
            "drain_training_logs":      lambda _p: self.drain_logs(),
            "get_training_last_params":    lambda _p: {
                "params": (__import__("copy").deepcopy(self._last_params)
                           if self._last_params is not None else None)
            },
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

        datasets_override = params.get("datasets", {}) or {}
        groups = sorted(self.marker_groups().keys())
        if not groups:
            return {"ok": False, "error": "No training markers in the graph"}

        # Resolve the primary B marker — the source of shared optimization
        # config (lr, optimizer, loss, epochs). Single-B graphs auto-promote
        # the only marker; multi-B graphs require exactly one explicit primary.
        b_markers = list(self.graph.nodes_by_role(MarkerRole.TRAIN_TARGET))
        primary_b, primary_err = _resolve_primary(b_markers)
        if primary_err:
            return {"ok": False, "error": primary_err}

        # Shared optimization config: marker default → param override → hard
        # default, in that order. The agent or panel can write into either.
        shared_loss     = _resolve_param("loss",      primary_b, params, "crossentropy", str)
        shared_optim    = _resolve_param("optimizer", primary_b, params, "adam",         str)
        shared_lr       = _resolve_param("lr",        primary_b, params, 0.001,          float)
        shared_epochs   = _resolve_param("epochs",    primary_b, params, 10,             int)
        device          = str(params.get("device", "cpu"))
        loss_fn         = build_loss(shared_loss)

        # Build a DataLoader per group. For each group, pull dataset config
        # from the first A marker in that group (markers in a group share
        # one dataset; modality picks which column each marker consumes).
        # Panel `datasets[group][field]` overrides marker defaults — same
        # priority chain as optimization params.
        loaders: dict[str, Any] = {}
        val_loaders: dict[str, Any] = {}
        batches: dict[str, Any] = {}
        a_by_group = _first_a_per_group(self.graph)
        for g in groups:
            a = a_by_group.get(g)
            override = datasets_override.get(g) or {}
            path = _resolve_param("path",         a, {"path": override.get("path")},
                                  "", str).strip()
            if not path:
                return {"ok": False, "error":
                        f"Set a dataset path for group '{g}' (on the Data In (A) "
                        f"marker or in the Training panel)."}
            batch_size = max(1, _resolve_param("batch_size",
                                               a, {"batch_size": override.get("batch_size")},
                                               32, int))
            split      = _resolve_param("split",   a, {"split": override.get("split")},
                                        "train", str)
            seq_len    = max(0, _resolve_param("seq_len",
                                               a, {"seq_len": override.get("seq_len")},
                                               0, int))
            chunk_size = max(1, _resolve_param("chunk_size",
                                               a, {"chunk_size": override.get("chunk_size")},
                                               1, int))
            val_fraction = max(0.0, min(0.5, _resolve_param(
                "val_fraction", a,
                {"val_fraction": override.get("val_fraction")},
                0.0, float,
            )))

            tmp = DatasetNode()
            result = tmp.execute({
                "path":          path,
                "batch_size":    batch_size,
                "split":         split,
                "seq_len":       seq_len,
                "chunk_size":    chunk_size,
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

            if val_fraction > 0.0:
                split_dl, split_val_dl, split_info = _split_train_val(
                    dl, batch_size, val_fraction,
                )
                if split_info:
                    self._pending_logs.append(f"[Train] [{g}] {split_info}")
                if split_dl is not None and split_val_dl is not None:
                    dl = split_dl
                    val_loaders[g] = split_val_dl

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
                "val_dataloader": val_loaders.get(group),
                "loss_is_output": lio,
                "loss_fn":        None if lio else loss_fn,
                "task_name":      name,
                "group":          group,
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

        optimizer = build_optimizer(shared_optim, model, shared_lr, 0.0, 0.9)
        epochs = max(1, shared_epochs)

        self._pending_logs.append(
            f"[Train] Starting: {epochs} epochs on {device}  "
            f"({n_params:,} params, tasks={[t['task_name'] for t in tasks]}, "
            f"optimizer={shared_optim}, lr={shared_lr})"
        )

        # The training controller validates on one task today — use the
        # first task's val loader if the user enabled val on that group.
        # Multi-group validation is tracked for later; val_loaders per
        # group are already stashed on each task dict above.
        self._ctrl.start({
            "model":          model,
            "optimizer":      optimizer,
            "tasks":          tasks,
            "loss_fn":        tasks[0].get("loss_fn"),
            "loss_is_output": tasks[0].get("loss_is_output", False),
            "dataloader":     tasks[0]["dataloader"],
            "val_dataloader": tasks[0].get("val_dataloader"),
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


# ── Module-level helpers ────────────────────────────────────────────────────

_VAL_SPLIT_SEED = 0


def _safe_float(value: Any, default: float) -> float:
    """Coerce panel field values (sometimes strings) to float; fall back
    on any parse failure."""
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _resolve_param(field: str, marker, override: dict, hard_default, cast):
    """Pull `field` with priority: panel/RPC override > marker default > hard_default.

    `marker` may be None (no A marker for the group, or no primary B). The
    override dict's `field` key is treated as missing if its value is None
    or the empty string — we want a blank panel field to fall through to
    the marker default, not stomp it.
    """
    val = override.get(field) if isinstance(override, dict) else None
    if val is None or (isinstance(val, str) and not val.strip()):
        if marker is not None and field in marker.inputs:
            val = marker.inputs[field].default_value
    if val is None or (isinstance(val, str) and not val.strip()):
        val = hard_default
    try:
        return cast(val)
    except (TypeError, ValueError):
        return hard_default


def _first_a_per_group(graph) -> dict:
    """Return {group_name: first A marker node in that group}."""
    out: dict = {}
    for n in graph.nodes_by_role(MarkerRole.INPUT):
        g = str(n.inputs["group"].default_value or "task_1")
        out.setdefault(g, n)
    return out


def _resolve_primary(b_markers: list) -> tuple:
    """Find the primary B marker.

    Rules:
      - Exactly one B → auto-promoted to primary.
      - Multiple Bs → exactly one must have `primary=True`.
      - Zero Bs → caller handles the empty case separately.

    Returns (primary_marker, error_message). On success error_message is "".
    """
    if not b_markers:
        return None, ""
    if len(b_markers) == 1:
        return b_markers[0], ""
    primaries = [m for m in b_markers
                 if bool(m.inputs.get("primary").default_value)
                 if "primary" in m.inputs]
    if len(primaries) == 1:
        return primaries[0], ""
    if not primaries:
        return None, (
            f"{len(b_markers)} B markers in graph but none flagged primary=True. "
            f"Set primary=True on the marker whose lr/optimizer/loss/epochs "
            f"should drive the shared optimizer."
        )
    return None, (
        f"{len(primaries)} B markers flagged primary=True; exactly one allowed."
    )


def _split_train_val(dl: Any, batch_size: int, val_fraction: float):
    """Split `dl`'s underlying dataset into (train, val) Subsets and wrap
    each in a fresh DataLoader that preserves the original collate_fn.

    Returns (train_loader, val_loader, info_string). Either loader may be
    None (plus an info describing why) for iterable datasets or datasets
    too small to split — callers fall back to the un-split loader.
    """
    import torch
    from torch.utils.data import DataLoader, Subset

    dataset = getattr(dl, "dataset", None)
    if dataset is None:
        return None, None, "val split skipped: loader has no .dataset"
    try:
        n = len(dataset)
    except TypeError:
        return None, None, "val split skipped: iterable dataset (no len)"
    if n < 2:
        return None, None, f"val split skipped: dataset too small ({n} samples)"

    n_val = max(1, int(round(val_fraction * n)))
    n_val = min(n_val, n - 1)
    n_train = n - n_val
    gen = torch.Generator().manual_seed(_VAL_SPLIT_SEED)
    perm = torch.randperm(n, generator=gen).tolist()
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    collate_fn = getattr(dl, "collate_fn", None)
    train_loader = DataLoader(train_subset, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_subset, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_fn)
    info = f"val split: {n_train} train / {n_val} val  (seed={_VAL_SPLIT_SEED})"
    return train_loader, val_loader, info
