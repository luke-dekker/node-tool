"""Training loop executor — runs in a background thread, GUI-agnostic.

Formerly gui/training_panel.py. Lives in the pytorch plugin because it
imports torch and implements the SGD/backward/step loop. Any frontend drains
its event queue and displays results; the class itself has no GUI coupling.
"""
from __future__ import annotations
import threading
import queue
from typing import Any


class TrainingController:
    """Manages background training thread and UI state."""

    def __init__(self):
        self._cmd_q: queue.Queue = queue.Queue()
        self._evt_q: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self.status = "idle"
        self.current_epoch = 0
        self.total_epochs = 0
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.best_loss = float("inf")
        self.last_model_state: dict | None = None
        self.last_model: Any | None = None
        self.error_message: str | None = None
        # In-epoch progress (updated each batch_progress event between epochs)
        self.current_batch: int = 0
        self.last_batch_loss: float | None = None
        # Optional callback — called from the main thread after each epoch and on done.
        # Use this to re-execute the graph so tensor_out values reflect updated weights.
        self.on_epoch_end: Any | None = None

    # -- Public API --------------------------------------------------------

    def start(self, config: dict) -> None:
        if self.status == "running":
            return
        self.status = "running"
        self.current_epoch = 0
        self.total_epochs = config.get("epochs", 10)
        self.train_losses = []
        self.val_losses = []
        self.best_loss = float("inf")
        self.error_message = None
        self._thread = threading.Thread(target=self._worker, args=(config,), daemon=True)
        self._thread.start()

    def pause(self) -> None:
        if self.status == "running":
            self.status = "paused"
            self._cmd_q.put("pause")

    def resume(self) -> None:
        if self.status == "paused":
            self.status = "running"
            self._cmd_q.put("resume")

    def stop(self) -> None:
        self._cmd_q.put("stop")
        self.status = "stopped"

    def poll(self) -> list[str]:
        """Drain event queue and return log lines. Call from main thread each frame."""
        lines = []
        for _ in range(20):  # max 20 events per frame
            try:
                evt = self._evt_q.get_nowait()
            except queue.Empty:
                break
            evt_type = evt[0]
            if evt_type == "log":
                lines.append(evt[1])
                continue
            if evt_type == "batch_progress":
                _, epoch, batch_n, loss_val = evt
                self.current_epoch = epoch
                self.current_batch = batch_n
                self.last_batch_loss = loss_val
                continue
            if evt_type == "epoch_end":
                _, epoch, loss, val_loss, current_lr = evt
                self.current_epoch = epoch
                self.train_losses.append(loss)
                if loss < self.best_loss:
                    self.best_loss = loss
                val_str = ""
                if val_loss is not None:
                    self.val_losses.append(val_loss)
                    val_str = f"  val={val_loss:.6f}"
                lr_str = f"  lr={current_lr:.2e}" if current_lr is not None else ""
                lines.append(f"[Train] Epoch {epoch}/{self.total_epochs}  loss={loss:.6f}  best={self.best_loss:.6f}{val_str}{lr_str}")
                if self.on_epoch_end is not None:
                    try:
                        self.on_epoch_end()
                    except Exception:
                        pass
            elif evt_type == "done":
                _, state_dict, model_obj = evt
                self.last_model_state = state_dict
                self.last_model = model_obj
                self.status = "done"
                lines.append(f"[Train] Training complete. Best loss: {self.best_loss:.6f}")
                if self.on_epoch_end is not None:
                    try:
                        self.on_epoch_end()
                    except Exception:
                        pass
            elif evt_type == "error":
                _, msg = evt
                self.error_message = msg
                self.status = "error"
                lines.append(f"[Train] ERROR: {msg}")
        return lines

    # -- Worker (background thread - NO GUI calls here) --------------------

    def _worker(self, config: dict) -> None:
        try:
            import torch
            model = config["model"]
            optimizer = config["optimizer"]
            epochs = int(config.get("epochs", 10))
            # device field comes through as "cuda:0 (RTX 4070)" when the
            # panel enumerates real GPUs — keep just the torch-valid prefix.
            device_str = str(config.get("device", "cpu")).split()[0].strip()
            device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")

            if model is None or optimizer is None:
                self._evt_q.put(("error", "Training config has None values."))
                return

            # Build the task list. New-style configs have a "tasks" list;
            # legacy configs have a single dataloader/loss_fn/loss_is_output.
            tasks = config.get("tasks")
            if not tasks:
                tasks = [{
                    "dataloader":     config["dataloader"],
                    "loss_fn":        config.get("loss_fn"),
                    "loss_is_output": config.get("loss_is_output", False),
                    "target_id":      getattr(model, "output_node_id", None),
                    "task_name":      "default",
                }]
            if not any(t.get("dataloader") for t in tasks):
                self._evt_q.put(("error", "No dataloader found for any task."))
                return

            model = model.to(device)
            scheduler = config.get("scheduler")
            val_dataloader = config.get("val_dataloader")

            for epoch in range(1, epochs + 1):
                # Check for stop/pause
                try:
                    cmd = self._cmd_q.get_nowait()
                    if cmd == "stop":
                        return
                    if cmd == "pause":
                        while True:
                            cmd2 = self._cmd_q.get()
                            if cmd2 == "resume":
                                break
                            if cmd2 == "stop":
                                return
                except queue.Empty:
                    pass

                model.train()
                epoch_loss = 0.0
                batches = 0

                # Multi-task: iterate each task's dataloader within the epoch.
                # For single-task graphs this is just one iteration of one loader.
                for task in tasks:
                    task_dl  = task.get("dataloader")
                    task_lio = task.get("loss_is_output", False)
                    task_lfn = task.get("loss_fn")
                    task_tid = task.get("target_id")

                    if task_dl is None:
                        continue

                    # Re-target GraphAsModule to this task's TrainOutput
                    if task_tid is not None:
                        model.output_node_id = task_tid
                        model.output_port = "tensor_in"

                    for batch in task_dl:
                        try:
                            cmd = self._cmd_q.get_nowait()
                            if cmd == "stop":
                                return
                        except queue.Empty:
                            pass

                        # Move batch to device
                        if isinstance(batch, dict) and "data" in batch:
                            dev_data = {m: (v.to(device) if hasattr(v, "to") else v)
                                        for m, v in batch["data"].items()}
                            labels = batch.get("label")
                            if hasattr(labels, "to"):
                                labels = labels.to(device)
                            dev_batch = {"data": dev_data, "label": labels,
                                         "present": batch.get("present", [])}
                        elif isinstance(batch, dict):
                            dev_batch = {k: (v.to(device) if hasattr(v, "to") else v)
                                         for k, v in batch.items()}
                            labels = dev_batch.get("label")
                        elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                            dev_batch = (batch[0].to(device), batch[1].to(device))
                            labels = dev_batch[1]
                        else:
                            dev_batch = batch.to(device) if hasattr(batch, "to") else batch
                            labels = None

                        optimizer.zero_grad()
                        try:
                            out = model(dev_batch)
                        except Exception as exc:
                            # Surface the error instead of silently skipping.
                            # Autoresearch/mutation loops used to report
                            # "loss=0.0, training complete" when every batch
                            # actually failed forward — never again.
                            self._evt_q.put((
                                "error",
                                f"model forward failed: {type(exc).__name__}: {exc}",
                            ))
                            return
                        if out is None:
                            self._evt_q.put((
                                "error",
                                "model forward returned None — graph is broken "
                                "(likely a shape mismatch or missing wire).",
                            ))
                            return

                        if task_lio:
                            loss = out
                        else:
                            if labels is None or not hasattr(labels, "shape"):
                                continue
                            loss = task_lfn(out, labels) if task_lfn else out
                        loss.backward()
                        optimizer.step()
                        loss_val = loss.item()
                        epoch_loss += loss_val
                        batches += 1

                        # Per-batch progress — every N batches emit a log
                        # so the UI shows activity during long CPU epochs.
                        # Also pushed as an "epoch_step" event for live loss
                        # plot updates without waiting for epoch end.
                        if batches % 25 == 0:
                            self._evt_q.put((
                                "log",
                                f"[Train] Epoch {epoch}/{epochs} step {batches}  "
                                f"loss={loss_val:.6f}",
                            ))
                            self._evt_q.put(("batch_progress", epoch, batches, loss_val))

                avg_loss = epoch_loss / max(batches, 1)

                # Validation pass — iterate every task that has its own
                # `val_dataloader`. Each task may wire a different TrainMarker
                # (multi-A/B graphs), so we retarget the model per task just
                # like the training loop above. Per-task val losses are
                # averaged by total batch count across tasks so a task with
                # more val batches dominates proportionally — matching how the
                # training loss is reported.
                val_loss = None
                val_total = 0.0
                val_total_batches = 0
                per_task_val: list[tuple[str, float]] = []
                # Backward compat: a legacy `val_dataloader` at the config
                # root is treated as task[0]'s val loader if no task carries
                # one of its own.
                legacy_val = val_dataloader
                if legacy_val is not None and not any(t.get("val_dataloader") for t in tasks):
                    tasks[0] = {**tasks[0], "val_dataloader": legacy_val}

                if any(t.get("val_dataloader") for t in tasks):
                    model.eval()
                    with torch.no_grad():
                        for task in tasks:
                            tdl = task.get("val_dataloader")
                            if tdl is None:
                                continue
                            tid = task.get("target_id")
                            if tid is not None:
                                model.output_node_id = tid
                                model.output_port = "tensor_in"
                            t_lio = task.get("loss_is_output", False)
                            t_lfn = task.get("loss_fn")
                            t_sum = 0.0
                            t_batches = 0
                            for batch in tdl:
                                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                                    dev_b = (batch[0].to(device), batch[1].to(device))
                                    labels_v = dev_b[1]
                                elif isinstance(batch, dict) and "data" in batch:
                                    dev_data = {m: (v.to(device) if hasattr(v, "to") else v)
                                                for m, v in batch["data"].items()}
                                    labels_v = batch.get("label")
                                    if hasattr(labels_v, "to"):
                                        labels_v = labels_v.to(device)
                                    dev_b = {"data": dev_data, "label": labels_v}
                                elif isinstance(batch, dict):
                                    dev_b = {k: (v.to(device) if hasattr(v, "to") else v)
                                             for k, v in batch.items()}
                                    labels_v = dev_b.get("label")
                                else:
                                    dev_b = batch.to(device) if hasattr(batch, "to") else batch
                                    labels_v = None
                                out_v = model(dev_b)
                                if out_v is None:
                                    continue
                                if t_lio:
                                    t_sum += out_v.item()
                                elif t_lfn and labels_v is not None:
                                    t_sum += t_lfn(out_v, labels_v).item()
                                t_batches += 1
                            if t_batches > 0:
                                task_avg = t_sum / t_batches
                                per_task_val.append((task.get("task_name", "task"), task_avg))
                                val_total += t_sum
                                val_total_batches += t_batches
                    if val_total_batches > 0:
                        val_loss = val_total / val_total_batches
                        if len(per_task_val) > 1:
                            parts = ", ".join(f"{n}={v:.4f}" for n, v in per_task_val)
                            self._evt_q.put((
                                "log",
                                f"[Train] val breakdown: {parts}  mean={val_loss:.4f}",
                            ))

                # Scheduler step
                current_lr = None
                if scheduler is not None:
                    try:
                        from torch.optim.lr_scheduler import ReduceLROnPlateau
                        if isinstance(scheduler, ReduceLROnPlateau):
                            scheduler.step(val_loss if val_loss is not None else avg_loss)
                        else:
                            scheduler.step()
                        current_lr = scheduler.get_last_lr()[0]
                    except Exception:
                        pass

                self._evt_q.put(("epoch_end", epoch, avg_loss, val_loss, current_lr))

            # Save final state dict + full model
            state = {k: v.cpu() for k, v in model.state_dict().items()}
            self._evt_q.put(("done", state, model))

        except Exception:
            import traceback
            self._evt_q.put(("error", traceback.format_exc()))
