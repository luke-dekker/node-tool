"""Training panel - background thread training with live loss plot."""
from __future__ import annotations
import threading
import queue
import time
from typing import Any

import dearpygui.dearpygui as dpg


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

    # -- Worker (background thread - NO DPG calls here) --------------------

    def _worker(self, config: dict) -> None:
        try:
            import torch
            model = config["model"]
            optimizer = config["optimizer"]
            loss_fn = config["loss_fn"]
            dataloader = config["dataloader"]
            val_dataloader = config.get("val_dataloader")
            scheduler = config.get("scheduler")
            epochs = int(config.get("epochs", 10))
            device_str = config.get("device", "cpu")
            device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")

            _is_vae_check = config.get("vae", False)
            if model is None or optimizer is None or dataloader is None:
                self._evt_q.put(("error", "Training config has None values - wire all inputs."))
                return
            if not _is_vae_check and loss_fn is None:
                self._evt_q.put(("error", "Training config has None values - wire all inputs."))
                return

            model = model.to(device)

            is_vae = config.get("vae", False)
            beta   = float(config.get("beta", 1.0))
            recon_loss_type = config.get("recon_loss_type", "mse")

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
                for batch in dataloader:
                    # Check for stop mid-epoch
                    try:
                        cmd = self._cmd_q.get_nowait()
                        if cmd == "stop":
                            return
                    except queue.Empty:
                        pass

                    if isinstance(batch, (list, tuple)):
                        x, y = batch[0].to(device), batch[1].to(device)
                    else:
                        x = batch.to(device)
                        y = x  # autoencoder-style fallback

                    optimizer.zero_grad()
                    if is_vae:
                        # VAE: model returns (recon, mu, log_var)
                        recon, mu, log_var = model(x)
                        if recon_loss_type == "bce":
                            import torch.nn.functional as F
                            recon_loss = F.binary_cross_entropy(recon, x, reduction="mean")
                        else:
                            recon_loss = ((recon - x) ** 2).mean()
                        kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                        loss = recon_loss + beta * kl
                    else:
                        out = model(x)
                        loss = loss_fn(out, y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    batches += 1

                avg_loss = epoch_loss / max(batches, 1)

                # Validation pass
                val_loss = None
                if val_dataloader is not None:
                    model.eval()
                    val_epoch_loss = 0.0
                    val_batches = 0
                    with torch.no_grad():
                        for batch in val_dataloader:
                            if isinstance(batch, (list, tuple)):
                                x, y = batch[0].to(device), batch[1].to(device)
                            else:
                                x = batch.to(device); y = x
                            if is_vae:
                                recon, mu, log_var = model(x)
                                if recon_loss_type == "bce":
                                    import torch.nn.functional as F
                                    r_loss = F.binary_cross_entropy(recon, x, reduction="mean")
                                else:
                                    r_loss = ((recon - x) ** 2).mean()
                                kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
                                val_epoch_loss += (r_loss + beta * kl).item()
                            else:
                                val_epoch_loss += loss_fn(model(x), y).item()
                            val_batches += 1
                    val_loss = val_epoch_loss / max(val_batches, 1)

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

        except Exception as e:
            import traceback
            self._evt_q.put(("error", traceback.format_exc()))
