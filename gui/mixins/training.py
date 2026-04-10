"""TrainingMixin — PyTorch model training controls."""
from __future__ import annotations
import dearpygui.dearpygui as dpg

from gui.theme import OK_GREEN, WARN_AMBER, ERR_RED, TEXT_DIM, ACCENT

_STATUS_COLORS = {
    "Idle":     TEXT_DIM,
    "Running":  OK_GREEN,
    "Paused":   WARN_AMBER,
    "Stopped":  TEXT_DIM,
    "Done":     ACCENT,
    "Error":    ERR_RED,
}


def _set_train_status(label: str) -> None:
    """Update the training status text + colored dot in the Training panel."""
    color = list(_STATUS_COLORS.get(label, TEXT_DIM))
    try:
        if dpg.does_item_exist("train_status_text"):
            dpg.set_value("train_status_text", label)
            dpg.configure_item("train_status_text", color=color)
        if dpg.does_item_exist("train_status_dot"):
            dpg.configure_item("train_status_dot", color=color)
    except Exception:
        pass


class TrainingMixin:
    """Methods for collecting model layers, starting/stopping training, and saving the model."""

    # ── Panel widget readers ──────────────────────────────────────────────

    def _read_panel_config(self) -> dict:
        """Read training hyperparameters from the panel's DPG widgets.

        These used to live on TrainingConfigNode; now they're persistent panel
        widgets. The graph only declares WHAT to optimize (via TrainOutputNode);
        the panel declares HOW.
        """
        def _get(tag, fallback):
            try:
                return dpg.get_value(tag)
            except Exception:
                return fallback

        return {
            "epochs":         max(1, int(_get("train_epochs_input", 10))),
            "lr":             float(_get("train_lr_input", 0.001)),
            "optimizer_name": str(_get("train_optimizer_combo", "adam")),
            "loss_name":      str(_get("train_loss_combo", "crossentropy")),
            "device":         str(_get("train_device_combo", "cpu")),
        }

    def _find_train_outputs(self, outputs: dict) -> list[tuple[str, dict]]:
        """Find all TrainOutputNode(s) in the graph that produced output."""
        results = []
        for node_id, node in self.graph.nodes.items():
            if node.type_name in ("pt_train_output", "pt_training_config") \
                    and node_id in outputs:
                cfg = outputs[node_id].get("config")
                if cfg is not None:
                    results.append((node_id, cfg))
        return results

    def _find_dataloaders(self, outputs: dict) -> list[tuple[str, object]]:
        """Auto-discover DataLoader objects from graph node outputs.

        Returns a list of (task_id, dataloader) tuples. The task_id comes from
        the dataset node's `task_id` input port; defaults to "default" if not
        set. The training mixin matches this against TrainOutput.task_name to
        pair datasets with training targets — no cross-canvas wires needed.
        """
        from core.node import PortType
        results: list[tuple[str, object]] = []
        for node_id, node in self.graph.nodes.items():
            has_dl = False
            for port_name, port in node.outputs.items():
                if port.port_type == PortType.DATALOADER:
                    val = outputs.get(node_id, {}).get(port_name)
                    if val is not None:
                        task_id = str(node.inputs.get("task_id", None)
                                      and node.inputs["task_id"].default_value or "default")
                        # Read task_id from the execute inputs if connected upstream
                        node_outs = outputs.get(node_id, {})
                        # But task_id is an INPUT, not an output. Read the port default
                        # which gets synced from the widget via _sync_inputs_from_widgets.
                        if "task_id" in node.inputs:
                            task_id = str(node.inputs["task_id"].default_value or "default")
                        results.append((task_id, val))
                        has_dl = True
                        break
        return results

    # ── Training start ─────────────────────────────────────────────────────

    def _train_start(self) -> None:
        """Wrap the graph as an nn.Module and start training.

        New architecture: training hyperparameters come from panel widgets,
        not from a TrainingConfigNode. The graph declares WHAT to optimize
        (via TrainOutputNode); the panel declares HOW.

        Flow:
          1. Run the graph once (live preview) to lazy-init all layer modules
          2. Find TrainOutputNode(s) + auto-discover DataLoader(s)
          3. Read training params from panel widgets
          4. Wrap graph as GraphAsModule targeting the TrainOutput's tensor_in
          5. Hand off to the background training thread
        """
        from nodes.pytorch.training import _build_optimizer, _build_scheduler
        from core.graph_module import GraphAsModule
        import torch

        self._sync_inputs_from_widgets()
        try:
            with torch.no_grad():
                outputs, _ = self.graph.execute()
        except Exception as exc:
            self._log(f"[Train] Graph error: {exc}")
            return

        # Find the training target (TrainOutputNode or legacy TrainingConfigNode)
        train_outputs = self._find_train_outputs(outputs)
        if not train_outputs:
            self._log("[Train] No Train Output node found. Add one to mark the training target.")
            return

        # Build the task list by matching dataset.task_id ↔ TrainOutput.task_name.
        # For single-task (both default to "default"), this auto-pairs trivially.
        # For multi-task, the user types the same label on both the dataset node
        # and the TrainOutput node — explicit, no wires, no guessing.
        discovered_loaders = self._find_dataloaders(outputs)  # [(task_id, loader)]
        loader_by_id: dict[str, object] = {}
        for task_id, loader in discovered_loaders:
            loader_by_id[task_id] = loader

        # Legacy fallback: if a TrainOutput/TrainingConfig had a dataloader
        if not loader_by_id:
            for _, cfg in train_outputs:
                dl_legacy = cfg.get("dataloader")
                if dl_legacy is not None:
                    loader_by_id["default"] = dl_legacy
                    break
        if not loader_by_id:
            self._log("[Train] No dataloader found. Add a dataset node to the graph.")
            return

        from nodes.pytorch.training import _build_loss
        panel = self._read_panel_config()
        loss_fn = _build_loss(panel["loss_name"])

        tasks = []
        for i, (target_id, target_cfg) in enumerate(train_outputs):
            lio = target_cfg.get("loss_is_output", False)
            name = target_cfg.get("task_name", f"task_{i}")
            # Match by label: find the dataloader whose task_id matches this
            # TrainOutput's task_name. Fall back to "default" then to first found.
            dl = loader_by_id.get(name)
            if dl is None:
                dl = loader_by_id.get("default")
            if dl is None:
                dl = next(iter(loader_by_id.values()), None)
            if dl is None:
                self._log(f"[Train] No dataloader found for task {name!r}. "
                          f"Set task_id on a dataset node to match.")
                continue
            tasks.append({
                "target_id":      target_id,
                "dataloader":     dl,
                "loss_is_output": lio,
                "loss_fn":        None if lio else loss_fn,
                "task_name":      name,
            })
        if not tasks:
            self._log("[Train] No valid tasks. Check task_id/task_name matching.")
            return

        # Use the first TrainOutput for the initial GraphAsModule target;
        # the training loop re-targets per task by updating model.output_node_id.
        first_target = tasks[0]["target_id"]

        try:
            model = GraphAsModule(self.graph, output_node_id=first_target,
                                  output_port="tensor_in")
        except Exception as exc:
            self._log(f"[Train] Failed to wrap graph as module: {exc}")
            return
        n_params = sum(p.numel() for p in model.parameters())
        if n_params == 0:
            self._log("[Train] Graph has no trainable layers upstream of Train Output.")
            return

        optimizer = _build_optimizer(
            panel["optimizer_name"], model,
            panel["lr"], 0.0, 0.9,
        )

        full_config = {
            "model":          model,
            "optimizer":      optimizer,
            "tasks":          tasks,
            # Legacy single-task fields for backward compat with training_panel
            "loss_fn":        tasks[0].get("loss_fn"),
            "loss_is_output": tasks[0].get("loss_is_output", False),
            "dataloader":     tasks[0]["dataloader"],
            "val_dataloader": None,
            "scheduler":      None,
            "epochs":         panel["epochs"],
            "device":         panel["device"],
            "multimodal":     False,
            "freeze_strategy": "hard",
            "log_modalities":  False,
            "graph_module":    True,
        }
        task_names = [t["task_name"] for t in tasks]
        self._log(f"[Train] Starting: {panel['epochs']} epochs on {panel['device']}  "
                  f"({n_params:,} params, tasks={task_names})")
        self._training_ctrl.on_epoch_end = self.refresh_graph_silent
        self._training_ctrl.start(full_config)
        _set_train_status("Running")

    def _train_check_wiring(self) -> None:
        """Run the graph and report training wiring status."""
        from core.graph_module import GraphAsModule
        import torch

        self._sync_inputs_from_widgets()
        self._log("-" * 40)
        self._log("Checking training wiring...")
        try:
            with torch.no_grad():
                outputs, _ = self.graph.execute()
        except Exception as exc:
            self._log(f"[Check] Graph error: {exc}")
            return

        # Find targets
        train_outputs = self._find_train_outputs(outputs)
        if not train_outputs:
            self._log("[Check] [x] No Train Output node found.")
            return
        self._log(f"[Check] [ok] {len(train_outputs)} Train Output(s) found")
        for nid, cfg in train_outputs:
            name = cfg.get("task_name", "?")
            lio  = cfg.get("loss_is_output", False)
            self._log(f"[Check]   - {name} (loss_is_output={lio})")

        # First target for the model check
        target_node_id = train_outputs[0][0]
        try:
            model = GraphAsModule(self.graph, output_node_id=target_node_id,
                                  output_port="tensor_in")
            n_params = sum(p.numel() for p in model.parameters())
            n_modules = len(model._layer_modules)
            if n_params == 0:
                self._log("[Check] [x] No trainable layers in the graph.")
            else:
                self._log(f"[Check] [ok] Graph model: {n_modules} layer nodes, "
                          f"{n_params:,} parameters")
        except Exception as exc:
            self._log(f"[Check] [x] Failed to wrap graph: {exc}")
            return

        # Dataloaders
        dataloaders = self._find_dataloaders(outputs)
        if dataloaders:
            self._log(f"[Check] [ok] {len(dataloaders)} dataloader(s) found")
        else:
            self._log("[Check] [x] No dataloader found in the graph")

        panel = self._read_panel_config()
        self._log(f"[Check]   optimizer: {panel['optimizer_name']}  lr={panel['lr']}")
        self._log(f"[Check]   epochs: {panel['epochs']}  device: {panel['device']}")
        if dataloaders:
            self._log("[Check] Ready to train!")

    def _train_pause_resume(self) -> None:
        if self._training_ctrl.status == "running":
            self._training_ctrl.pause()
            _set_train_status("Paused")
        elif self._training_ctrl.status == "paused":
            self._training_ctrl.resume()
            _set_train_status("Running")

    def _train_stop(self) -> None:
        self._training_ctrl.stop()
        _set_train_status("Stopped")

    def _save_model(self) -> None:
        model = self._training_ctrl.last_model
        if model is None:
            self._log("[Train] No trained model to save.")
            return
        import tkinter as tk
        from tkinter import filedialog
        try:
            root = tk.Tk()
            root.withdraw()
            path = filedialog.asksaveasfilename(
                title="Save Model As",
                defaultextension=".pt",
                filetypes=[("PyTorch model", "*.pt"), ("All files", "*.*")],
            )
            root.destroy()
            if not path:
                return
            import torch
            torch.save(model, path)
            total = sum(p.numel() for p in model.parameters())
            self._log(f"[Train] Model saved -> {path}  ({total:,} params)")
            self._log("[Train] Load it on any canvas with the Pretrained Block node.")
        except Exception as e:
            self._log(f"[Train] Save failed: {e}")
