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


# ── Panel → torch object builders ───────────────────────────────────────
# These used to live in nodes/pytorch/training.py alongside the deleted
# TrainingConfigNode. They're panel glue, not nodes, so they live here now
# where their sole caller (this mixin) can see them directly.

def _build_optimizer(name: str, model, lr: float, weight_decay: float, momentum: float):
    """Construct a torch optimizer from a string name."""
    import torch.optim as optim
    if model is None:
        return None
    params = model.parameters()
    key = name.strip().lower().replace("_", "").replace(" ", "")
    if key == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if key == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    if key == "rmsprop":
        return optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    return optim.Adam(params, lr=lr, weight_decay=weight_decay)  # default: adam


def _build_loss(name: str):
    """Construct a torch loss function from a string name."""
    import torch.nn as nn
    key = name.strip().lower().replace("_", "").replace(" ", "").replace("-", "")
    return {
        "mse":            nn.MSELoss(),
        "bce":            nn.BCELoss(),
        "bcewithlogits":  nn.BCEWithLogitsLoss(),
        "l1":             nn.L1Loss(),
    }.get(key, nn.CrossEntropyLoss())  # default: crossentropy


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
        """Read training + per-group dataset config from the panel's DPG widgets."""
        def _get(tag, fallback):
            try:
                return dpg.get_value(tag)
            except Exception:
                return fallback

        groups = self._discover_groups()
        datasets = {}
        for g in groups:
            datasets[g] = {
                "path":       str(_get(f"train_ds_{g}_path", "")),
                "batch_size": max(1, int(_get(f"train_ds_{g}_batch", 32))),
                "split":      str(_get(f"train_ds_{g}_split", "train")),
                "seq_len":    max(0, int(_get(f"train_ds_{g}_seq", 0))),
                "chunk_size": max(1, int(_get(f"train_ds_{g}_chunk", 1))),
            }
        if not datasets:
            datasets["task_1"] = {
                "path": "", "batch_size": 32, "split": "train",
                "seq_len": 0, "chunk_size": 1,
            }

        return {
            "epochs":         max(1, int(_get("train_epochs_input", 10))),
            "lr":             float(_get("train_lr_input", 0.001)),
            "optimizer_name": str(_get("train_optimizer_combo", "adam")),
            "loss_name":      str(_get("train_loss_combo", "crossentropy")),
            "device":         str(_get("train_device_combo", "cpu")),
            "datasets":       datasets,
        }

    def _discover_groups(self) -> list[str]:
        """Return sorted unique group names from all markers in the graph."""
        groups = set()
        for n in self.graph.nodes.values():
            if n.type_name in ("pt_input_marker", "pt_train_marker"):
                groups.add(str(n.inputs["group"].default_value or "task_1"))
        return sorted(groups) or ["task_1"]

    def _rebuild_dataset_panel(self) -> None:
        """Scan graph for marker groups and rebuild the dataset config section.

        Called on Check Wiring, Start, and template load. Creates one row of
        widgets per group inside the "train_dataset_section" child_window.
        """
        container = "train_dataset_section"
        if not dpg.does_item_exist(container):
            return
        dpg.delete_item(container, children_only=True)

        groups = self._discover_groups()
        modalities: dict[str, list[str]] = {}
        for n in self.graph.nodes.values():
            if n.type_name == "pt_input_marker":
                g = str(n.inputs["group"].default_value or "task_1")
                m = str(n.inputs["modality"].default_value or "x")
                modalities.setdefault(g, []).append(m)

        iw = -1  # auto-width: fill the column
        for g in groups:
            cols = modalities.get(g, [])
            cols_str = ", ".join(cols) if cols else "?"
            with dpg.group(parent=container):
                dpg.add_text(f"[{g}] {cols_str}", color=[180, 180, 255])
                dpg.add_input_text(label="path", tag=f"train_ds_{g}_path",
                                   default_value="", width=iw,
                                   hint="mnist, cifar10, path...")
                with dpg.group(horizontal=True):
                    dpg.add_input_int(label="batch", tag=f"train_ds_{g}_batch",
                                      default_value=32, width=60, step=0, min_value=1)
                    dpg.add_combo(label="split", tag=f"train_ds_{g}_split",
                                  items=["train", "test", "val"],
                                  default_value="train", width=60)
                with dpg.group(horizontal=True):
                    dpg.add_input_int(label="seq", tag=f"train_ds_{g}_seq",
                                      default_value=0, width=60, step=0, min_value=0)
                    dpg.add_input_int(label="chunk", tag=f"train_ds_{g}_chunk",
                                      default_value=1, width=60, step=0, min_value=1)
                if g != groups[-1]:
                    dpg.add_separator()

        pass  # group auto-sizes to content

    def _has_markers(self) -> bool:
        """True if the graph uses the marker architecture (no DatasetNode)."""
        return any(n.type_name == "pt_input_marker"
                   for n in self.graph.nodes.values())

    def _build_loader_from_panel(self, ds_cfg: dict):
        """Build a DataLoader from a per-group dataset config dict.

        `ds_cfg` has keys: path, batch_size, split, seq_len, chunk_size.
        Reuses DatasetNode's format-detection + loading code without putting
        a DatasetNode in the graph. Returns (dataloader, batch_preview, info)
        or (None, None, error_string).
        """
        from nodes.pytorch._dataset_loader import DatasetNode
        tmp = DatasetNode()
        result = tmp.execute({
            "path":       ds_cfg["path"],
            "batch_size": ds_cfg["batch_size"],
            "split":      ds_cfg["split"],
            "seq_len":    ds_cfg["seq_len"],
            "chunk_size": ds_cfg["chunk_size"],
            "chunk_columns": "",
            "columns":    "",
            "shuffle":    True,
            "extra_outputs": "",
            "task_id":    "default",
        })
        loader = result.get("dataloader")
        info   = result.get("info", "")
        if loader is None:
            return None, None, info or "Failed to build dataloader"
        try:
            batch = next(iter(loader))
        except StopIteration:
            return None, None, "Dataset is empty"
        return loader, batch, info

    def _prime_markers(self, batch):
        """Set _probe_tensor on every InputMarker from a batch dict."""
        for n in self.graph.nodes.values():
            if n.type_name != "pt_input_marker":
                continue
            modality = str(n.inputs["modality"].default_value or "x")
            if isinstance(batch, dict) and modality in batch:
                n._probe_tensor = batch[modality]

    def _find_train_markers(self, outputs: dict) -> list[tuple[str, dict]]:
        """Find TrainMarker (B) nodes and return their configs."""
        results = []
        for node_id, node in self.graph.nodes.items():
            if node.type_name == "pt_train_marker" and node_id in outputs:
                cfg = outputs[node_id].get("config")
                if cfg is not None:
                    results.append((node_id, cfg))
        return results

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

        Two modes:
          - Marker mode (graph has InputMarker + TrainMarker): panel owns
            the dataset config; loader built from panel widgets.
          - Legacy mode (graph has DatasetNode + TrainOutputNode): loader
            discovered from the graph's DatasetNode output ports.
        """
        from core.graph_module import GraphAsModule
        import torch

        self._sync_inputs_from_widgets()
        panel = self._read_panel_config()
        loss_fn = _build_loss(panel["loss_name"])

        if self._has_markers():
            return self._train_start_markers(panel, loss_fn)
        return self._train_start_legacy(panel, loss_fn)

    def _train_start_markers(self, panel: dict, loss_fn) -> None:
        """Marker path: build loader per group from panel, prime markers, train."""
        from core.graph_module import GraphAsModule
        import torch

        groups = self._discover_groups()
        loaders: dict[str, object] = {}
        batches: dict[str, dict] = {}
        for g in groups:
            ds_cfg = panel["datasets"].get(g)
            if not ds_cfg or not ds_cfg["path"].strip():
                self._log(f"[Train] Set a dataset path for group '{g}' in the panel.")
                return
            self._log(f"[Train] Loading dataset for [{g}]...")
            loader, batch, info = self._build_loader_from_panel(ds_cfg)
            if loader is None:
                self._log(f"[Train] [{g}] Dataset error: {info}")
                return
            self._log(f"[Train] [{g}] {info}")
            loaders[g] = loader
            batches[g] = batch

        # Prime all markers from their group's batch
        for n in self.graph.nodes.values():
            if n.type_name != "pt_input_marker":
                continue
            g = str(n.inputs["group"].default_value or "task_1")
            m = str(n.inputs["modality"].default_value or "x")
            b = batches.get(g, {})
            if isinstance(b, dict) and m in b:
                n._probe_tensor = b[m]

        try:
            with torch.no_grad():
                outputs, _, _ = self.graph.execute()
        except Exception as exc:
            self._log(f"[Train] Graph error: {exc}")
            return

        train_markers = self._find_train_markers(outputs)
        if not train_markers:
            self._log("[Train] No Data Out (B) marker found.")
            return

        tasks = []
        for i, (target_id, cfg) in enumerate(train_markers):
            kind = cfg.get("kind", "logits")
            lio = kind == "loss"
            name = cfg.get("task_name", f"task_{i}")
            group = cfg.get("group", "task_1")
            dl = loaders.get(group, next(iter(loaders.values()), None))
            if dl is None:
                self._log(f"[Train] No loader for group {group!r}.")
                return
            tasks.append({
                "target_id":      target_id,
                "dataloader":     dl,
                "loss_is_output": lio,
                "loss_fn":        None if lio else loss_fn,
                "task_name":      name,
            })

        first_target = tasks[0]["target_id"]

        try:
            model = GraphAsModule(self.graph, output_node_id=first_target,
                                  output_port="tensor_in")
        except Exception as exc:
            self._log(f"[Train] Failed to wrap graph as module: {exc}")
            return
        n_params = sum(p.numel() for p in model.parameters())
        if n_params == 0:
            self._log("[Train] Graph has no trainable layers upstream of Data Out (B).")
            return

        optimizer = _build_optimizer(
            panel["optimizer_name"], model,
            panel["lr"], 0.0, 0.9,
        )

        full_config = {
            "model":          model,
            "optimizer":      optimizer,
            "tasks":          tasks,
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

    def _train_start_legacy(self, panel: dict, loss_fn) -> None:
        """Legacy path: discover dataloaders from DatasetNodes in the graph."""
        from core.graph_module import GraphAsModule
        import torch

        try:
            with torch.no_grad():
                outputs, _, _ = self.graph.execute()
        except Exception as exc:
            self._log(f"[Train] Graph error: {exc}")
            return

        train_outputs = self._find_train_outputs(outputs)
        if not train_outputs:
            self._log("[Train] No Train Output node found. Add one to mark the training target.")
            return

        discovered_loaders = self._find_dataloaders(outputs)
        loader_by_id: dict[str, object] = {}
        for task_id, loader in discovered_loaders:
            loader_by_id[task_id] = loader

        if not loader_by_id:
            for _, cfg in train_outputs:
                dl_legacy = cfg.get("dataloader")
                if dl_legacy is not None:
                    loader_by_id["default"] = dl_legacy
                    break
        if not loader_by_id:
            ds_present = False
            for node_id, node in self.graph.nodes.items():
                if node.type_name == "pt_dataset":
                    ds_present = True
                    info = outputs.get(node_id, {}).get("info", "") or ""
                    self._log(f"[Train] Dataset node returned no loader. "
                              f"Info: {info}")
            if not ds_present:
                self._log("[Train] No dataset node in the graph. Add one.")
            return

        tasks = []
        for i, (target_id, target_cfg) in enumerate(train_outputs):
            lio = target_cfg.get("loss_is_output", False)
            name = target_cfg.get("task_name", f"task_{i}")
            dl = loader_by_id.get(name)
            if dl is None:
                dl = loader_by_id.get("default")
            if dl is None:
                dl = next(iter(loader_by_id.values()), None)
            if dl is None:
                self._log(f"[Train] No dataloader found for task {name!r}.")
                continue
            tasks.append({
                "target_id":      target_id,
                "dataloader":     dl,
                "loss_is_output": lio,
                "loss_fn":        None if lio else loss_fn,
                "task_name":      name,
            })
        if not tasks:
            self._log("[Train] No valid tasks.")
            return

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

        self._rebuild_dataset_panel()

        self._sync_inputs_from_widgets()
        self._log("-" * 40)
        self._log("Checking training wiring...")
        try:
            with torch.no_grad():
                outputs, _, _ = self.graph.execute()
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
