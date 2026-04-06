"""TrainingMixin — PyTorch model training controls."""
from __future__ import annotations
import dearpygui.dearpygui as dpg


class TrainingMixin:
    """Methods for collecting model layers, starting/stopping training, and saving the model."""

    def _collect_model_layers(self, cfg_node_id: str):
        """Walk tensor_in connections backwards from Training Config to collect
        layer nodes in forward order. Returns nn.Sequential or None."""
        import torch.nn as nn
        conn_map = {(c.to_node_id, c.to_port): (c.from_node_id, c.from_port)
                    for c in self.graph.connections}
        ordered_nodes = []
        current_id   = cfg_node_id
        current_port = "tensor_in"
        while True:
            key = (current_id, current_port)
            if key not in conn_map:
                break
            from_id, _ = conn_map[key]
            node = self.graph.nodes.get(from_id)
            if node is None:
                break
            if hasattr(node, "get_layers"):
                ordered_nodes.append(node)
            current_id   = from_id
            current_port = "tensor_in"
        ordered_nodes.reverse()
        all_modules = []
        for node in ordered_nodes:
            all_modules.extend(node.get_layers())
        return nn.Sequential(*all_modules) if all_modules else None

    def _train_start(self) -> None:
        """Find pt_training_config node, build model from graph, start training."""
        from nodes.pytorch.training import _build_optimizer, _build_scheduler
        self._sync_inputs_from_widgets()
        try:
            outputs, _ = self.graph.execute()
        except Exception as exc:
            self._log(f"[Train] Graph error: {exc}")
            return
        cfg_node_id = None
        config = None
        for node_id, node in self.graph.nodes.items():
            if node.type_name == "pt_training_config" and node_id in outputs:
                cfg_node_id = node_id
                config = outputs[node_id].get("config")
                break
        if config is None:
            self._log("[Train] No Training Config node found.")
            return
        if config.get("dataloader") is None:
            self._log("[Train] No dataloader connected to Training Config.")
            return
        model = self._collect_model_layers(cfg_node_id)
        if model is None:
            self._log("[Train] No layers found - wire layers into Training Config's tensor_in.")
            return
        optimizer = _build_optimizer(
            config["optimizer_name"], model,
            config["lr"], config["weight_decay"], config["momentum"],
        )
        scheduler = _build_scheduler(
            config["scheduler_name"], optimizer,
            config["step_size"], config["gamma"], config["T_max"],
        )
        full_config = {
            "model":          model,
            "optimizer":      optimizer,
            "loss_fn":        config["loss_fn"],
            "dataloader":     config["dataloader"],
            "val_dataloader": config.get("val_dataloader"),
            "scheduler":      scheduler,
            "epochs":         config["epochs"],
            "device":         config["device"],
        }
        self._log(f"[Train] Starting: {config['epochs']} epochs on {config['device']}  "
                  f"({len(list(model.parameters()))} param tensors)")
        self._training_ctrl.on_epoch_end = self.refresh_graph_silent
        self._training_ctrl.start(full_config)
        try:
            dpg.set_value("train_status_text", "Status: Running")
        except Exception:
            pass

    def _train_check_wiring(self) -> None:
        """Run the graph and report training wiring status."""
        self._sync_inputs_from_widgets()
        self._log("-" * 40)
        self._log("Checking training wiring...")
        try:
            outputs, _ = self.graph.execute()
        except Exception as exc:
            self._log(f"[Check] Graph error: {exc}")
            return
        cfg_node_id = None
        config = None
        for node_id, node in self.graph.nodes.items():
            if node.type_name == "pt_training_config" and node_id in outputs:
                cfg_node_id = node_id
                config = outputs[node_id].get("config")
                break
        if config is None:
            self._log("[Check] No Training Config node found.")
            return
        model = self._collect_model_layers(cfg_node_id)
        if model is None:
            self._log("[Check] [x] No layers connected to tensor_in.")
        else:
            n_params = sum(p.numel() for p in model.parameters())
            self._log(f"[Check] [ok] Model: {len(list(model.children()))} modules, {n_params:,} parameters")
        if config.get("dataloader") is not None:
            self._log("[Check] [ok] dataloader connected")
        else:
            self._log("[Check] [x] dataloader not connected")
        val_dl = config.get("val_dataloader")
        self._log(f"[Check]   val_dataloader: {'connected' if val_dl else 'not connected (optional)'}")
        self._log(f"[Check]   optimizer: {config.get('optimizer_name', '?')}  lr={config.get('lr', '?')}")
        self._log(f"[Check]   loss: {config.get('loss_fn', '?').__class__.__name__}")
        if model is not None and config.get("dataloader") is not None:
            self._log("[Check] Ready to train!")

    def _train_pause_resume(self) -> None:
        if self._training_ctrl.status == "running":
            self._training_ctrl.pause()
            try:
                dpg.set_value("train_status_text", "Status: Paused")
            except Exception:
                pass
        elif self._training_ctrl.status == "paused":
            self._training_ctrl.resume()
            try:
                dpg.set_value("train_status_text", "Status: Running")
            except Exception:
                pass

    def _train_stop(self) -> None:
        self._training_ctrl.stop()
        try:
            dpg.set_value("train_status_text", "Status: Stopped")
        except Exception:
            pass

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
