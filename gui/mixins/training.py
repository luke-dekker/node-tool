"""TrainingMixin — PyTorch model training controls."""
from __future__ import annotations
import dearpygui.dearpygui as dpg


class TrainingMixin:
    """Methods for collecting model layers, starting/stopping training, and saving the model."""

    def _train_start(self) -> None:
        """Wrap the graph as an nn.Module and start training.

        Flow:
          1. Run the graph once (live preview) to lazy-init all layer modules
          2. Find the training-config node (regular or multimodal)
          3. Wrap the graph as GraphAsModule, targeting cfg.tensor_in as the output
          4. Hand off to the background training thread
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

        # Find the training config node
        cfg_node_id = None
        config = None
        for node_id, node in self.graph.nodes.items():
            if node.type_name in ("pt_training_config", "pt_multimodal_training_config") \
                    and node_id in outputs:
                cfg_node_id = node_id
                config = outputs[node_id].get("config")
                break
        if config is None:
            self._log("[Train] No Training Config node found.")
            return
        if config.get("dataloader") is None:
            self._log("[Train] No dataloader connected to Training Config.")
            return

        # The graph IS the model
        try:
            model = GraphAsModule(self.graph, output_node_id=cfg_node_id, output_port="tensor_in")
        except Exception as exc:
            self._log(f"[Train] Failed to wrap graph as module: {exc}")
            return
        n_params = sum(p.numel() for p in model.parameters())
        if n_params == 0:
            self._log("[Train] Graph has no trainable layers upstream of Training Config.")
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
            "multimodal":      config.get("multimodal", False),
            "freeze_strategy": config.get("freeze_strategy", "hard"),
            "log_modalities":  config.get("log_modalities", True),
            "graph_module":    True,   # tells the worker this is a GraphAsModule (dict input)
        }
        self._log(f"[Train] Starting: {config['epochs']} epochs on {config['device']}  "
                  f"({n_params:,} params)")
        self._training_ctrl.on_epoch_end = self.refresh_graph_silent
        self._training_ctrl.start(full_config)
        try:
            dpg.set_value("train_status_text", "Status: Running")
        except Exception:
            pass

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
        cfg_node_id = None
        config = None
        for node_id, node in self.graph.nodes.items():
            if node.type_name in ("pt_training_config", "pt_multimodal_training_config") \
                    and node_id in outputs:
                cfg_node_id = node_id
                config = outputs[node_id].get("config")
                break
        if config is None:
            self._log("[Check] No Training Config node found.")
            return

        try:
            model = GraphAsModule(self.graph, output_node_id=cfg_node_id, output_port="tensor_in")
            n_params = sum(p.numel() for p in model.parameters())
            n_modules = len(model._layer_modules)
            if n_params == 0:
                self._log("[Check] [x] No trainable layers in the graph.")
            else:
                self._log(f"[Check] [ok] Graph model: {n_modules} layer nodes, {n_params:,} parameters")
        except Exception as exc:
            self._log(f"[Check] [x] Failed to wrap graph: {exc}")
            return

        if config.get("dataloader") is not None:
            self._log("[Check] [ok] dataloader connected")
        else:
            self._log("[Check] [x] dataloader not connected")
        val_dl = config.get("val_dataloader")
        self._log(f"[Check]   val_dataloader: {'connected' if val_dl else 'not connected (optional)'}")
        self._log(f"[Check]   optimizer: {config.get('optimizer_name', '?')}  lr={config.get('lr', '?')}")
        self._log(f"[Check]   loss: {config.get('loss_fn', '?').__class__.__name__}")
        if config.get("dataloader") is not None:
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
