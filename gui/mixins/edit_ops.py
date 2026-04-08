"""EditOpsMixin — editor operations: clear, copy/paste, palette search, demo graph."""
from __future__ import annotations
import dearpygui.dearpygui as dpg
from core.node import PortType
from nodes import NODE_REGISTRY


class EditOpsMixin:
    """Editor operations: clear canvas, copy/paste nodes, palette filter, demo graph."""

    def _clear_editor(self) -> None:
        """Delete all nodes and links from DPG, clear all dicts, and reset the graph."""
        for lt in list(self.dpg_link_to_conn.keys()):
            try:
                if dpg.does_item_exist(lt):
                    dpg.delete_item(lt)
            except Exception:
                pass
        self.dpg_link_to_conn.clear()
        for node_tag in list(self.node_id_to_dpg.values()):
            try:
                if dpg.does_item_exist(node_tag):
                    dpg.delete_item(node_tag)
            except Exception:
                pass
        self.dpg_node_to_node_id.clear()
        self.node_id_to_dpg.clear()
        self.dpg_attr_to_key.clear()
        self.input_widgets.clear()
        self.output_displays.clear()
        self.selected_node_id = None
        self._update_inspector(None)
        self._node_base_pos.clear()
        self._zoom = 1.0
        self.graph.clear()

    def _copy_nodes(self) -> None:
        try:
            selected = list(dpg.get_selected_nodes("NodeEditor"))
        except Exception:
            return
        self._clipboard.clear()
        positions = []
        for nt in selected:
            node_id = self.dpg_node_to_node_id.get(nt)
            if node_id is None:
                continue
            node = self.graph.get_node(node_id)
            if node is None:
                continue
            try:
                pos = dpg.get_item_pos(self.node_id_to_dpg.get(node_id, nt))
            except Exception:
                pos = [100, 100]
            positions.append(pos)
            self._clipboard.append({
                "type_name": node.type_name,
                "orig_id": node_id,
                "input_defaults": {k: p.default_value for k, p in node.inputs.items()},
                "pos": pos,
            })
        if not positions:
            return
        cx = sum(p[0] for p in positions) / len(positions)
        cy = sum(p[1] for p in positions) / len(positions)
        for entry in self._clipboard:
            entry["rel_pos"] = [entry["pos"][0] - cx, entry["pos"][1] - cy]
        self._log(f"Copied {len(self._clipboard)} node(s)")

    def _paste_nodes(self) -> None:
        if not self._clipboard:
            return
        ox, oy = 400, 300
        id_map = {}
        for entry in self._clipboard:
            cls = NODE_REGISTRY.get(entry["type_name"])
            if cls is None:
                continue
            node = cls()
            for k, v in entry.get("input_defaults", {}).items():
                if k in node.inputs and v is not None:
                    if node.inputs[k].port_type in (PortType.FLOAT, PortType.INT, PortType.BOOL, PortType.STRING):
                        node.inputs[k].default_value = v
            self.graph.add_node(node)
            rx, ry = entry.get("rel_pos", [0, 0])
            pos = (int(ox + rx + 40), int(oy + ry + 40))
            self.add_node_to_editor(node, pos)
            id_map[entry["orig_id"]] = node.id
        self._log(f"Pasted {len(id_map)} node(s)")

    def _filter_palette(self, text: str) -> None:
        text = text.lower().strip()
        for type_name, cls in NODE_REGISTRY.items():
            btn_tag = f"palette_btn_{type_name}"
            if dpg.does_item_exist(btn_tag):
                if not text or text in type_name.lower() or text in cls.label.lower():
                    dpg.show_item(btn_tag)
                else:
                    dpg.hide_item(btn_tag)

    def _process_requests(self) -> None:
        if self._undo_requested:
            self._undo_requested = False
            self.command_stack.undo()
        if self._redo_requested:
            self._redo_requested = False
            self.command_stack.redo()
        if self._save_requested:
            self._save_requested = False
            self._save_graph()
        if self._export_requested:
            self._export_requested = False
            self._export_script()
        if self._copy_requested:
            self._copy_requested = False
            self._copy_nodes()
        if self._paste_requested:
            self._paste_requested = False
            self._paste_nodes()

    def _build_demo_graph(self) -> None:
        """Pre-populate with a multimodal training demo (audio + image, 2 datasets)."""
        from nodes.pytorch.linear  import LinearNode
        from nodes.pytorch.flatten import FlattenNode
        from nodes.pytorch.batch_input import BatchInputNode
        from nodes.pytorch.folder_multimodal_dataset import FolderMultimodalDatasetNode
        from nodes.pytorch.multi_dataset import MultiDatasetNode
        from nodes.pytorch.multimodal_model import MultimodalModelNode
        from nodes.pytorch.multimodal_training_config import MultimodalTrainingConfigNode

        # Lazy-create a tiny synthetic dataset on disk so the demo trains out of the box
        self._ensure_demo_multimodal_data()

        # ---- Two datasets: one audio-only, one image-only --------------------
        ds_audio = FolderMultimodalDatasetNode()
        ds_audio.inputs["root_path"].default_value  = "demo_data/audio_set"
        ds_audio.inputs["modalities"].default_value = "audio,image"
        self.graph.add_node(ds_audio)

        ds_image = FolderMultimodalDatasetNode()
        ds_image.inputs["root_path"].default_value  = "demo_data/image_set"
        ds_image.inputs["modalities"].default_value = "audio,image"
        self.graph.add_node(ds_image)

        # ---- MultiLoader: round-robin through both ---------------------------
        multi = MultiDatasetNode()
        multi.inputs["batch_size"].default_value = 4
        multi.inputs["strategy"].default_value   = "round_robin"
        self.graph.add_node(multi)

        # ---- Batch Input: where the training batch enters the model ---------
        batch_in = BatchInputNode()
        self.graph.add_node(batch_in)

        # ---- Audio encoder branch: 64 -> 32 ----------------------------------
        audio_lin = LinearNode()
        audio_lin.inputs["in_features"].default_value  = 64
        audio_lin.inputs["out_features"].default_value = 32
        audio_lin.inputs["activation"].default_value   = "relu"
        self.graph.add_node(audio_lin)

        # ---- Image encoder branch: Flatten -> 192 -> 32 ---------------------
        img_flat = FlattenNode()
        self.graph.add_node(img_flat)

        img_lin = LinearNode()
        img_lin.inputs["in_features"].default_value  = 192   # 8*8*3
        img_lin.inputs["out_features"].default_value = 32
        img_lin.inputs["activation"].default_value   = "relu"
        self.graph.add_node(img_lin)

        # ---- Multimodal model: concat fusion, learnable token for missing ---
        mm = MultimodalModelNode()
        mm.inputs["fusion"].default_value           = "concat"
        mm.inputs["fusion_dim"].default_value       = 32
        mm.inputs["missing_strategy"].default_value = "learnable_token"
        mm.inputs["active_flag"].default_value      = True
        self.graph.add_node(mm)

        # ---- Classification head: (6 modalities * 32) + 6 flags = 198 -> 2 ---
        head = LinearNode()
        head.inputs["in_features"].default_value  = 198
        head.inputs["out_features"].default_value = 2
        self.graph.add_node(head)

        # ---- Multimodal training config -------------------------------------
        cfg = MultimodalTrainingConfigNode()
        cfg.inputs["epochs"].default_value          = 10
        cfg.inputs["device"].default_value          = "cpu"
        cfg.inputs["optimizer"].default_value       = "adam"
        cfg.inputs["lr"].default_value              = 0.005
        cfg.inputs["loss"].default_value            = "crossentropy"
        cfg.inputs["freeze_strategy"].default_value = "hard"
        cfg.inputs["log_modalities"].default_value  = True
        self.graph.add_node(cfg)

        # ---- Connections ----------------------------------------------------
        self.graph.add_connection(ds_audio.id, "dataset",    multi.id,     "dataset_1")
        self.graph.add_connection(ds_image.id, "dataset",    multi.id,     "dataset_2")
        self.graph.add_connection(multi.id,    "dataloader", cfg.id,       "dataloader")
        self.graph.add_connection(multi.id,    "dataloader", batch_in.id,  "dataloader")
        # Batch Input -> encoder chains
        self.graph.add_connection(batch_in.id,  "audio",      audio_lin.id, "tensor_in")
        self.graph.add_connection(batch_in.id,  "image",      img_flat.id,  "tensor_in")
        # Image chain
        self.graph.add_connection(img_flat.id,  "tensor_out", img_lin.id,   "tensor_in")
        # Encoder outputs -> multimodal ports
        self.graph.add_connection(audio_lin.id, "tensor_out", mm.id,        "audio")
        self.graph.add_connection(img_lin.id,   "tensor_out", mm.id,        "image")
        # Fusion -> head -> training config
        self.graph.add_connection(mm.id,        "tensor_out", head.id,      "tensor_in")
        self.graph.add_connection(head.id,      "tensor_out", cfg.id,       "tensor_in")

        # ---- Layout ---------------------------------------------------------
        positions = {
            ds_audio.id:  (  20,  20),
            ds_image.id:  (  20, 200),
            multi.id:     ( 240, 100),
            batch_in.id:  ( 460, 100),
            audio_lin.id: ( 680, 320),
            img_flat.id:  ( 680, 460),
            img_lin.id:   ( 900, 460),
            mm.id:        (1140, 380),
            head.id:      (1380, 380),
            cfg.id:       (1620, 200),
        }
        all_nodes = [ds_audio, ds_image, multi, batch_in,
                     audio_lin, img_flat, img_lin, mm, head, cfg]
        for node in all_nodes:
            self.add_node_to_editor(node, positions[node.id])

        # ---- DPG visual links -----------------------------------------------
        link_pairs = [
            (ds_audio.id,  "dataset",    multi.id,     "dataset_1"),
            (ds_image.id,  "dataset",    multi.id,     "dataset_2"),
            (multi.id,     "dataloader", cfg.id,       "dataloader"),
            (multi.id,     "dataloader", batch_in.id,  "dataloader"),
            (batch_in.id,  "audio",      audio_lin.id, "tensor_in"),
            (batch_in.id,  "image",      img_flat.id,  "tensor_in"),
            (img_flat.id,  "tensor_out", img_lin.id,   "tensor_in"),
            (audio_lin.id, "tensor_out", mm.id,        "audio"),
            (img_lin.id,   "tensor_out", mm.id,        "image"),
            (mm.id,        "tensor_out", head.id,      "tensor_in"),
            (head.id,      "tensor_out", cfg.id,       "tensor_in"),
        ]
        for fn, fp, tn, tp in link_pairs:
            from_attr = f"attr_out_{fn}_{fp}"
            to_attr   = f"attr_in_{tn}_{tp}"
            if dpg.does_item_exist(from_attr) and dpg.does_item_exist(to_attr):
                link_tag = dpg.add_node_link(from_attr, to_attr, parent="NodeEditor")
                self.dpg_link_to_conn[link_tag] = (fn, fp, tn, tp)

    def _ensure_demo_multimodal_data(self) -> None:
        """Generate a tiny synthetic 2-class multimodal dataset on disk if it doesn't exist."""
        import os
        if os.path.isdir("demo_data/audio_set") and os.path.isdir("demo_data/image_set"):
            return
        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            return

        def make_set(name, modality, n=20):
            root = os.path.join("demo_data", name)
            os.makedirs(os.path.join(root, modality), exist_ok=True)
            rows = ["id,label,audio,image"]
            for i in range(n):
                label = i % 2
                if modality == "audio":
                    audio = np.random.randn(64).astype(np.float32) + (label * 2.0)
                    np.save(os.path.join(root, "audio", f"{i:03d}.npy"), audio)
                    rows.append(f"{i:03d},{label},{i:03d}.npy,")
                else:  # image
                    base = np.zeros((8, 8, 3), dtype=np.uint8)
                    base[:] = (label * 200, 50, 100)
                    base += np.random.randint(0, 30, base.shape, dtype=np.uint8)
                    Image.fromarray(base).save(os.path.join(root, "image", f"{i:03d}.png"))
                    rows.append(f"{i:03d},{label},,{i:03d}.png")
            with open(os.path.join(root, "samples.csv"), "w") as f:
                f.write("\n".join(rows))

        try:
            make_set("audio_set", "audio", n=20)
            make_set("image_set", "image", n=20)
        except Exception:
            pass
