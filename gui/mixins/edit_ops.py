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
        from gui.constants import PALETTE_W
        query = text.lower().strip()
        tree_tag = "palette_tree_group"
        results_tag = "palette_search_results_group"

        if dpg.does_item_exist(results_tag):
            for child in dpg.get_item_children(results_tag, 1) or []:
                dpg.delete_item(child)

        if not query:
            if dpg.does_item_exist(tree_tag):
                dpg.show_item(tree_tag)
            if dpg.does_item_exist(results_tag):
                dpg.hide_item(results_tag)
            return

        if dpg.does_item_exist(tree_tag):
            dpg.hide_item(tree_tag)
        if not dpg.does_item_exist(results_tag):
            return
        dpg.show_item(results_tag)

        scored: list[tuple[int, str, type]] = []
        for type_name, cls in NODE_REGISTRY.items():
            label = (getattr(cls, "label", "") or type_name).lower()
            tn = type_name.lower()
            cat = (getattr(cls, "category", "") or "").lower()
            sub = (getattr(cls, "subcategory", "") or "").lower()
            if label == query or tn == query:
                score = 1000
            elif label.startswith(query) or tn.startswith(query):
                score = 500
            elif query in label:
                score = 200
            elif query in tn:
                score = 100
            elif query in sub:
                score = 50
            elif query in cat:
                score = 30
            else:
                continue
            scored.append((score, type_name, cls))

        scored.sort(key=lambda x: (-x[0], x[1]))

        if not scored:
            dpg.add_text("No matching nodes", color=[140, 140, 140], parent=results_tag)
            return

        theme = getattr(self, "_palette_btn_theme", None)
        for _score, type_name, cls in scored[:60]:
            category = getattr(cls, "category", "") or ""
            sub = getattr(cls, "subcategory", "") or ""
            trail = f"  —  {category}/{sub}" if sub else (f"  —  {category}" if category else "")
            btn = dpg.add_button(
                label=f"{cls.label}{trail}",
                width=PALETTE_W - 24,
                callback=lambda s, a, u: self.spawn_node(u),
                user_data=type_name,
                parent=results_tag,
            )
            if theme is not None:
                dpg.bind_item_theme(btn, theme)
            dpg.add_spacer(height=1, parent=results_tag)

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
        """Pre-populate with a classification demo using markers.

        Data In (A:x) → Flatten → Linear+ReLU → Linear → Data Out (B:logits).
        Set the dataset path in the Training panel to train.
        """
        from nodes.pytorch.input_marker import InputMarkerNode
        from nodes.pytorch.flatten      import FlattenNode
        from nodes.pytorch.linear       import LinearNode
        from nodes.pytorch.train_marker import TrainMarkerNode
        from templates._helpers         import grid

        pos = grid(step_x=240)
        positions: dict[str, tuple[int, int]] = {}

        data_in = InputMarkerNode()
        data_in.inputs["modality"].default_value = "x"
        self.graph.add_node(data_in); positions[data_in.id] = pos()

        flat = FlattenNode()
        self.graph.add_node(flat); positions[flat.id] = pos()

        h1 = LinearNode()
        h1.inputs["in_features"].default_value  = 192
        h1.inputs["out_features"].default_value = 32
        h1.inputs["activation"].default_value   = "relu"
        self.graph.add_node(h1); positions[h1.id] = pos()

        head = LinearNode()
        head.inputs["in_features"].default_value  = 32
        head.inputs["out_features"].default_value = 2
        head.inputs["activation"].default_value   = "none"
        self.graph.add_node(head); positions[head.id] = pos()

        data_out = TrainMarkerNode()
        data_out.inputs["kind"].default_value = "logits"
        data_out.inputs["target"].default_value = "label"
        self.graph.add_node(data_out); positions[data_out.id] = pos()

        self.graph.add_connection(data_in.id, "tensor",     flat.id,     "tensor_in")
        self.graph.add_connection(flat.id,    "tensor_out", h1.id,       "tensor_in")
        self.graph.add_connection(h1.id,      "tensor_out", head.id,     "tensor_in")
        self.graph.add_connection(head.id,    "tensor_out", data_out.id, "tensor_in")

        # Build DPG visuals
        for node_id, node in self.graph.nodes.items():
            self.add_node_to_editor(node, positions.get(node_id, (100, 100)))

        for conn in self.graph.connections:
            from_attr = f"attr_out_{conn.from_node_id}_{conn.from_port}"
            to_attr   = f"attr_in_{conn.to_node_id}_{conn.to_port}"
            if dpg.does_item_exist(from_attr) and dpg.does_item_exist(to_attr):
                lt = dpg.add_node_link(from_attr, to_attr, parent="NodeEditor")
                self.dpg_link_to_conn[lt] = (
                    conn.from_node_id, conn.from_port,
                    conn.to_node_id, conn.to_port,
                )

    def _ensure_demo_classify_data(self) -> None:
        """Generate a tiny synthetic 2-class image classification dataset."""
        import os
        if os.path.isdir("demo_data/classify"):
            return
        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            return
        os.makedirs("demo_data/classify/images", exist_ok=True)
        rows = ["id,image,label"]
        for i in range(24):
            label = "cat" if i % 2 == 0 else "dog"
            img = np.zeros((8, 8, 3), dtype=np.uint8)
            img[:] = ((i % 2) * 180, 50, 100)
            img += np.random.randint(0, 40, img.shape, dtype=np.uint8)
            Image.fromarray(img).save(f"demo_data/classify/images/{i:03d}.png")
            rows.append(f"{i:03d},images/{i:03d}.png,{label}")
        with open("demo_data/classify/samples.csv", "w") as f:
            f.write("\n".join(rows))

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
