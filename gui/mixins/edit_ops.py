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
        """Pre-populate with a classification demo using the universal DatasetNode.

        Generates a tiny synthetic dataset on disk (20 images + labels) and wires:
        Dataset → Flatten → Linear+ReLU → Linear → TrainOutput. Shows the
        recommended workflow with the new architecture — zero legacy nodes.
        """
        self._ensure_demo_classify_data()

        from nodes.pytorch.dataset      import DatasetNode
        from nodes.pytorch.flatten       import FlattenNode
        from nodes.pytorch.linear        import LinearNode
        from nodes.pytorch.train_output  import TrainOutputNode
        from templates._helpers          import grid

        pos = grid(step_x=240)
        positions: dict[str, tuple[int, int]] = {}

        ds = DatasetNode()
        ds.inputs["path"].default_value      = "demo_data/classify"
        ds.inputs["batch_size"].default_value = 8
        self.graph.add_node(ds); positions[ds.id] = pos()

        flat = FlattenNode()
        self.graph.add_node(flat); positions[flat.id] = pos()

        h1 = LinearNode()
        h1.inputs["in_features"].default_value  = 192  # 8*8*3 flattened
        h1.inputs["out_features"].default_value = 32
        h1.inputs["activation"].default_value   = "relu"
        self.graph.add_node(h1); positions[h1.id] = pos()

        head = LinearNode()
        head.inputs["in_features"].default_value  = 32
        head.inputs["out_features"].default_value = 2
        head.inputs["activation"].default_value   = "none"
        self.graph.add_node(head); positions[head.id] = pos()

        target = TrainOutputNode()
        self.graph.add_node(target); positions[target.id] = pos()

        # Wire: dataset.image → flatten → linear → linear → train output
        self.graph.add_connection(ds.id,   "image",      flat.id,   "tensor_in")
        self.graph.add_connection(flat.id,  "tensor_out", h1.id,     "tensor_in")
        self.graph.add_connection(h1.id,    "tensor_out", head.id,   "tensor_in")
        self.graph.add_connection(head.id,  "tensor_out", target.id, "tensor_in")

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
