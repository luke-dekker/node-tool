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
        """Pre-populate with the MNIST digit recognition demo."""
        from nodes.pytorch.layers import FlattenNode, LinearNode
        from nodes.pytorch.data import MNISTDatasetNode, SampleBatchNode
        from nodes.pytorch.training import TrainingConfigNode

        flatten = FlattenNode()
        self.graph.add_node(flatten)

        lin1 = LinearNode()
        lin1.inputs["in_features"].default_value  = 784
        lin1.inputs["out_features"].default_value = 256
        lin1.inputs["activation"].default_value   = "relu"
        self.graph.add_node(lin1)

        lin2 = LinearNode()
        lin2.inputs["in_features"].default_value  = 256
        lin2.inputs["out_features"].default_value = 128
        lin2.inputs["activation"].default_value   = "relu"
        self.graph.add_node(lin2)

        lin3 = LinearNode()
        lin3.inputs["in_features"].default_value  = 128
        lin3.inputs["out_features"].default_value = 10
        self.graph.add_node(lin3)

        mnist_train = MNISTDatasetNode()
        mnist_train.inputs["train"].default_value      = True
        mnist_train.inputs["batch_size"].default_value = 64
        self.graph.add_node(mnist_train)

        mnist_val = MNISTDatasetNode()
        mnist_val.inputs["train"].default_value      = False
        mnist_val.inputs["batch_size"].default_value = 64
        self.graph.add_node(mnist_val)

        sample = SampleBatchNode()
        self.graph.add_node(sample)

        cfg = TrainingConfigNode()
        cfg.inputs["epochs"].default_value    = 5
        cfg.inputs["device"].default_value    = "cpu"
        cfg.inputs["optimizer"].default_value = "adam"
        cfg.inputs["lr"].default_value        = 0.001
        cfg.inputs["loss"].default_value      = "crossentropy"
        cfg.inputs["scheduler"].default_value = "none"
        self.graph.add_node(cfg)

        self.graph.add_connection(mnist_train.id, "dataloader", cfg.id,    "dataloader")
        self.graph.add_connection(mnist_train.id, "dataloader", sample.id, "dataloader")
        self.graph.add_connection(mnist_val.id,   "dataloader", cfg.id,    "val_dataloader")
        self.graph.add_connection(sample.id,      "x",          flatten.id, "tensor_in")
        self.graph.add_connection(flatten.id,     "tensor_out", lin1.id,    "tensor_in")
        self.graph.add_connection(lin1.id,        "tensor_out", lin2.id,    "tensor_in")
        self.graph.add_connection(lin2.id,        "tensor_out", lin3.id,    "tensor_in")
        self.graph.add_connection(lin3.id,        "tensor_out", cfg.id,     "tensor_in")

        positions = {
            mnist_train.id: (  40,  30),
            mnist_val.id:   (  40, 200),
            sample.id:      ( 270,  30),
            flatten.id:     ( 460,  30),
            lin1.id:        ( 640,  30),
            lin2.id:        ( 840,  30),
            lin3.id:        (1040,  30),
            cfg.id:         (1240,  30),
        }
        for node in [mnist_train, mnist_val, sample, flatten, lin1, lin2, lin3, cfg]:
            self.add_node_to_editor(node, positions[node.id])

        link_pairs = [
            (mnist_train.id, "dataloader", cfg.id,    "dataloader"),
            (mnist_train.id, "dataloader", sample.id, "dataloader"),
            (mnist_val.id,   "dataloader", cfg.id,    "val_dataloader"),
            (sample.id,      "x",          flatten.id, "tensor_in"),
            (flatten.id,     "tensor_out", lin1.id,    "tensor_in"),
            (lin1.id,        "tensor_out", lin2.id,    "tensor_in"),
            (lin2.id,        "tensor_out", lin3.id,    "tensor_in"),
            (lin3.id,        "tensor_out", cfg.id,     "tensor_in"),
        ]
        for fn, fp, tn, tp in link_pairs:
            from_attr = f"attr_out_{fn}_{fp}"
            to_attr   = f"attr_in_{tn}_{tp}"
            if dpg.does_item_exist(from_attr) and dpg.does_item_exist(to_attr):
                link_tag = dpg.add_node_link(from_attr, to_attr, parent="NodeEditor")
                self.dpg_link_to_conn[link_tag] = (fn, fp, tn, tp)
