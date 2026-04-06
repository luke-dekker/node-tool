"""Main DearPyGui application for the node-based programming tool."""

from __future__ import annotations
import sys
import os
import time
from typing import Any, Type

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import dearpygui.dearpygui as dpg

from core.graph import Graph, CommandStack, Command
from core.node import PortType, BaseNode
from nodes import NODE_REGISTRY, get_nodes_by_category, CATEGORY_ORDER
from gui.constants import VIEWPORT_W, VIEWPORT_H, PALETTE_W, INSPECTOR_W, TERMINAL_H, NODE_INPUT_W
from gui.theme import (
    create_global_theme, create_node_theme, CATEGORY_COLORS, TEXT, TEXT_DIM,
    ACCENT, ACCENT2, BG_DARK, BG_MID, BG_LIGHT,
    FLOAT_PIN, INT_PIN, BOOL_PIN, STRING_PIN, ANY_PIN,
    TENSOR_PIN, MODULE_PIN, DATALOADER_PIN, OPTIMIZER_PIN, LOSS_FN_PIN,
    DATAFRAME_PIN, NDARRAY_PIN, SERIES_PIN, SKLEARN_PIN, IMAGE_PIN,
    SCHEDULER_PIN, DATASET_PIN, TRANSFORM_PIN,
)
from gui.training_panel import TrainingController
from gui.mixins import (
    TrainingMixin, PollingMixin, FileOpsMixin,
    EditOpsMixin, LayoutMixin, HandlersMixin,
)

PIN_COLORS = {
    PortType.FLOAT:        FLOAT_PIN,
    PortType.INT:          INT_PIN,
    PortType.BOOL:         BOOL_PIN,
    PortType.STRING:       STRING_PIN,
    PortType.ANY:          ANY_PIN,
    PortType.TENSOR:       TENSOR_PIN,
    PortType.MODULE:       MODULE_PIN,
    PortType.DATALOADER:   DATALOADER_PIN,
    PortType.OPTIMIZER:    OPTIMIZER_PIN,
    PortType.LOSS_FN:      LOSS_FN_PIN,
    PortType.DATAFRAME:    DATAFRAME_PIN,
    PortType.NDARRAY:      NDARRAY_PIN,
    PortType.SERIES:       SERIES_PIN,
    PortType.SKLEARN_MODEL: SKLEARN_PIN,
    PortType.IMAGE:        IMAGE_PIN,
    PortType.SCHEDULER:    SCHEDULER_PIN,
    PortType.DATASET:      DATASET_PIN,
    PortType.TRANSFORM:    TRANSFORM_PIN,
}

PIN_SHAPES = {
    PortType.FLOAT:        dpg.mvNode_PinShape_CircleFilled,
    PortType.INT:          dpg.mvNode_PinShape_TriangleFilled,
    PortType.BOOL:         dpg.mvNode_PinShape_QuadFilled,
    PortType.STRING:       dpg.mvNode_PinShape_CircleFilled,
    PortType.ANY:          dpg.mvNode_PinShape_Circle,
    PortType.TENSOR:       dpg.mvNode_PinShape_Circle,
    PortType.MODULE:       dpg.mvNode_PinShape_Triangle,
    PortType.DATALOADER:   dpg.mvNode_PinShape_Quad,
    PortType.OPTIMIZER:    dpg.mvNode_PinShape_Circle,
    PortType.LOSS_FN:      dpg.mvNode_PinShape_Circle,
    PortType.DATAFRAME:    dpg.mvNode_PinShape_QuadFilled,
    PortType.NDARRAY:      dpg.mvNode_PinShape_TriangleFilled,
    PortType.SERIES:       dpg.mvNode_PinShape_CircleFilled,
    PortType.SKLEARN_MODEL: dpg.mvNode_PinShape_Triangle,
    PortType.IMAGE:        dpg.mvNode_PinShape_Quad,
    PortType.SCHEDULER:    dpg.mvNode_PinShape_CircleFilled,
    PortType.DATASET:      dpg.mvNode_PinShape_CircleFilled,
    PortType.TRANSFORM:    dpg.mvNode_PinShape_CircleFilled,
}

# Reference PortTypes that cannot be edited via widget
def _summarise(val, has_pd=False, pd=None) -> str:
    """Return a short, node-safe string for any output value."""
    if val is None:
        return "None"
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, float):
        return f"{val:.4g}"
    if isinstance(val, int):
        return str(val)
    if isinstance(val, str):
        return val[:60] + ("..." if len(val) > 60 else "")

    # numpy
    try:
        import numpy as np
        if isinstance(val, np.ndarray):
            return f"ndarray {list(val.shape)} {val.dtype}"
    except ImportError:
        pass

    # torch tensor
    try:
        import torch
        if isinstance(val, torch.Tensor):
            return f"Tensor {list(val.shape)} {val.dtype}"
        if isinstance(val, torch.nn.Module):
            n = sum(p.numel() for p in val.parameters())
            return f"{val.__class__.__name__} ({n:,} params)"
        if isinstance(val, torch.optim.Optimizer):
            lr = val.param_groups[0].get("lr", "?")
            return f"{val.__class__.__name__} lr={lr}"
    except ImportError:
        pass

    # torch dataloader
    try:
        from torch.utils.data import DataLoader
        if isinstance(val, DataLoader):
            return f"DataLoader batches={len(val)} bs={val.batch_size}"
    except ImportError:
        pass

    # torch loss
    try:
        import torch.nn as nn
        if isinstance(val, nn.Module) and "Loss" in val.__class__.__name__:
            return val.__class__.__name__
    except ImportError:
        pass

    # pandas
    if has_pd and pd is not None:
        if isinstance(val, pd.DataFrame):
            return f"DataFrame {val.shape}"
        if isinstance(val, pd.Series):
            return f"Series len={len(val)}"

    # dict - summarise keys only, never dump values
    if isinstance(val, dict):
        keys = list(val.keys())
        keys_str = ", ".join(str(k) for k in keys[:6])
        suffix = "..." if len(keys) > 6 else ""
        return f"dict {{{keys_str}{suffix}}}"

    if isinstance(val, (list, tuple)):
        return f"{type(val).__name__}[{len(val)}]"

    # fallback - class name only, never full repr
    return val.__class__.__name__


_REFERENCE_TYPES = {
    PortType.TENSOR, PortType.MODULE, PortType.DATALOADER,
    PortType.OPTIMIZER, PortType.LOSS_FN,
    PortType.DATAFRAME, PortType.NDARRAY, PortType.SERIES,
    PortType.SKLEARN_MODEL, PortType.IMAGE, PortType.SCHEDULER,
    PortType.DATASET, PortType.TRANSFORM, PortType.ANY,
}

# Output ports that show a live value in the node after execution
_PRIMITIVE_OUT_TYPES = {
    PortType.FLOAT, PortType.INT, PortType.BOOL, PortType.STRING,
}


class NodeApp(
    TrainingMixin, PollingMixin, FileOpsMixin,
    EditOpsMixin, LayoutMixin, HandlersMixin,
):
    """Main application class. Behaviour split across gui/mixins/ by concern."""

    def __init__(self) -> None:
        self.graph = Graph()

        # DPG tag -> Python mapping
        self.dpg_node_to_node_id: dict[int | str, str] = {}
        self.node_id_to_dpg: dict[str, int | str] = {}
        self.dpg_attr_to_key: dict[int | str, tuple[str, str, bool]] = {}
        # dpg_link_id -> (from_node_id, from_port, to_node_id, to_port)
        self.dpg_link_to_conn: dict[int | str, tuple[str, str, str, str]] = {}

        # Input widget tags: (node_id, port_name) -> dpg widget tag
        self.input_widgets: dict[tuple[str, str], int | str] = {}
        # Output display tags: (node_id, port_name) -> dpg text tag
        self.output_displays: dict[tuple[str, str], int | str] = {}

        # Selected node
        self.selected_node_id: str | None = None

        # Category themes cache
        self._cat_themes: dict[str, int] = {}

        # Node spawn counter for unique positioning
        self._spawn_count = 0

        # Last execution outputs
        self._last_outputs: dict[str, dict[str, Any]] = {}

        # Screenshot frame tracking
        self.screenshot_path: str | None = None
        self._frame_count = 0

        # Terminal log lines
        self._terminal_lines: list[str] = []

        # Deletion flags - set mid-render, acted on between frames.
        self._delete_requested: bool = False
        self._delink_pending: list = []   # link tags from delink_callback

        # Undo/redo + save/copy/paste
        self.command_stack = CommandStack()
        self._clipboard: list[dict] = []
        self._save_path: str | None = None

        # Request flags (set in key handlers, processed between frames)
        self._undo_requested = False
        self._redo_requested = False
        self._copy_requested = False
        self._paste_requested = False
        self._save_requested = False
        self._export_requested = False

        # Training controller
        self._training_ctrl = TrainingController()

        # Hot reload
        from core.custom import HotReloader
        self._hot_reloader = HotReloader()
        self._palette_cat_items: dict[str, int] = {}  # category -> collapsing_header DPG ID
        self._term_last_size: tuple[int, int] = (0, 0)  # track terminal window size for resize
        self._layout_last: tuple = (0, 0, 0)           # (vw, vh, term_h) - triggers reanchor

        # Canvas zoom
        self._zoom: float = 1.0
        self._node_base_pos: dict[str, list[float]] = {}  # node_id -> [x,y] at zoom=1

        # Inline image textures for Viz nodes: tex_tag -> dpg texture tag
        self._node_textures: dict[str, int] = {}

    # -- Theme helpers -----------------------------------------------------

    def _get_cat_theme(self, category: str) -> int:
        if category not in self._cat_themes:
            self._cat_themes[category] = create_node_theme(category)
        return self._cat_themes[category]

    # -- Node creation helpers ---------------------------------------------

    def _next_pos(self) -> tuple[int, int]:
        """Staggered position for newly spawned nodes from palette."""
        x = 300 + (self._spawn_count % 4) * 220
        y = 120 + (self._spawn_count // 4) * 140 + (self._spawn_count % 4) * 30
        self._spawn_count += 1
        return x, y

    def _create_input_widget(self, node: BaseNode, port_name: str, parent_tag) -> None:
        """Create a DPG input widget for the given input port."""
        port = node.inputs[port_name]
        key = (node.id, port_name)
        ptype = port.port_type
        default = port.default_value

        if ptype in _REFERENCE_TYPES:
            # Show the port name as the label - tells user exactly what to connect
            dpg.add_text(port_name, parent=parent_tag, color=[160, 160, 180])
            return

        if ptype == PortType.FLOAT:
            tag = dpg.add_input_float(
                label=port_name, default_value=float(default) if default is not None else 0.0,
                width=NODE_INPUT_W, step=0, parent=parent_tag,
                callback=lambda s, a, u: self._on_input_changed(u),
                user_data=key,
            )
        elif ptype == PortType.INT:
            tag = dpg.add_input_int(
                label=port_name, default_value=int(default) if default is not None else 0,
                width=NODE_INPUT_W, step=0, parent=parent_tag,
                callback=lambda s, a, u: self._on_input_changed(u),
                user_data=key,
            )
        elif ptype == PortType.BOOL:
            tag = dpg.add_checkbox(
                label=port_name, default_value=bool(default) if default is not None else False,
                parent=parent_tag,
                callback=lambda s, a, u: self._on_input_changed(u),
                user_data=key,
            )
        elif ptype == PortType.STRING:
            if port.choices:
                tag = dpg.add_combo(
                    label=port_name,
                    items=port.choices,
                    default_value=str(default) if default is not None else port.choices[0],
                    width=NODE_INPUT_W, parent=parent_tag,
                    callback=lambda s, a, u: self._on_input_changed(u),
                    user_data=key,
                )
            else:
                tag = dpg.add_input_text(
                    label=port_name,
                    default_value=str(default) if default is not None else "",
                    width=NODE_INPUT_W, parent=parent_tag,
                    callback=lambda s, a, u: self._on_input_changed(u),
                    user_data=key,
                )
        else:  # ANY
            tag = dpg.add_input_text(
                label=port_name,
                default_value=str(default) if default is not None else "",
                width=NODE_INPUT_W, parent=parent_tag,
                callback=lambda s, a, u: self._on_input_changed(u),
                user_data=key,
            )
        self.input_widgets[key] = tag

    def _on_input_changed(self, key: tuple[str, str]) -> None:
        """Sync widget value back to graph node default."""
        node_id, port_name = key
        node = self.graph.get_node(node_id)
        if node is None or port_name not in node.inputs:
            return
        tag = self.input_widgets.get(key)
        if tag is None:
            return
        try:
            val = dpg.get_value(tag)
            node.inputs[port_name].default_value = val
        except Exception:
            pass

    def add_node_to_editor(self, node: BaseNode, pos: tuple[int, int],
                           editor_tag="NodeEditor") -> str:
        """
        Creates DPG node items for the given Python node and registers all mappings.
        Returns the dpg node tag.
        """
        cat_theme = self._get_cat_theme(node.category)
        node_tag = f"dpgnode_{node.id}"

        # Store base position (unscaled) for zoom calculations
        scaled_pos = [pos[0] * self._zoom, pos[1] * self._zoom]
        self._node_base_pos[node.id] = list(pos)

        with dpg.node(label=node.label, tag=node_tag,
                      pos=scaled_pos, parent=editor_tag):

            # Input attributes
            for port_name, port in node.inputs.items():
                attr_tag = f"attr_in_{node.id}_{port_name}"
                shape = PIN_SHAPES.get(port.port_type, dpg.mvNode_PinShape_Circle)
                with dpg.node_attribute(
                    tag=attr_tag,
                    attribute_type=dpg.mvNode_Attr_Input,
                    shape=shape,
                ):
                    self._create_input_widget(node, port_name, attr_tag)
                self.dpg_attr_to_key[attr_tag] = (node.id, port_name, True)
                self.dpg_attr_to_key[dpg.get_alias_id(attr_tag)] = (node.id, port_name, True)

            # Output attributes
            for port_name, port in node.outputs.items():
                attr_tag = f"attr_out_{node.id}_{port_name}"
                shape = PIN_SHAPES.get(port.port_type, dpg.mvNode_PinShape_Circle)
                with dpg.node_attribute(
                    tag=attr_tag,
                    attribute_type=dpg.mvNode_Attr_Output,
                    shape=shape,
                ):
                    out_display = f"outdisp_{node.id}_{port_name}"
                    if port.port_type in _PRIMITIVE_OUT_TYPES:
                        # Primitive types: show live value after execution
                        dpg.add_text(f"{port_name}: -", tag=out_display)
                        self.output_displays[(node.id, port_name)] = out_display
                    else:
                        # Complex types: just show the port name, never dump the value
                        dpg.add_text(port_name, color=[160, 160, 180], tag=out_display)
                self.dpg_attr_to_key[attr_tag] = (node.id, port_name, False)
                self.dpg_attr_to_key[dpg.get_alias_id(attr_tag)] = (node.id, port_name, False)

        dpg.bind_item_theme(node_tag, cat_theme)
        # Map both string alias AND integer id -> node.id
        # get_selected_nodes() returns integer IDs, not string aliases
        self.dpg_node_to_node_id[node_tag] = node.id
        self.dpg_node_to_node_id[dpg.get_alias_id(node_tag)] = node.id
        self.node_id_to_dpg[node.id] = node_tag
        return node_tag

    def spawn_node(self, type_name: str, pos: tuple[int, int] | None = None) -> BaseNode | None:
        """Instantiate a node, add to graph, create DPG items."""
        cls = NODE_REGISTRY.get(type_name)
        if cls is None:
            return None
        node = cls()
        self.graph.add_node(node)
        if pos is None:
            pos = self._next_pos()
        self.add_node_to_editor(node, pos)
        return node

    def delete_selected(self) -> None:
        """Called by key handler - fires INSIDE render_dearpygui_frame().
        Only set a flag; do NOT touch DPG items here at all."""
        self._delete_requested = True

    def _flush_deletions(self) -> None:
        """Called BEFORE render_dearpygui_frame() - safe to delete items here."""
        need_work = self._delete_requested or bool(self._delink_pending)
        if not need_work:
            return

        node_tags: list = []
        link_tags: list = []

        if self._delete_requested:
            self._delete_requested = False
            try:
                node_tags = list(dpg.get_selected_nodes("NodeEditor"))
            except Exception:
                node_tags = []
            try:
                link_tags = list(dpg.get_selected_links("NodeEditor"))
            except Exception:
                link_tags = []

        # Merge in any delink_callback-sourced link tags
        link_tags = list(set(link_tags + self._delink_pending))
        self._delink_pending.clear()

        if not node_tags and not link_tags:
            return

        # Clear DPG selection before touching anything
        try:
            dpg.clear_selected_nodes("NodeEditor")
            dpg.clear_selected_links("NodeEditor")
        except Exception:
            pass

        # Delete standalone selected links
        for lt in link_tags:
            conn = self.dpg_link_to_conn.pop(lt, None)
            if conn:
                self.graph.remove_connection(*conn)
            try:
                if dpg.does_item_exist(lt):
                    dpg.delete_item(lt)
            except Exception:
                pass

        # Delete nodes (linked items already gone)
        for node_tag in node_tags:
            node_id = self.dpg_node_to_node_id.get(node_tag)
            if node_id:
                # Remove any remaining links touching this node from DPG
                connected = [
                    lt for lt, c in list(self.dpg_link_to_conn.items())
                    if c[0] == node_id or c[2] == node_id
                ]
                for lt in connected:
                    conn = self.dpg_link_to_conn.pop(lt, None)
                    if conn:
                        self.graph.remove_connection(*conn)
                    try:
                        if dpg.does_item_exist(lt):
                            dpg.delete_item(lt)
                    except Exception:
                        pass

                # Clean up tracking dicts
                for k in [k for k in list(self.input_widgets) if k[0] == node_id]:
                    self.input_widgets.pop(k, None)
                for k in [k for k in list(self.output_displays) if k[0] == node_id]:
                    self.output_displays.pop(k, None)

                self.graph.remove_node(node_id)
                self.node_id_to_dpg.pop(node_id, None)
                self.dpg_node_to_node_id.pop(node_tag, None)
                self._node_base_pos.pop(node_id, None)

                if self.selected_node_id == node_id:
                    self.selected_node_id = None
                    self._update_inspector(None)

            try:
                if dpg.does_item_exist(node_tag):
                    dpg.delete_item(node_tag)
            except Exception:
                pass

    # -- Link callbacks ----------------------------------------------------

    def _link_callback(self, sender, app_data) -> None:
        """Called when user drags between two pins."""
        from_attr, to_attr = app_data[0], app_data[1]
        from_info = self.dpg_attr_to_key.get(from_attr)
        to_info = self.dpg_attr_to_key.get(to_attr)
        if from_info is None or to_info is None:
            return

        from_node_id, from_port, from_is_input = from_info
        to_node_id, to_port, to_is_input = to_info

        # Ensure we're going output -> input
        if from_is_input and not to_is_input:
            from_node_id, from_port, to_node_id, to_port = to_node_id, to_port, from_node_id, from_port
        elif not from_is_input and to_is_input:
            pass  # correct direction already
        else:
            return  # both same direction, invalid

        conn = self.graph.add_connection(from_node_id, from_port, to_node_id, to_port)
        if conn is None:
            self._log(f"[WARN] Could not connect {from_port} -> {to_port} (cycle or invalid)")
            return

        link_tag = dpg.add_node_link(from_attr, to_attr, parent=sender)
        self.dpg_link_to_conn[link_tag] = (from_node_id, from_port, to_node_id, to_port)

    def _delink_callback(self, sender, app_data) -> None:
        """Called mid-render when user Ctrl+drags to disconnect a link."""
        self._delink_pending.append(app_data)

    # -- Execution ---------------------------------------------------------

    def _sync_inputs_from_widgets(self) -> None:
        """Read all DPG input widget values and push to graph node defaults."""
        for (node_id, port_name), widget_tag in self.input_widgets.items():
            node = self.graph.get_node(node_id)
            if node is None or port_name not in node.inputs:
                continue
            try:
                val = dpg.get_value(widget_tag)
                node.inputs[port_name].default_value = val
            except Exception:
                pass

    def refresh_graph_silent(self) -> None:
        """Re-execute the graph without logging - used after each training epoch
        so that tensor_out values on layer nodes reflect updated weights."""
        self._sync_inputs_from_widgets()
        try:
            outputs, _ = self.graph.execute()
            self._last_outputs = outputs
        except Exception:
            pass

    def run_graph(self) -> None:
        """Execute the graph and show results."""
        self._sync_inputs_from_widgets()
        self._log("-" * 40)
        self._log("> Running graph...")
        t0 = time.perf_counter()

        try:
            outputs, terminal_lines = self.graph.execute()
        except Exception as exc:
            self._log(f"[FATAL] {exc}")
            return

        elapsed = (time.perf_counter() - t0) * 1000
        self._last_outputs = outputs

        for line in terminal_lines:
            self._log(line)

        if not terminal_lines:
            self._log("(no Print nodes reached)")

        self._log(f"Done in {elapsed:.1f}ms")

        # Refresh code panel
        self._refresh_code_panel()

        # Update output displays on nodes
        import numpy as _np
        try:
            import pandas as _pd
            _has_pd = True
        except ImportError:
            _has_pd = False

        for (node_id, port_name), disp_tag in self.output_displays.items():
            if node_id in outputs and port_name in outputs[node_id]:
                val = outputs[node_id][port_name]
                display_str = _summarise(val, _has_pd, _pd if _has_pd else None)
                try:
                    dpg.set_value(disp_tag, f"{port_name}: {display_str}")
                except Exception:
                    pass

        # Update inline viz images
        self._update_node_images(outputs)

        # Update inspector if a node is selected
        if self.selected_node_id:
            self._update_inspector(self.selected_node_id)

    def _update_node_images(self, outputs: dict) -> None:
        """For any node that output an IMAGE, update/create a DPG texture and show it in the node."""
        import numpy as np
        for node_id, node_outputs in outputs.items():
            for port_name, value in node_outputs.items():
                if not isinstance(value, np.ndarray):
                    continue
                if value.ndim != 3 or value.shape[2] != 3:
                    continue
                node = self.graph.get_node(node_id)
                if node is None:
                    continue
                # This is an IMAGE output - display it inline in the node
                tex_tag = f"tex_{node_id}_{port_name}"
                h, w = value.shape[:2]
                # Convert RGB uint8 -> RGBA float32 normalised (DPG requirement)
                rgba = np.ones((h, w, 4), dtype=np.float32)
                rgba[:, :, :3] = value.astype(np.float32) / 255.0
                flat = rgba.flatten().tolist()
                if tex_tag not in self._node_textures:
                    # Create texture registry if needed
                    if not dpg.does_item_exist("__tex_registry__"):
                        with dpg.texture_registry(tag="__tex_registry__"):
                            pass
                    tex_id = dpg.add_dynamic_texture(
                        width=w, height=h, default_value=flat,
                        tag=tex_tag, parent="__tex_registry__"
                    )
                    self._node_textures[tex_tag] = tex_id
                    # Add image widget inside node's output attribute
                    out_attr_tag = f"attr_out_{node_id}_{port_name}"
                    if dpg.does_item_exist(out_attr_tag):
                        # Replace the text placeholder with an image
                        disp_tag = f"outdisp_{node_id}_{port_name}"
                        if dpg.does_item_exist(disp_tag):
                            dpg.delete_item(disp_tag)
                        dpg.add_image(tex_tag, width=min(w, 320), height=min(h, 224),
                                      parent=out_attr_tag, tag=f"img_{node_id}_{port_name}")
                else:
                    # Update existing texture
                    dpg.set_value(tex_tag, flat)

    # -- Terminal ----------------------------------------------------------

    def _log(self, message: str) -> None:
        self._terminal_lines.append(message)
        try:
            current = dpg.get_value("terminal_text") or ""
            lines = current.split("\n") if current else []
            lines.append(message)
            # Keep last 200 lines
            if len(lines) > 200:
                lines = lines[-200:]
            dpg.set_value("terminal_text", "\n".join(lines))
            # Scroll to bottom
            dpg.set_y_scroll("terminal_scroll", dpg.get_y_scroll_max("terminal_scroll"))
        except Exception:
            pass

    def _clear_terminal(self) -> None:
        self._terminal_lines.clear()
        try:
            dpg.set_value("terminal_text", "")
        except Exception:
            pass

    def _clear_all(self) -> None:
        """Clear canvas, terminal output, and code panel."""
        self._clear_editor()
        self._clear_terminal()
        try:
            dpg.set_value("code_text", "# Run the graph to generate code preview.")
        except Exception:
            pass
        self._log("Canvas cleared.")

    # -- Inspector ---------------------------------------------------------

    def _update_inspector(self, node_id: str | None) -> None:
        """Refresh the inspector panel for the selected node."""
        try:
            # Clear inspector content
            if dpg.does_item_exist("inspector_content"):
                dpg.delete_item("inspector_content", children_only=True)

            if node_id is None:
                dpg.add_text("No node selected.", parent="inspector_content",
                             color=list(TEXT_DIM))
                return

            node = self.graph.get_node(node_id)
            if node is None:
                dpg.add_text("Node not found.", parent="inspector_content",
                             color=list(TEXT_DIM))
                return

            # Title + category on one line
            cat_color = list(CATEGORY_COLORS.get(node.category, (160, 160, 180, 255)))
            with dpg.group(horizontal=True, parent="inspector_content"):
                dpg.add_text(node.label, color=[255, 255, 255, 255])
                dpg.add_text(f"({node.category})", color=cat_color)

            # Description
            dpg.add_text(node.description, parent="inspector_content",
                         wrap=INSPECTOR_W - 20, color=list(TEXT_DIM))
            dpg.add_separator(parent="inspector_content")

            # Inputs
            if node.inputs:
                dpg.add_text("Inputs", parent="inspector_content", color=list(TEXT_DIM))
                for port_name, port in node.inputs.items():
                    pin_col = list(PIN_COLORS.get(port.port_type, (160, 160, 180, 255)))
                    val = port.default_value
                    widget_tag = self.input_widgets.get((node_id, port_name))
                    if widget_tag is not None:
                        try: val = dpg.get_value(widget_tag)
                        except Exception: pass
                    with dpg.group(horizontal=True, parent="inspector_content"):
                        dpg.add_text(f"  {port_name}", color=pin_col)
                        dpg.add_text(f"= {val}", color=list(TEXT))

            # Outputs
            if node.outputs:
                dpg.add_separator(parent="inspector_content")
                dpg.add_text("Outputs", parent="inspector_content", color=list(TEXT_DIM))
                for port_name, port in node.outputs.items():
                    if port_name == "__terminal__":
                        continue
                    pin_col = list(PIN_COLORS.get(port.port_type, (160, 160, 180, 255)))
                    result_val = "-"
                    if node_id in self._last_outputs and port_name in self._last_outputs[node_id]:
                        raw = self._last_outputs[node_id][port_name]
                        if isinstance(raw, float):
                            result_val = f"{raw:.6g}"
                        elif hasattr(raw, "shape"):
                            result_val = f"shape={list(raw.shape)} dtype={getattr(raw, 'dtype', '?')}"
                        else:
                            result_val = str(raw)
                    with dpg.group(horizontal=True, parent="inspector_content"):
                        dpg.add_text(f"  {port_name}", color=pin_col)
                        dpg.add_text(f"= {result_val}", color=list(ACCENT))

        except Exception as exc:
            print(f"[Inspector] Error: {exc}")

    # -- Selection handler -------------------------------------------------

    def _check_selection(self) -> None:
        """Poll selected nodes and update inspector (called each frame)."""
        selected = dpg.get_selected_nodes("NodeEditor")
        if selected:
            node_tag = selected[0]
            node_id = self.dpg_node_to_node_id.get(node_tag)
            if node_id and node_id != self.selected_node_id:
                self.selected_node_id = node_id
                self._update_inspector(node_id)
        else:
            if self.selected_node_id is not None:
                self.selected_node_id = None
                self._update_inspector(None)

    # -- Context menu ------------------------------------------------------

    def _show_context_menu(self, sender, app_data) -> None:
        """Right-click context menu on the node editor."""
        if dpg.does_item_exist("node_ctx_menu"):
            dpg.delete_item("node_ctx_menu")

        categories = get_nodes_by_category()
        with dpg.window(
            tag="node_ctx_menu",
            popup=True,
            autosize=True,
            no_title_bar=True,
            no_move=True,
        ):
            dpg.add_text("Add Node", color=list(ACCENT))
            dpg.add_separator()
            for cat in CATEGORY_ORDER:
                if cat not in categories:
                    continue
                cat_color = list(CATEGORY_COLORS.get(cat, (160, 160, 180, 255)))
                dpg.add_text(cat, color=cat_color)
                for cls in categories[cat]:
                    dpg.add_menu_item(
                        label=f"  {cls.label}",
                        callback=lambda s, a, u: self.spawn_node(u),
                        user_data=cls.type_name,
                    )

    def run(self, screenshot_path: str | None = None) -> None:
        self.screenshot_path = screenshot_path

        dpg.create_context()
        dpg.configure_app(docking=True, docking_space=True)

        # Create viewport
        dpg.create_viewport(
            title="Node Tool - Python Visual Scripting",
            width=VIEWPORT_W,
            height=VIEWPORT_H,
        )

        # Apply global theme
        global_theme = create_global_theme()
        dpg.bind_theme(global_theme)

        # Build UI
        self._build_layout()
        self._setup_handlers()

        # Populate demo graph
        self._build_demo_graph()

        # Initial log
        self._log("Node Tool ready. Wire nodes and click > Run Graph.")
        self._log("Tip: right-click the editor to add nodes.")

        dpg.setup_dearpygui()
        dpg.show_viewport()

        frame_count = 0
        screenshot_taken = False

        _code_counter = 0
        while dpg.is_dearpygui_running():
            self._process_requests()      # handle Ctrl+Z/Y/C/V/S between frames
            self._flush_deletions()       # safe: between frames, before render
            self._check_selection()
            self._poll_training()         # drain training events, update plot
            self._poll_hot_reload()       # check nodes/custom/ for changes
            self._poll_terminal_resize()  # resize output/code widgets with window
            self._poll_layout()           # reanchor panels to viewport edges
            _code_counter += 1
            if _code_counter >= 30:       # refresh code panel ~2x/sec at 60fps
                _code_counter = 0
                self._refresh_code_panel()
            dpg.render_dearpygui_frame()
            frame_count += 1

            # Take screenshot after ~60 frames (app fully rendered)
            if (screenshot_path and not screenshot_taken and frame_count >= 60):
                try:
                    import mss
                    with mss.mss() as sct:
                        sct.shot(output=screenshot_path)
                    self._log(f"Screenshot saved: {screenshot_path}")
                    screenshot_taken = True
                    print(f"Screenshot saved to: {screenshot_path}")
                except Exception as exc:
                    print(f"Screenshot failed: {exc}")
                    screenshot_taken = True

            # Auto-exit after screenshot if --screenshot flag
            if screenshot_taken and "--screenshot" in sys.argv:
                # Give a few more frames then exit
                if frame_count >= 70:
                    break

        dpg.destroy_context()
