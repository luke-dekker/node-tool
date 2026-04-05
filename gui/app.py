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
from core.io import Serializer
from nodes import NODE_REGISTRY, get_nodes_by_category, CATEGORY_ORDER
from gui.theme import (
    create_global_theme, create_node_theme, create_run_button_theme,
    create_clear_button_theme, CATEGORY_COLORS, TEXT, TEXT_DIM,
    ACCENT, ACCENT2, BG_DARK, BG_MID, BG_LIGHT,
    FLOAT_PIN, INT_PIN, BOOL_PIN, STRING_PIN, ANY_PIN,
    TENSOR_PIN, MODULE_PIN, DATALOADER_PIN, OPTIMIZER_PIN, LOSS_FN_PIN,
    DATAFRAME_PIN, NDARRAY_PIN, SERIES_PIN, SKLEARN_PIN, IMAGE_PIN,
    SCHEDULER_PIN, DATASET_PIN, TRANSFORM_PIN,
)
from gui.training_panel import TrainingController

# -- Constants ----------------------------------------------------------------

VIEWPORT_W = 1600
VIEWPORT_H = 950
PALETTE_W  = 220
INSPECTOR_W = 270
TERMINAL_H  = 185

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


class NodeApp:
    """Main application class."""

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
                width=130, step=0, parent=parent_tag,
                callback=lambda s, a, u: self._on_input_changed(u),
                user_data=key,
            )
        elif ptype == PortType.INT:
            tag = dpg.add_input_int(
                label=port_name, default_value=int(default) if default is not None else 0,
                width=130, step=0, parent=parent_tag,
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
                    width=130, parent=parent_tag,
                    callback=lambda s, a, u: self._on_input_changed(u),
                    user_data=key,
                )
            else:
                tag = dpg.add_input_text(
                    label=port_name,
                    default_value=str(default) if default is not None else "",
                    width=130, parent=parent_tag,
                    callback=lambda s, a, u: self._on_input_changed(u),
                    user_data=key,
                )
        else:  # ANY
            tag = dpg.add_input_text(
                label=port_name,
                default_value=str(default) if default is not None else "",
                width=130, parent=parent_tag,
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

            # Title
            cat_color = list(CATEGORY_COLORS.get(node.category, (160, 160, 180, 255)))
            dpg.add_text(node.label, parent="inspector_content",
                         color=[255, 255, 255, 255])
            dpg.add_spacer(height=2, parent="inspector_content")

            # Category badge
            dpg.add_text(f"[ {node.category} ]", parent="inspector_content",
                         color=cat_color)
            dpg.add_spacer(height=4, parent="inspector_content")
            dpg.add_separator(parent="inspector_content")
            dpg.add_spacer(height=4, parent="inspector_content")

            # Description
            dpg.add_text("Description:", parent="inspector_content", color=list(TEXT_DIM))
            dpg.add_text(node.description, parent="inspector_content",
                         wrap=INSPECTOR_W - 20, color=list(TEXT))
            dpg.add_spacer(height=6, parent="inspector_content")
            dpg.add_separator(parent="inspector_content")
            dpg.add_spacer(height=4, parent="inspector_content")

            # Inputs section
            if node.inputs:
                dpg.add_text("Inputs:", parent="inspector_content", color=list(TEXT_DIM))
                dpg.add_spacer(height=2, parent="inspector_content")
                for port_name, port in node.inputs.items():
                    ptype_str = port.port_type.name
                    pin_col = list(PIN_COLORS.get(port.port_type, (160, 160, 180, 255)))
                    with dpg.group(horizontal=True, parent="inspector_content"):
                        dpg.add_text(f"  {port_name}", color=pin_col)
                        dpg.add_text(f"({ptype_str})", color=list(TEXT_DIM))
                    val = port.default_value
                    # Show current value from widget if available
                    widget_tag = self.input_widgets.get((node_id, port_name))
                    if widget_tag is not None:
                        try:
                            val = dpg.get_value(widget_tag)
                        except Exception:
                            pass
                    dpg.add_text(f"    = {val}", parent="inspector_content",
                                 color=list(TEXT))
                dpg.add_spacer(height=6, parent="inspector_content")

            # Outputs section
            if node.outputs:
                dpg.add_separator(parent="inspector_content")
                dpg.add_spacer(height=4, parent="inspector_content")
                dpg.add_text("Outputs:", parent="inspector_content", color=list(TEXT_DIM))
                dpg.add_spacer(height=2, parent="inspector_content")
                for port_name, port in node.outputs.items():
                    if port_name == "__terminal__":
                        continue
                    ptype_str = port.port_type.name
                    pin_col = list(PIN_COLORS.get(port.port_type, (160, 160, 180, 255)))
                    with dpg.group(horizontal=True, parent="inspector_content"):
                        dpg.add_text(f"  {port_name}", color=pin_col)
                        dpg.add_text(f"({ptype_str})", color=list(TEXT_DIM))
                    # Show last execution result
                    result_val = "-"
                    if node_id in self._last_outputs and port_name in self._last_outputs[node_id]:
                        raw = self._last_outputs[node_id][port_name]
                        if isinstance(raw, float):
                            result_val = f"{raw:.6g}"
                        elif hasattr(raw, "shape"):  # torch.Tensor or np.ndarray
                            result_val = f"shape={list(raw.shape)} dtype={getattr(raw, 'dtype', '?')}"
                        else:
                            result_val = str(raw)
                    dpg.add_text(f"    = {result_val}", parent="inspector_content",
                                 color=list(ACCENT))

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

    # -- Training methods --------------------------------------------------

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

    def _poll_hot_reload(self) -> None:
        """Check nodes/custom/ for new/changed files and update palette."""
        results = self._hot_reloader.poll()
        for msg, new_types in results:
            self._log(msg)
            for type_name in new_types:
                self._add_palette_button(type_name)
                # Hide the hint text once real nodes appear
                if dpg.does_item_exist("custom_hint_text"):
                    dpg.hide_item("custom_hint_text")

    def _add_palette_button(self, type_name: str) -> None:
        """Add a button for type_name to the correct palette category."""
        from nodes import NODE_REGISTRY
        cls = NODE_REGISTRY.get(type_name)
        if cls is None:
            return
        tag = f"palette_btn_{type_name}"
        if dpg.does_item_exist(tag):
            return  # already present

        cat = cls.category
        sub = getattr(cls, "subcategory", "") or ""
        sub_key = f"{cat}/{sub}" if sub else cat

        if cat not in self._palette_cat_items:
            with dpg.collapsing_header(
                label=cat, default_open=False, parent="PaletteWindow"
            ) as hdr:
                self._palette_cat_items[cat] = hdr

        if sub and sub_key not in self._palette_cat_items:
            with dpg.collapsing_header(
                label=sub, default_open=False,
                parent=self._palette_cat_items[cat]
            ) as sub_hdr:
                self._palette_cat_items[sub_key] = sub_hdr

        hdr = self._palette_cat_items[sub_key]
        btn_w = PALETTE_W - 36 if sub else PALETTE_W - 24

        dpg.add_button(
            label=cls.label,
            tag=tag,
            width=btn_w,
            callback=lambda s, a, u: self.spawn_node(u),
            user_data=type_name,
            parent=hdr,
        )
        dpg.add_spacer(height=2, parent=hdr)

    def _poll_terminal_resize(self) -> None:
        """Resize terminal content widgets to fill the (user-resizable) TerminalWindow."""
        try:
            w, h = dpg.get_item_rect_size("TerminalWindow")
        except Exception:
            return
        if (w, h) == self._term_last_size:
            return
        self._term_last_size = (w, h)
        content_h = max(60, h - 70)   # subtract toolbar + tab-bar header
        content_w = max(100, w - 30)
        try:
            dpg.set_item_height("terminal_scroll", content_h)
        except Exception:
            pass
        try:
            dpg.set_item_height("code_scroll", content_h)
            dpg.set_item_height("code_text", content_h - 10)
            dpg.set_item_width("code_text", content_w)
        except Exception:
            pass
        try:
            dpg.set_item_width("terminal_text", content_w)
            dpg.set_item_height("terminal_text", content_h - 10)
        except Exception:
            pass

    def _poll_layout(self) -> None:
        """Reanchor all panels to the viewport edges whenever the viewport or terminal is resized."""
        try:
            vw = dpg.get_viewport_client_width()
            vh = dpg.get_viewport_client_height()
        except Exception:
            return
        try:
            _, term_h = dpg.get_item_rect_size("TerminalWindow")
            term_h = max(80, term_h)
        except Exception:
            term_h = TERMINAL_H

        key = (vw, vh, term_h)
        if key == self._layout_last:
            return
        self._layout_last = key

        MENU_H   = 20
        center_x = PALETTE_W + 4
        center_w = max(200, vw - PALETTE_W - INSPECTOR_W - 8)
        right_x  = vw - INSPECTOR_W - 2
        editor_h = max(100, vh - term_h - MENU_H - 40)
        term_y   = editor_h + MENU_H + 4

        # Palette - full height, left edge
        try:
            dpg.set_item_height("PaletteWindow", vh - MENU_H - 18)
        except Exception:
            pass

        # Editor - fills centre above terminal
        try:
            dpg.set_item_pos("EditorWindow", [center_x, MENU_H])
            dpg.set_item_width("EditorWindow", center_w)
            dpg.set_item_height("EditorWindow", editor_h)
        except Exception:
            pass

        # Terminal - bottom of centre column, right-edge tracks viewport
        try:
            dpg.set_item_pos("TerminalWindow", [center_x, term_y])
            dpg.set_item_width("TerminalWindow", center_w)
        except Exception:
            pass

        # Right column - all three panels track the right edge
        try:
            dpg.set_item_pos("InspectorWindow", [right_x, MENU_H])
            dpg.set_item_pos("TrainingWindow",  [right_x, MENU_H + 284])
            dpg.set_item_pos("InferenceWindow", [right_x, MENU_H + 668])
        except Exception:
            pass

    def _poll_training(self) -> None:
        """Called each frame - drain training events, update UI."""
        lines = self._training_ctrl.poll()
        for line in lines:
            self._log(line)
        if lines:  # update plot
            losses = self._training_ctrl.train_losses
            if losses:
                try:
                    xs = list(range(1, len(losses) + 1))
                    dpg.set_value("loss_series", [xs, losses])
                    val_losses = self._training_ctrl.val_losses
                    if val_losses:
                        dpg.set_value("val_loss_series", [list(range(1, len(val_losses) + 1)), val_losses])
                    dpg.fit_axis_data("loss_x_axis")
                    dpg.fit_axis_data("loss_y_axis")
                    dpg.set_value("train_epoch_text",
                        f"Epoch: {self._training_ctrl.current_epoch}/{self._training_ctrl.total_epochs}")
                    val_str = f"  val={val_losses[-1]:.6f}" if val_losses else ""
                    dpg.set_value("train_loss_text",
                        f"Best loss: {self._training_ctrl.best_loss:.6f}{val_str}")
                    if self._training_ctrl.status in ("done", "error"):
                        status_str = f"Status: {self._training_ctrl.status.capitalize()}"
                        dpg.set_value("train_status_text", status_str)
                except Exception:
                    pass

    # -- Save/Load methods -------------------------------------------------

    def _save_graph(self) -> None:
        """Ctrl+S - save graph to JSON."""
        try:
            positions = {}
            for node_id, node_tag in self.node_id_to_dpg.items():
                try:
                    pos = dpg.get_item_pos(node_tag)
                    positions[node_id] = pos
                except Exception:
                    positions[node_id] = [100, 100]
            path = self._save_path or "graph.json"
            Serializer.save(self.graph, positions, path)
            self._save_path = path
            self._log(f"Graph saved to {path}")
        except Exception as e:
            self._log(f"Save failed: {e}")

    def _save_graph_as(self) -> None:
        """Save graph to a user-chosen path."""
        import tkinter as tk
        from tkinter import filedialog
        try:
            root = tk.Tk()
            root.withdraw()
            path = filedialog.asksaveasfilename(
                title="Save Graph As",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            )
            root.destroy()
            if not path:
                return
            self._save_path = path
            self._save_graph()
        except Exception as e:
            self._log(f"Save As failed: {e}")

    def _load_graph(self) -> None:
        """Ctrl+O - load graph from JSON."""
        import tkinter as tk
        from tkinter import filedialog
        try:
            root = tk.Tk()
            root.withdraw()
            path = filedialog.askopenfilename(
                title="Load Graph",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            root.destroy()
            if not path:
                return
            graph, positions = Serializer.load(path)
            # Clear existing editor (clears self.graph in-place)
            self._clear_editor()
            # Populate the same graph object the executor already holds
            for node in graph.nodes.values():
                self.graph.add_node(node)
            for conn in graph.connections:
                self.graph.add_connection(
                    conn.from_node_id, conn.from_port,
                    conn.to_node_id, conn.to_port
                )
            # Rebuild DPG nodes
            for node_id, node in graph.nodes.items():
                pos = positions.get(node_id, [100, 100])
                self.add_node_to_editor(node, tuple(pos))
            # Rebuild DPG links
            for conn in graph.connections:
                from_attr = f"attr_out_{conn.from_node_id}_{conn.from_port}"
                to_attr = f"attr_in_{conn.to_node_id}_{conn.to_port}"
                if dpg.does_item_exist(from_attr) and dpg.does_item_exist(to_attr):
                    lt = dpg.add_node_link(from_attr, to_attr, parent="NodeEditor")
                    self.dpg_link_to_conn[lt] = (conn.from_node_id, conn.from_port, conn.to_node_id, conn.to_port)
            self._log(f"Graph loaded from {path}")
        except Exception as e:
            import traceback
            self._log(f"Load failed: {traceback.format_exc()}")

    def _refresh_code_panel(self) -> None:
        """Regenerate the export script and show it in the Code tab."""
        try:
            from core.io import GraphExporter
            script = GraphExporter().export(self.graph)
            dpg.set_value("code_text", script)
        except Exception as e:
            try:
                dpg.set_value("code_text", f"# Code generation error: {e}")
            except Exception:
                pass

    def _export_script(self) -> None:
        """Ctrl+E - export graph to a runnable Python script."""
        import tkinter as tk
        from tkinter import filedialog
        try:
            root = tk.Tk()
            root.withdraw()
            path = filedialog.asksaveasfilename(
                title="Export Python Script",
                defaultextension=".py",
                filetypes=[("Python files", "*.py"), ("All files", "*.*")]
            )
            root.destroy()
            if not path:
                return
            from core.io import GraphExporter
            script = GraphExporter().export(self.graph)
            with open(path, "w", encoding="utf-8") as f:
                f.write(script)
            self._log(f"Script exported to {path}")
            try:
                dpg.set_value("code_text", script)
            except Exception:
                pass
        except Exception as e:
            import traceback
            self._log(f"Export failed: {traceback.format_exc()}")

    def _clear_editor(self) -> None:
        """Delete all nodes and links from DPG, clear all dicts, and reset the graph."""
        # Delete all links first
        for lt in list(self.dpg_link_to_conn.keys()):
            try:
                if dpg.does_item_exist(lt):
                    dpg.delete_item(lt)
            except Exception:
                pass
        self.dpg_link_to_conn.clear()
        # Delete all nodes
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
        # Clear graph in-place so executor and code panel stay in sync
        self.graph.clear()

    # -- Copy/paste --------------------------------------------------------

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
        # Make positions relative to centroid
        cx = sum(p[0] for p in positions) / len(positions)
        cy = sum(p[1] for p in positions) / len(positions)
        for entry in self._clipboard:
            entry["rel_pos"] = [entry["pos"][0] - cx, entry["pos"][1] - cy]
        self._log(f"Copied {len(self._clipboard)} node(s)")

    def _paste_nodes(self) -> None:
        if not self._clipboard:
            return
        ox, oy = 400, 300  # default paste position
        id_map = {}  # orig_id -> new node_id
        for entry in self._clipboard:
            cls = NODE_REGISTRY.get(entry["type_name"])
            if cls is None:
                continue
            node = cls()
            # Restore primitive input defaults
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

    # -- Palette search ----------------------------------------------------

    def _filter_palette(self, text: str) -> None:
        text = text.lower().strip()
        for type_name, cls in NODE_REGISTRY.items():
            btn_tag = f"palette_btn_{type_name}"
            if dpg.does_item_exist(btn_tag):
                if not text or text in type_name.lower() or text in cls.label.lower():
                    dpg.show_item(btn_tag)
                else:
                    dpg.hide_item(btn_tag)

    # -- Request processing (between frames) -------------------------------

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

    # -- Demo graph --------------------------------------------------------
    # Demo graph only uses existing Math/Data nodes (FloatConst, Add, Multiply, Print)

    def _build_demo_graph(self) -> None:
        """Pre-populate with the MNIST digit recognition demo.

        Each layer node owns its weights (self._layer). Two wires per layer:
          - tensor_out: live forward pass through that layers current weights
          - model: assembles the Sequential chain for the training loop

        After each training epoch the graph re-executes automatically, so
        tensor_out values update to reflect the newly trained weights.

          MNIST (train) --dataloader--> Training Config
                        --dataloader--> Sample Batch --x--> Flatten --tensor_out--> Lin1 --tensor_out--> Lin2 --tensor_out--> Lin3
          MNIST (val)   --dataloader--> Training Config
          Flatten --model--> Lin1 --model--> Lin2 --model--> Lin3 --model--> Training Config
        """
        from nodes.pytorch.layers import FlattenNode, LinearNode
        from nodes.pytorch.data import MNISTDatasetNode, SampleBatchNode
        from nodes.pytorch.training import TrainingConfigNode

        # -- Architecture nodes --------------------------------------------
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

        # -- Data nodes ----------------------------------------------------
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

        # -- Training Config -----------------------------------------------
        cfg = TrainingConfigNode()
        cfg.inputs["epochs"].default_value    = 5
        cfg.inputs["device"].default_value    = "cpu"
        cfg.inputs["optimizer"].default_value = "adam"
        cfg.inputs["lr"].default_value        = 0.001
        cfg.inputs["loss"].default_value      = "crossentropy"
        cfg.inputs["scheduler"].default_value = "none"
        self.graph.add_node(cfg)

        # -- Connections ---------------------------------------------------
        self.graph.add_connection(mnist_train.id, "dataloader", cfg.id,    "dataloader")
        self.graph.add_connection(mnist_train.id, "dataloader", sample.id, "dataloader")
        self.graph.add_connection(mnist_val.id,   "dataloader", cfg.id,    "val_dataloader")
        self.graph.add_connection(sample.id,      "x",          flatten.id, "tensor_in")
        self.graph.add_connection(flatten.id,     "tensor_out", lin1.id,    "tensor_in")
        self.graph.add_connection(lin1.id,        "tensor_out", lin2.id,    "tensor_in")
        self.graph.add_connection(lin2.id,        "tensor_out", lin3.id,    "tensor_in")
        self.graph.add_connection(lin3.id,        "tensor_out", cfg.id,     "tensor_in")

        # -- Positions -----------------------------------------------------
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
        all_nodes = [mnist_train, mnist_val, sample, flatten, lin1, lin2, lin3, cfg]
        for node in all_nodes:
            self.add_node_to_editor(node, positions[node.id])

        # -- DPG visual links ----------------------------------------------
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

    # -- Layout ------------------------------------------------------------

    def _build_layout(self) -> None:
        """Create all DPG windows and the node editor."""
        # Create texture registry early (needed for inline Viz node images)
        with dpg.texture_registry(tag="__tex_registry__"):
            pass

        MENU_H = 20   # viewport menu bar height

        editor_x = PALETTE_W + 4
        editor_w = VIEWPORT_W - PALETTE_W - INSPECTOR_W - 8
        editor_h = VIEWPORT_H - TERMINAL_H - 40 - MENU_H
        inspector_x = PALETTE_W + 4 + editor_w + 4

        # -- Palette header themes -----------------------------------------
        with dpg.theme() as _cat_theme:
            with dpg.theme_component(dpg.mvCollapsingHeader):
                dpg.add_theme_color(dpg.mvThemeCol_Header,        (40,  42,  70,  255))
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered,  (55,  58,  95,  255))
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive,   (70,  74, 118,  255))
        self._cat_header_theme = _cat_theme

        with dpg.theme() as _sub_theme:
            with dpg.theme_component(dpg.mvCollapsingHeader):
                dpg.add_theme_color(dpg.mvThemeCol_Header,        (28,  35,  45,  255))
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered,  (38,  48,  62,  255))
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive,   (48,  60,  78,  255))
        self._sub_header_theme = _sub_theme

        # -- File menu bar -------------------------------------------------
        with dpg.viewport_menu_bar():
            with dpg.menu(label="File"):
                dpg.add_menu_item(label="New",         shortcut="Ctrl+N",
                                  callback=lambda: self._clear_editor())
                dpg.add_separator()
                dpg.add_menu_item(label="Open Graph...", shortcut="Ctrl+O",
                                  callback=lambda: self._load_graph())
                dpg.add_menu_item(label="Save Graph",  shortcut="Ctrl+S",
                                  callback=lambda: self._save_graph())
                dpg.add_menu_item(label="Save Graph As...",
                                  callback=lambda: self._save_graph_as())
                dpg.add_separator()
                dpg.add_menu_item(label="Export Script...", shortcut="Ctrl+E",
                                  callback=lambda: self._export_script())
            with dpg.menu(label="Edit"):
                dpg.add_menu_item(label="Undo",  shortcut="Ctrl+Z",
                                  callback=lambda: setattr(self, "_undo_requested", True))
                dpg.add_menu_item(label="Redo",  shortcut="Ctrl+Y",
                                  callback=lambda: setattr(self, "_redo_requested", True))
                dpg.add_separator()
                dpg.add_menu_item(label="Copy",  shortcut="Ctrl+C",
                                  callback=lambda: setattr(self, "_copy_requested",  True))
                dpg.add_menu_item(label="Paste", shortcut="Ctrl+V",
                                  callback=lambda: setattr(self, "_paste_requested", True))

        # -- Palette window ------------------------------------------------
        with dpg.window(
            label="Node Palette",
            tag="PaletteWindow",
            pos=[0, MENU_H],
            width=PALETTE_W,
            height=VIEWPORT_H - 20,
            no_close=True,
            no_collapse=True,
        ):
            dpg.add_text("NODES", color=list(ACCENT))
            dpg.add_separator()
            dpg.add_spacer(height=4)

            # Search box
            dpg.add_input_text(
                tag="palette_search",
                hint="Search nodes...",
                width=-1,
                callback=lambda s, a: self._filter_palette(a),
            )
            dpg.add_spacer(height=4)

            categories = get_nodes_by_category()
            for cat in CATEGORY_ORDER:
                if cat not in categories:
                    continue

                nodes_in_cat = categories[cat]
                # Check if any node in this category uses subcategories
                has_subs = any(getattr(cls, "subcategory", "") for cls in nodes_in_cat)

                with dpg.collapsing_header(label=cat, default_open=False) as hdr:
                    self._palette_cat_items[cat] = hdr
                    dpg.bind_item_theme(hdr, self._cat_header_theme)

                    if not has_subs:
                        for cls in nodes_in_cat:
                            dpg.add_button(
                                label=cls.label,
                                tag=f"palette_btn_{cls.type_name}",
                                width=PALETTE_W - 24,
                                callback=lambda s, a, u: self.spawn_node(u),
                                user_data=cls.type_name,
                            )
                            dpg.add_spacer(height=2)
                    else:
                        # Group by subcategory, preserving insertion order
                        sub_groups: dict[str, list] = {}
                        for cls in nodes_in_cat:
                            sub = getattr(cls, "subcategory", "") or ""
                            sub_groups.setdefault(sub, []).append(cls)
                        for sub_label, sub_nodes in sub_groups.items():
                            sub_key = f"{cat}/{sub_label}"
                            dpg.add_spacer(height=1)
                            with dpg.collapsing_header(
                                label=f"  > {sub_label}", default_open=False,
                                indent=8,
                            ) as sub_hdr:
                                self._palette_cat_items[sub_key] = sub_hdr
                                dpg.bind_item_theme(sub_hdr, self._sub_header_theme)
                                for cls in sub_nodes:
                                    dpg.add_button(
                                        label=cls.label,
                                        tag=f"palette_btn_{cls.type_name}",
                                        width=PALETTE_W - 48,
                                        callback=lambda s, a, u: self.spawn_node(u),
                                        user_data=cls.type_name,
                                    )
                                    dpg.add_spacer(height=2)
                dpg.add_spacer(height=4)

            # Custom (hot-reload) section - starts empty, populated at runtime
            with dpg.collapsing_header(label="Custom", default_open=False) as hdr:
                self._palette_cat_items["Custom"] = hdr
                dpg.add_text("Drop .py files in nodes/custom/", color=list(TEXT_DIM),
                             tag="custom_hint_text", wrap=PALETTE_W - 24)
            dpg.add_spacer(height=4)

            dpg.add_separator()
            dpg.add_spacer(height=6)
            dpg.add_text("Hotkeys:", color=list(TEXT_DIM))
            dpg.add_text("Del - delete nodes or links", color=list(TEXT_DIM))
            dpg.add_text("Right-click - context menu", color=list(TEXT_DIM))
            dpg.add_text("Ctrl+Z / Y - undo / redo", color=list(TEXT_DIM))
            dpg.add_text("Ctrl+C / V - copy / paste", color=list(TEXT_DIM))
            dpg.add_text("Ctrl+S - save graph", color=list(TEXT_DIM))
            dpg.add_text("Ctrl+E - export to .py", color=list(TEXT_DIM))

        # -- Node Editor window --------------------------------------------
        with dpg.window(
            label="Node Editor",
            tag="EditorWindow",
            pos=[PALETTE_W + 4, MENU_H],
            width=editor_w,
            height=editor_h,
            no_close=True,
            no_collapse=True,
        ):
            with dpg.node_editor(
                tag="NodeEditor",
                callback=self._link_callback,
                delink_callback=self._delink_callback,
                minimap=True,
                minimap_location=dpg.mvNodeMiniMap_Location_BottomRight,
            ):
                pass  # nodes added dynamically

        # -- Inspector window ----------------------------------------------
        inspector_height = 280
        with dpg.window(
            label="Inspector",
            tag="InspectorWindow",
            pos=[inspector_x, MENU_H],
            width=INSPECTOR_W,
            height=inspector_height,
            no_close=True,
            no_collapse=True,
        ):
            dpg.add_text("INSPECTOR", color=list(ACCENT))
            dpg.add_separator()
            dpg.add_spacer(height=4)
            with dpg.child_window(tag="inspector_content", autosize_x=True,
                                  height=inspector_height - 60, border=False):
                dpg.add_text("Click a node to inspect.", color=list(TEXT_DIM))

        # -- Training Panel window -----------------------------------------
        training_y = inspector_height + MENU_H + 4
        training_h = 380
        with dpg.window(
            label="Training",
            tag="TrainingWindow",
            pos=[inspector_x, training_y],
            width=INSPECTOR_W,
            height=training_h,
            no_close=True,
            no_collapse=True,
        ):
            dpg.add_text("Training", color=list(ACCENT))
            dpg.add_separator()
            dpg.add_spacer(height=4)
            dpg.add_text("Status: Idle", tag="train_status_text", color=list(TEXT_DIM))
            dpg.add_spacer(height=4)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Start", tag="train_start_btn",
                               callback=lambda: self._train_start())
                dpg.add_button(label="Pause", tag="train_pause_btn",
                               callback=lambda: self._train_pause_resume())
                dpg.add_button(label="Stop", tag="train_stop_btn",
                               callback=lambda: self._train_stop())
            dpg.add_button(label="Check Wiring", width=-1,
                           callback=lambda: self._train_check_wiring())
            dpg.add_spacer(height=4)
            dpg.add_text("Epoch: 0/0", tag="train_epoch_text", color=list(TEXT))
            dpg.add_text("Best loss: -", tag="train_loss_text", color=list(TEXT))
            dpg.add_spacer(height=4)
            with dpg.plot(label="Loss", height=150, width=-1, tag="loss_plot"):
                dpg.add_plot_axis(dpg.mvXAxis, label="Epoch", tag="loss_x_axis")
                with dpg.plot_axis(dpg.mvYAxis, label="Loss", tag="loss_y_axis"):
                    dpg.add_line_series([], [], label="train", tag="loss_series")
                    dpg.add_line_series([], [], label="val", tag="val_loss_series")
            dpg.add_spacer(height=4)
            dpg.add_button(label="Save Model", tag="save_model_btn",
                           callback=lambda: self._save_model())

        # -- Terminal window -----------------------------------------------
        term_y = editor_h + MENU_H + 4
        term_w = VIEWPORT_W - PALETTE_W - INSPECTOR_W - 8
        with dpg.window(
            label="Terminal",
            tag="TerminalWindow",
            pos=[PALETTE_W + 4, term_y],
            width=term_w,
            height=TERMINAL_H,
            no_close=True,
            no_collapse=True,
        ):
            with dpg.group(horizontal=True):
                run_btn = dpg.add_button(
                    label="  >  Run Graph  ",
                    callback=lambda: self.run_graph(),
                    height=32,
                )
                dpg.add_spacer(width=8)
                dpg.add_button(
                    label="  Export .py  ",
                    callback=lambda: self._export_script(),
                    height=32,
                )
                dpg.add_spacer(width=8)
                clear_btn = dpg.add_button(
                    label="  Clear All  ",
                    callback=lambda: self._clear_all(),
                    height=32,
                )

            dpg.add_spacer(height=4)
            tab_content_h = TERMINAL_H - 70

            with dpg.tab_bar(tag="terminal_tab_bar"):
                with dpg.tab(label="Output", tag="tab_output"):
                    with dpg.child_window(tag="terminal_scroll", autosize_x=True,
                                          height=tab_content_h, border=False):
                        dpg.add_input_text(
                            tag="terminal_text",
                            default_value="",
                            multiline=True,
                            readonly=True,
                            width=-1,
                            height=tab_content_h - 10,
                        )

                with dpg.tab(label="Code", tag="tab_code"):
                    with dpg.child_window(tag="code_scroll", autosize_x=True,
                                          height=tab_content_h, border=False):
                        dpg.add_input_text(
                            tag="code_text",
                            default_value="# Run the graph to generate code preview.",
                            multiline=True,
                            readonly=True,
                            width=term_w - 30,
                            height=tab_content_h - 10,
                            tab_input=True,
                        )

            # Apply colored button themes
            dpg.bind_item_theme(run_btn, create_run_button_theme())
            dpg.bind_item_theme(clear_btn, create_clear_button_theme())

    # -- Event handlers ----------------------------------------------------

    def _handle_zoom(self, sender, app_data) -> None:
        """Ctrl+scroll to zoom the canvas by rescaling all node positions."""
        if not (dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl)):
            return
        # app_data is the scroll delta (positive = up = zoom in)
        delta = app_data
        factor = 1.1 if delta > 0 else (1 / 1.1)
        new_zoom = max(0.2, min(3.0, self._zoom * factor))
        if new_zoom == self._zoom:
            return

        # Get mouse position as zoom pivot
        mx, my = dpg.get_mouse_pos(local=False)
        # Convert to canvas coords at current zoom
        try:
            ex, ey = dpg.get_item_pos("EditorWindow")
        except Exception:
            ex, ey = 0, 0
        pivot_x = (mx - ex) / self._zoom
        pivot_y = (my - ey) / self._zoom

        self._zoom = new_zoom

        # Sync base positions from actual current DPG positions (user may have dragged nodes)
        for node_id, node_tag in self.node_id_to_dpg.items():
            try:
                cx, cy = dpg.get_item_pos(node_tag)
                self._node_base_pos[node_id] = [cx / (new_zoom / factor),
                                                cy / (new_zoom / factor)]
            except Exception:
                pass

        # Reposition every node: scale from pivot
        for node_id, node_tag in self.node_id_to_dpg.items():
            base = self._node_base_pos.get(node_id)
            if base is None:
                continue
            try:
                nx = pivot_x + (base[0] - pivot_x) * new_zoom
                ny = pivot_y + (base[1] - pivot_y) * new_zoom
                dpg.set_item_pos(node_tag, [int(nx), int(ny)])
            except Exception:
                pass

    def _handle_undo(self):
        if dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl):
            self._undo_requested = True

    def _handle_redo(self):
        if dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl):
            self._redo_requested = True

    def _handle_copy(self):
        if dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl):
            self._copy_requested = True

    def _handle_paste(self):
        if dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl):
            self._paste_requested = True

    def _handle_save(self):
        if dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl):
            self._save_requested = True

    def _handle_export(self):
        if dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl):
            self._export_requested = True

    def _setup_handlers(self) -> None:
        """Register keyboard and mouse handlers."""
        with dpg.handler_registry():
            dpg.add_key_press_handler(
                key=dpg.mvKey_Delete,
                callback=lambda: self.delete_selected(),
            )
            dpg.add_key_press_handler(key=dpg.mvKey_Z, callback=self._handle_undo)
            dpg.add_key_press_handler(key=dpg.mvKey_Y, callback=self._handle_redo)
            dpg.add_key_press_handler(key=dpg.mvKey_C, callback=self._handle_copy)
            dpg.add_key_press_handler(key=dpg.mvKey_V, callback=self._handle_paste)
            dpg.add_key_press_handler(key=dpg.mvKey_S, callback=self._handle_save)
            dpg.add_key_press_handler(key=dpg.mvKey_E, callback=self._handle_export)
            dpg.add_mouse_click_handler(
                button=1,  # right mouse
                callback=self._show_context_menu,
            )
            dpg.add_mouse_wheel_handler(callback=self._handle_zoom)

    # -- Main run ----------------------------------------------------------

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
