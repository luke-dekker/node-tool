"""HandlersMixin — keyboard and mouse event handlers."""
from __future__ import annotations
import dearpygui.dearpygui as dpg


class HandlersMixin:
    """Keyboard shortcuts and mouse handlers."""

    def _handle_zoom(self, sender, app_data) -> None:
        """Scroll wheel to zoom the canvas. Nodes visually shrink/grow via
        global font scale + position rescaling. Hold Ctrl to pan instead."""
        # If Ctrl is held, let DPG handle normal scroll (pan). Only zoom on plain scroll.
        if dpg.is_key_down(dpg.mvKey_LControl) or dpg.is_key_down(dpg.mvKey_RControl):
            return

        # Only zoom when mouse is over the editor area
        mx, my = dpg.get_mouse_pos(local=False)
        try:
            ex, ey = dpg.get_item_pos("EditorWindow")
            ew = dpg.get_item_width("EditorWindow")
            eh = dpg.get_item_height("EditorWindow")
        except Exception:
            return
        if not (ex <= mx <= ex + ew and ey <= my <= ey + eh):
            return

        delta = app_data
        factor = 1.1 if delta > 0 else (1 / 1.1)
        new_zoom = max(0.3, min(2.5, self._zoom * factor))
        if new_zoom == self._zoom:
            return

        pivot_x = (mx - ex) / self._zoom
        pivot_y = (my - ey) / self._zoom

        self._zoom = new_zoom

        # Update base positions from current DPG positions
        for node_id, node_tag in self.node_id_to_dpg.items():
            try:
                cx, cy = dpg.get_item_pos(node_tag)
                self._node_base_pos[node_id] = [cx / (new_zoom / factor),
                                                cy / (new_zoom / factor)]
            except Exception:
                pass

        # Rescale node positions around pivot
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
            dpg.add_key_press_handler(key=dpg.mvKey_Delete,
                                      callback=lambda: self.delete_selected())
            dpg.add_key_press_handler(key=dpg.mvKey_Z, callback=self._handle_undo)
            dpg.add_key_press_handler(key=dpg.mvKey_Y, callback=self._handle_redo)
            dpg.add_key_press_handler(key=dpg.mvKey_C, callback=self._handle_copy)
            dpg.add_key_press_handler(key=dpg.mvKey_V, callback=self._handle_paste)
            dpg.add_key_press_handler(key=dpg.mvKey_S, callback=self._handle_save)
            dpg.add_key_press_handler(key=dpg.mvKey_E, callback=self._handle_export)
            dpg.add_mouse_click_handler(button=1, callback=self._show_context_menu)
            dpg.add_mouse_wheel_handler(callback=self._handle_zoom)
