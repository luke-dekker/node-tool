"""PollingMixin — per-frame polling: hot-reload, layout, terminal, training."""
from __future__ import annotations
import dearpygui.dearpygui as dpg
from gui.constants import PALETTE_W, INSPECTOR_W, TERMINAL_H


class PollingMixin:
    """Per-frame polling methods called in the render loop."""

    def _poll_hot_reload(self) -> None:
        """Check nodes/custom/ for new/changed files and update palette."""
        results = self._hot_reloader.poll()
        for msg, new_types in results:
            self._log(msg)
            for type_name in new_types:
                self._add_palette_button(type_name)
                if dpg.does_item_exist("custom_hint_text"):
                    dpg.hide_item("custom_hint_text")

    def _poll_subgraph_reload(self) -> None:
        """Check subgraphs/ for new/changed/removed .subgraph.json files.

        Each event mutates NODE_REGISTRY and refreshes the palette in place so
        edits made to a subgraph file in another window show up live without
        restarting the app. Existing canvas instances of a changed subgraph are
        not refreshed automatically — re-spawn them to pick up new ports.
        """
        from nodes.subgraphs._reloader import SubgraphReloader
        events = self._subgraph_reloader.poll()
        for kind, type_name, msg, _cls in events:
            self._log(msg)
            SubgraphReloader.apply_event((kind, type_name, msg, _cls))
            if kind == "added":
                self._add_palette_button(type_name)
            elif kind == "changed":
                # Re-create the button so the label updates if it was renamed
                self._remove_palette_button(type_name)
                self._add_palette_button(type_name)
            elif kind == "removed":
                self._remove_palette_button(type_name)

    def _remove_palette_button(self, type_name: str) -> None:
        tag = f"palette_btn_{type_name}"
        try:
            if dpg.does_item_exist(tag):
                dpg.delete_item(tag)
        except Exception:
            pass

    def _add_palette_button(self, type_name: str) -> None:
        """Add a button for type_name to the correct palette category."""
        from nodes import NODE_REGISTRY
        cls = NODE_REGISTRY.get(type_name)
        if cls is None:
            return
        tag = f"palette_btn_{type_name}"
        if dpg.does_item_exist(tag):
            return

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
        content_h = max(60, h - 36)
        content_w = max(100, w - 30)
        try:
            dpg.set_item_height("terminal_scroll", content_h)
        except Exception:
            pass
        try:
            # Code tab has a button row above the text area, leave room for it
            dpg.set_item_height("code_scroll", content_h)
            dpg.set_item_height("code_text", content_h - 38)
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
        BOTTOM   = 4           # uniform margin from viewport bottom
        usable_h = vh - MENU_H - BOTTOM
        center_x = PALETTE_W + 4
        center_w = max(200, vw - PALETTE_W - INSPECTOR_W - 8)
        right_x  = vw - INSPECTOR_W - 2
        editor_h = max(100, usable_h - term_h - 4)
        term_y   = MENU_H + editor_h + 4

        # Palette — full left column
        try:
            dpg.set_item_height("PaletteWindow", usable_h)
        except Exception:
            pass
        # Editor — centre, above terminal
        try:
            dpg.set_item_pos("EditorWindow", [center_x, MENU_H])
            dpg.set_item_width("EditorWindow", center_w)
            dpg.set_item_height("EditorWindow", editor_h)
        except Exception:
            pass
        # Terminal — centre, below editor
        try:
            dpg.set_item_pos("TerminalWindow", [center_x, term_y])
            dpg.set_item_width("TerminalWindow", center_w)
            dpg.set_item_height("TerminalWindow", term_h)
        except Exception:
            pass
        # Right column — inspector + training fill to same bottom edge
        try:
            training_h_actual = 380
            insp_h = max(180, usable_h - training_h_actual - 4)
            training_h_final = usable_h - insp_h - 4
            dpg.set_item_pos("InspectorWindow", [right_x, MENU_H])
            dpg.set_item_height("InspectorWindow", insp_h)
            try:
                dpg.set_item_height("inspector_content", insp_h - 60)
            except Exception:
                pass
            training_y = MENU_H + insp_h + 4
            dpg.set_item_pos("TrainingWindow", [right_x, training_y])
            dpg.set_item_height("TrainingWindow", training_h_final)
        except Exception:
            pass

    def _poll_training(self) -> None:
        """Called each frame — drain training events, update UI."""
        lines = self._training_ctrl.poll()
        for line in lines:
            self._log(line)
        if lines:
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
                        f"Epoch  {self._training_ctrl.current_epoch} / {self._training_ctrl.total_epochs}")
                    val_str = f"   val {val_losses[-1]:.4f}" if val_losses else ""
                    dpg.set_value("train_loss_text",
                        f"Best loss  {self._training_ctrl.best_loss:.4f}{val_str}")
                    if self._training_ctrl.status in ("done", "error"):
                        from gui.mixins.training import _set_train_status
                        _set_train_status(self._training_ctrl.status.capitalize())
                except Exception:
                    pass
