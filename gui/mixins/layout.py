"""LayoutMixin — build all DearPyGui windows and the node editor."""
from __future__ import annotations
import dearpygui.dearpygui as dpg
from gui.constants import VIEWPORT_W, VIEWPORT_H, PALETTE_W, INSPECTOR_W, TERMINAL_H
from gui.theme import (
    create_run_button_theme, create_clear_button_theme,
    create_section_header_theme, create_palette_button_theme,
    ACCENT, TEXT, TEXT_DIM, TEXT_BRIGHT, OK_GREEN,
)
from nodes import get_nodes_by_category, CATEGORY_ORDER


def _section_header(label: str, header_theme: int, subtitle: str | None = None) -> None:
    """Render a styled section header bar inside a window."""
    with dpg.child_window(height=32, border=True, autosize_x=True) as hdr:
        with dpg.group(horizontal=True):
            dpg.add_text(label, color=list(TEXT_BRIGHT))
            if subtitle:
                dpg.add_text(subtitle, color=list(TEXT_DIM))
    dpg.bind_item_theme(hdr, header_theme)
    dpg.add_spacer(height=6)


class LayoutMixin:
    """Builds the initial DearPyGui layout: palette, editor, inspector, training, terminal."""

    def _build_layout(self) -> None:
        """Create all DPG windows and the node editor."""
        with dpg.texture_registry(tag="__tex_registry__"):
            pass

        MENU_H   = 20
        BOTTOM   = 4
        usable_h = VIEWPORT_H - MENU_H - BOTTOM
        editor_x = PALETTE_W + 4
        editor_w = VIEWPORT_W - PALETTE_W - INSPECTOR_W - 8
        editor_h = usable_h - TERMINAL_H - 4
        inspector_x = PALETTE_W + 4 + editor_w + 4

        # Palette header themes
        with dpg.theme() as _cat_theme:
            with dpg.theme_component(dpg.mvCollapsingHeader):
                dpg.add_theme_color(dpg.mvThemeCol_Header,        (40, 48, 66, 255))
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (56, 68, 92, 255))
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive,  (72, 88, 118, 255))
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding,  10.0, 6.0)
        self._cat_header_theme = _cat_theme

        with dpg.theme() as _sub_theme:
            with dpg.theme_component(dpg.mvCollapsingHeader):
                dpg.add_theme_color(dpg.mvThemeCol_Header,        (30, 36, 50, 255))
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, (44, 54, 74, 255))
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive,  (56, 68, 92, 255))
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding,  8.0, 4.0)
        self._sub_header_theme = _sub_theme

        # Reusable themes for section headers and palette buttons
        self._section_header_theme = create_section_header_theme()
        self._palette_btn_theme    = create_palette_button_theme()

        # Menu bar with File, Edit, and action buttons
        with dpg.viewport_menu_bar():
            with dpg.menu(label="File"):
                dpg.add_menu_item(label="New",              shortcut="Ctrl+N",
                                  callback=lambda: self._clear_editor())
                dpg.add_separator()
                dpg.add_menu_item(label="Open Graph...",    shortcut="Ctrl+O",
                                  callback=lambda: self._load_graph())
                dpg.add_menu_item(label="Save Graph",       shortcut="Ctrl+S",
                                  callback=lambda: self._save_graph())
                dpg.add_menu_item(label="Save Graph As...",
                                  callback=lambda: self._save_graph_as())
                dpg.add_separator()
                with dpg.menu(label="Templates"):
                    try:
                        from templates import TEMPLATES
                        for label, description, builder in TEMPLATES:
                            # Bind builder via default arg to avoid late-binding closure bug
                            dpg.add_menu_item(
                                label=label,
                                callback=lambda s, a, u=(builder, label): self._load_template(u[0], u[1]),
                            )
                            with dpg.tooltip(dpg.last_item()):
                                dpg.add_text(description, wrap=320)
                    except Exception as exc:
                        dpg.add_menu_item(label=f"(load failed: {exc})", enabled=False)
                dpg.add_separator()
                dpg.add_menu_item(label="Export Script...", shortcut="Ctrl+E",
                                  callback=lambda: self._export_script())
                dpg.add_menu_item(label="Export As Class...",
                                  callback=lambda: self._export_class())
                dpg.add_separator()
                dpg.add_menu_item(label="Pack Selection as Subgraph...",
                                  callback=lambda: self._pack_as_subgraph())
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
            # Action buttons in the menu bar
            dpg.add_spacer(width=24)
            run_btn = dpg.add_button(label="  Run Graph  ", callback=lambda: self.run_graph())
            dpg.add_spacer(width=8)
            export_btn = dpg.add_button(label="  Export .py  ", callback=lambda: self._export_script())
            dpg.add_spacer(width=8)
            clear_btn = dpg.add_button(label="  Clear All  ", callback=lambda: self._clear_all())
            dpg.bind_item_theme(run_btn,   create_run_button_theme())
            dpg.bind_item_theme(clear_btn, create_clear_button_theme())

        # Palette window
        with dpg.window(
            label="Node Palette", tag="PaletteWindow",
            pos=[0, MENU_H], width=PALETTE_W, height=usable_h,
            no_close=True, no_collapse=True,
        ):
            _section_header("Nodes", self._section_header_theme)
            dpg.add_input_text(
                tag="palette_search", hint=" Search nodes...", width=-1,
                callback=lambda s, a: self._filter_palette(a),
            )
            dpg.add_spacer(height=8)

            categories = get_nodes_by_category()
            for cat in CATEGORY_ORDER:
                if cat not in categories:
                    continue
                nodes_in_cat = categories[cat]
                has_subs = any(getattr(cls, "subcategory", "") for cls in nodes_in_cat)
                with dpg.collapsing_header(label=cat, default_open=False) as hdr:
                    self._palette_cat_items[cat] = hdr
                    dpg.bind_item_theme(hdr, self._cat_header_theme)
                    if not has_subs:
                        for cls in nodes_in_cat:
                            btn = dpg.add_button(
                                label=cls.label,
                                tag=f"palette_btn_{cls.type_name}",
                                width=PALETTE_W - 24,
                                callback=lambda s, a, u: self.spawn_node(u),
                                user_data=cls.type_name,
                            )
                            dpg.bind_item_theme(btn, self._palette_btn_theme)
                            dpg.add_spacer(height=1)
                    else:
                        sub_groups: dict[str, list] = {}
                        for cls in nodes_in_cat:
                            sub = getattr(cls, "subcategory", "") or ""
                            sub_groups.setdefault(sub, []).append(cls)
                        for sub_label, sub_nodes in sub_groups.items():
                            sub_key = f"{cat}/{sub_label}"
                            dpg.add_spacer(height=1)
                            with dpg.collapsing_header(
                                label=sub_label, default_open=False, indent=8,
                            ) as sub_hdr:
                                self._palette_cat_items[sub_key] = sub_hdr
                                dpg.bind_item_theme(sub_hdr, self._sub_header_theme)
                                for cls in sub_nodes:
                                    btn = dpg.add_button(
                                        label=cls.label,
                                        tag=f"palette_btn_{cls.type_name}",
                                        width=PALETTE_W - 48,
                                        callback=lambda s, a, u: self.spawn_node(u),
                                        user_data=cls.type_name,
                                    )
                                    dpg.bind_item_theme(btn, self._palette_btn_theme)
                                    dpg.add_spacer(height=1)
                dpg.add_spacer(height=4)

            with dpg.collapsing_header(label="Custom", default_open=False) as hdr:
                self._palette_cat_items["Custom"] = hdr
                dpg.add_text("Drop .py files in nodes/custom/", color=list(TEXT_DIM),
                             tag="custom_hint_text", wrap=PALETTE_W - 24)
            dpg.add_spacer(height=4)
            dpg.add_separator()
            dpg.add_spacer(height=6)
            dpg.add_text("Hotkeys:", color=list(TEXT_DIM))
            dpg.add_text("Del - delete nodes or links",   color=list(TEXT_DIM))
            dpg.add_text("Right-click - context menu",    color=list(TEXT_DIM))
            dpg.add_text("Ctrl+Z / Y - undo / redo",     color=list(TEXT_DIM))
            dpg.add_text("Ctrl+C / V - copy / paste",    color=list(TEXT_DIM))
            dpg.add_text("Ctrl+S - save graph",          color=list(TEXT_DIM))
            dpg.add_text("Ctrl+E - export to .py",       color=list(TEXT_DIM))

        # Node Editor window
        with dpg.window(
            label="Node Editor", tag="EditorWindow",
            pos=[PALETTE_W + 4, MENU_H], width=editor_w, height=editor_h,
            no_close=True, no_collapse=True,
        ):
            with dpg.node_editor(
                tag="NodeEditor",
                callback=self._link_callback,
                delink_callback=self._delink_callback,
                minimap=True,
                minimap_location=dpg.mvNodeMiniMap_Location_BottomRight,
            ):
                pass

        # Inspector window — fills right column above training panel
        training_h = 380
        inspector_height = usable_h - training_h - 4
        with dpg.window(
            label="Inspector", tag="InspectorWindow",
            pos=[inspector_x, MENU_H], width=INSPECTOR_W, height=inspector_height,
            no_close=True, no_collapse=True,
        ):
            _section_header("Inspector", self._section_header_theme)
            with dpg.child_window(tag="inspector_content", autosize_x=True,
                                  height=inspector_height - 70, border=False):
                dpg.add_text("Select a node to inspect.", color=list(TEXT_DIM))

        # Training panel window
        training_y = inspector_height + MENU_H + 4
        training_h = usable_h - inspector_height - 4
        with dpg.window(
            label="Training", tag="TrainingWindow",
            pos=[inspector_x, training_y], width=INSPECTOR_W, height=training_h,
            no_close=True, no_collapse=True,
        ):
            _section_header("Training", self._section_header_theme)
            with dpg.group(horizontal=True):
                dpg.add_text("*", tag="train_status_dot", color=list(TEXT_DIM))
                dpg.add_text("Idle", tag="train_status_text", color=list(TEXT_DIM))
            dpg.add_spacer(height=6)
            with dpg.group(horizontal=True):
                dpg.add_button(label=" Start ", tag="train_start_btn",
                               callback=lambda: self._train_start())
                dpg.add_button(label=" Pause ", tag="train_pause_btn",
                               callback=lambda: self._train_pause_resume())
                dpg.add_button(label=" Stop ",  tag="train_stop_btn",
                               callback=lambda: self._train_stop())
            dpg.add_spacer(height=4)
            dpg.add_button(label="Check Wiring", width=-1,
                           callback=lambda: self._train_check_wiring())
            dpg.add_spacer(height=8)
            dpg.add_text("Epoch  0 / 0", tag="train_epoch_text", color=list(TEXT))
            dpg.add_text("Best loss  —", tag="train_loss_text",  color=list(TEXT))
            dpg.add_spacer(height=6)
            with dpg.plot(label="Loss", height=150, width=-1, tag="loss_plot"):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="epoch", tag="loss_x_axis")
                with dpg.plot_axis(dpg.mvYAxis, label="loss", tag="loss_y_axis"):
                    dpg.add_line_series([], [], label="train", tag="loss_series")
                    dpg.add_line_series([], [], label="val",   tag="val_loss_series")
            dpg.add_spacer(height=6)
            dpg.add_button(label="Save Model", tag="save_model_btn", width=-1,
                           callback=lambda: self._save_model())

        # Terminal / output window (buttons moved to menu bar)
        term_y = editor_h + MENU_H + 4
        term_w = VIEWPORT_W - PALETTE_W - INSPECTOR_W - 8
        with dpg.window(
            label="Terminal", tag="TerminalWindow",
            pos=[PALETTE_W + 4, term_y], width=term_w, height=TERMINAL_H,
            no_close=True, no_collapse=True,
        ):
            tab_content_h = TERMINAL_H - 36
            with dpg.tab_bar(tag="terminal_tab_bar"):
                with dpg.tab(label="Output", tag="tab_output"):
                    with dpg.child_window(tag="terminal_scroll", autosize_x=True,
                                          height=tab_content_h, border=False):
                        dpg.add_input_text(
                            tag="terminal_text", default_value="", multiline=True,
                            readonly=True, width=-1, height=tab_content_h - 10,
                        )
                with dpg.tab(label="Code", tag="tab_code"):
                    with dpg.child_window(tag="code_scroll", autosize_x=True,
                                          height=tab_content_h, border=False):
                        with dpg.group(horizontal=True):
                            dpg.add_button(label="Save .py",
                                           callback=lambda: self._export_script())
                            dpg.add_button(label="Copy",
                                           callback=lambda: self._copy_code_panel())
                            dpg.add_text("", tag="code_status", color=list(TEXT_DIM))
                        dpg.add_spacer(height=4)
                        dpg.add_input_text(
                            tag="code_text",
                            default_value="# Run the graph to generate code preview.",
                            multiline=True, readonly=True,
                            width=term_w - 30, height=tab_content_h - 38,
                            tab_input=True,
                        )
