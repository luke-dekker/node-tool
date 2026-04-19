"""DearPyGui renderer for PanelSpec.

The only DPG-side code that knows about panels. Reads a spec (from any
plugin) and builds native DPG widgets; polls status/plot sections; wires
buttons to the dispatcher. Add a new plugin with a new panel — it shows up
without any edits here. If you find yourself wanting to edit this file to
support a specific panel, add a custom section kind instead.
"""
from __future__ import annotations
import time
from typing import Any, Callable

import dearpygui.dearpygui as dpg

Dispatcher = Callable[[str, dict], Any]


class PanelRuntime:
    """One per panel. Holds widget tags, polls data sources, fires actions."""

    def __init__(self, spec: dict, dispatcher: Dispatcher, panel_id: str,
                 log: Callable[[str], None] | None = None):
        self.spec = spec
        self.dispatcher = dispatcher
        self.panel_id = panel_id
        self.log = log or (lambda _: None)
        # (section_id, item_key, field_id) → dpg widget tag
        self._field_tags: dict[tuple[str, str, str], str] = {}
        # section_id → list of dpg tags that need updating for status sections
        self._status_tags: dict[str, dict[str, str]] = {}
        # section_id → last poll timestamp (ms)
        self._last_poll: dict[str, float] = {}
        # DynamicForm section id → currently-rendered items key-list (for
        # change detection on poll)
        self._dyn_keys: dict[str, list[str]] = {}
        # DynamicForm section id → container tag for rebuild
        self._dyn_containers: dict[str, str] = {}
        # Plot section id → {series_name: line_series_tag}
        self._plot_series: dict[str, dict[str, str]] = {}
        self._plot_x_axis: dict[str, str] = {}

    # ── Build ──────────────────────────────────────────────────────────

    def build(self, parent_tag: str) -> None:
        """Build all sections under parent_tag. Call once when the panel
        tab is created. poll() keeps it live afterward."""
        for section in self.spec.get("sections", []):
            self._build_section(section, parent_tag)

    def _tag(self, *parts: str) -> str:
        return f"panel_{self.panel_id}_" + "_".join(parts)

    def _build_section(self, sec: dict, parent: str) -> None:
        kind = sec.get("kind")
        if kind == "form":          self._build_form(sec, parent)
        elif kind == "dynamic_form": self._build_dynamic_form(sec, parent)
        elif kind == "status":       self._build_status(sec, parent)
        elif kind == "plot":         self._build_plot_section(sec, parent)
        elif kind == "buttons":      self._build_buttons(sec, parent)
        elif kind == "custom":       self._build_custom(sec, parent)
        else:
            dpg.add_text(f"[unknown section kind: {kind}]", parent=parent,
                         color=[255, 120, 120])

    # ── Section builders ───────────────────────────────────────────────

    def _build_form(self, sec: dict, parent: str) -> None:
        if sec.get("label"):
            dpg.add_text(sec["label"], parent=parent, color=[180, 180, 255])
        for f in sec.get("fields", []):
            self._build_field(f, parent, sec["id"], item_key="")

    def _build_dynamic_form(self, sec: dict, parent: str) -> None:
        if sec.get("label"):
            dpg.add_text(sec["label"], parent=parent, color=[180, 180, 255])
            dpg.add_separator(parent=parent)
        container = self._tag(sec["id"], "container")
        with dpg.group(parent=parent, tag=container):
            pass
        self._dyn_containers[sec["id"]] = container
        self._dyn_keys[sec["id"]] = []
        # Kick off initial population
        self._refresh_dynamic_form(sec)

    def _build_status(self, sec: dict, parent: str) -> None:
        if sec.get("label"):
            dpg.add_text(sec["label"], parent=parent, color=[180, 180, 255])
        tags: dict[str, str] = {}
        for f in sec.get("fields", []):
            tag = self._tag(sec["id"], f["id"])
            with dpg.group(horizontal=True, parent=parent):
                if f.get("label"):
                    dpg.add_text(f"{f['label']}:", color=[150, 150, 170])
                dpg.add_text("—", tag=tag)
            tags[f["id"]] = tag
        self._status_tags[sec["id"]] = tags

    def _build_plot_section(self, sec: dict, parent: str) -> None:
        # The generic PlotSection — custom kind 'loss_plot' has its own path.
        plot_tag = self._tag(sec["id"], "plot")
        x_tag    = self._tag(sec["id"], "x")
        y_tag    = self._tag(sec["id"], "y")
        with dpg.plot(parent=parent, tag=plot_tag, height=-1, width=-1,
                      label=sec.get("label", "")):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label=sec.get("x_label", ""), tag=x_tag)
            with dpg.plot_axis(dpg.mvYAxis, label=sec.get("y_label", ""), tag=y_tag):
                pass  # series added dynamically on first poll
        self._plot_x_axis[sec["id"]] = x_tag
        self._plot_series[sec["id"]] = {}

    def _build_buttons(self, sec: dict, parent: str) -> None:
        with dpg.group(horizontal=True, parent=parent):
            for action in sec.get("actions", []):
                a = action  # capture for lambda
                dpg.add_button(label=a["label"],
                               callback=lambda _s=None, _u=None, _a=a: self._fire(_a))

    def _build_custom(self, sec: dict, parent: str) -> None:
        kind = sec.get("custom_kind", "")
        if kind == "loss_plot":
            self._build_loss_plot(sec, parent)
        else:
            dpg.add_text(f"[no renderer for custom kind '{kind}']",
                         parent=parent, color=[200, 160, 80])

    def _build_loss_plot(self, sec: dict, parent: str) -> None:
        plot_tag = self._tag(sec["id"], "plot")
        x_tag    = self._tag(sec["id"], "x")
        y_tag    = self._tag(sec["id"], "y")
        if sec.get("label"):
            dpg.add_text(sec["label"], parent=parent, color=[180, 180, 255])
        with dpg.plot(parent=parent, tag=plot_tag, height=-1, width=-1):
            dpg.add_plot_legend()
            dpg.add_plot_axis(dpg.mvXAxis, label="epoch", tag=x_tag)
            with dpg.plot_axis(dpg.mvYAxis, label="loss", tag=y_tag):
                series_names = sec.get("params", {}).get("series", ["train", "val"])
                for sname in series_names:
                    stag = self._tag(sec["id"], sname)
                    dpg.add_line_series([], [], label=sname, tag=stag)
                    self._plot_series.setdefault(sec["id"], {})[sname] = stag
        self._plot_x_axis[sec["id"]] = x_tag

    # ── Field builder ─────────────────────────────────────────────────

    def _build_field(self, f: dict, parent: str, section_id: str,
                     item_key: str = "") -> None:
        tag = self._tag(section_id, item_key or "_", f["id"])
        ftype = f.get("type", "str")
        label = f.get("label") or f["id"]
        default = f.get("default")
        hint = f.get("hint", "")
        choices = f.get("choices") or []
        w = -1  # fill available

        if ftype == "choice" or choices:
            dpg.add_combo(label=label, tag=tag,
                          items=choices or [""],
                          default_value=str(default) if default is not None else "",
                          width=w, parent=parent)
        elif ftype == "int":
            dpg.add_input_int(label=label, tag=tag,
                              default_value=int(default) if default is not None else 0,
                              min_value=int(f["min"]) if f.get("min") is not None else 0,
                              step=0, width=w, parent=parent)
        elif ftype == "float":
            dpg.add_input_float(label=label, tag=tag,
                                default_value=float(default) if default is not None else 0.0,
                                step=float(f.get("step") or 0.0),
                                format="%.5f", width=w, parent=parent)
        elif ftype == "bool":
            dpg.add_checkbox(label=label, tag=tag,
                             default_value=bool(default), parent=parent)
        else:  # str
            dpg.add_input_text(label=label, tag=tag,
                               default_value=str(default) if default is not None else "",
                               hint=hint, width=w, parent=parent)

        self._field_tags[(section_id, item_key, f["id"])] = tag

    # ── Polling ────────────────────────────────────────────────────────

    def poll(self) -> None:
        """Called each frame by the app. Polls status/plot sections, refreshes
        dynamic_form sections on item-set changes."""
        now = time.time() * 1000
        for sec in self.spec.get("sections", []):
            kind = sec.get("kind")
            sid  = sec["id"]
            if kind == "status":
                if now - self._last_poll.get(sid, 0) >= sec.get("poll_ms", 500):
                    self._last_poll[sid] = now
                    try:
                        data = self.dispatcher(sec.get("source_rpc", ""), {})
                        self._apply_status(sec, data)
                    except Exception:
                        pass
            elif kind == "custom" and sec.get("custom_kind") == "loss_plot":
                p = sec.get("params", {})
                if now - self._last_poll.get(sid, 0) >= p.get("poll_ms", 500):
                    self._last_poll[sid] = now
                    try:
                        data = self.dispatcher(p.get("source_rpc", ""), {})
                        self._apply_loss_plot(sec, data)
                    except Exception:
                        pass
            elif kind == "dynamic_form":
                # Less frequent — structure change detection is cheap
                if now - self._last_poll.get(sid, 0) >= 500:
                    self._last_poll[sid] = now
                    self._refresh_dynamic_form(sec)

    def _apply_status(self, sec: dict, data: dict) -> None:
        tags = self._status_tags.get(sec["id"], {})
        for f in sec.get("fields", []):
            tag = tags.get(f["id"])
            if tag and dpg.does_item_exist(tag):
                val = data.get(f["id"], "")
                dpg.set_value(tag, str(val) if val != "" else "—")

    def _apply_loss_plot(self, sec: dict, data: dict) -> None:
        series = (data or {}).get("series", {})
        for sname, stag in self._plot_series.get(sec["id"], {}).items():
            ys = series.get(sname, [])
            xs = list(range(1, len(ys) + 1))
            if dpg.does_item_exist(stag):
                dpg.set_value(stag, [xs, ys])

    def _refresh_dynamic_form(self, sec: dict) -> None:
        try:
            resp = self.dispatcher(sec.get("source_rpc", ""), {})
        except Exception:
            return
        groups = resp.get("groups", {}) if isinstance(resp, dict) else {}
        new_keys = sorted(groups.keys())
        if new_keys == self._dyn_keys.get(sec["id"], []):
            return

        container = self._dyn_containers.get(sec["id"])
        if not container or not dpg.does_item_exist(container):
            return
        # Preserve current values for keys that still exist
        preserved = self._snapshot_dynamic_form(sec)
        dpg.delete_item(container, children_only=True)
        # Also purge stale tag entries for this section
        for key in list(self._field_tags.keys()):
            if key[0] == sec["id"]:
                self._field_tags.pop(key, None)

        if not new_keys:
            hint = sec.get("empty_hint", "")
            if hint:
                dpg.add_text(hint, parent=container, color=[120, 128, 142])
            self._dyn_keys[sec["id"]] = []
            return

        for i, key in enumerate(new_keys):
            meta = groups.get(key, {})
            label_tpl = sec.get("item_label_template", "{key}")
            try:
                title = label_tpl.format(key=key, **meta)
            except Exception:
                title = key
            with dpg.group(parent=container):
                dpg.add_text(title, color=[180, 180, 255])
                for f in sec.get("fields", []):
                    # seed default from preserved value if present
                    f2 = dict(f)
                    val = preserved.get((key, f["id"]))
                    if val is not None:
                        f2["default"] = val
                    self._build_field(f2, container, sec["id"], item_key=key)
                if i < len(new_keys) - 1:
                    dpg.add_separator()
        self._dyn_keys[sec["id"]] = new_keys

    def _snapshot_dynamic_form(self, sec: dict) -> dict[tuple[str, str], Any]:
        out: dict[tuple[str, str], Any] = {}
        for (sid, key, fid), tag in list(self._field_tags.items()):
            if sid != sec["id"] or not key:
                continue
            if dpg.does_item_exist(tag):
                try:
                    out[(key, fid)] = dpg.get_value(tag)
                except Exception:
                    pass
        return out

    # ── Actions ────────────────────────────────────────────────────────

    def _fire(self, action: dict) -> None:
        params = self._collect_params(action.get("collect", []))
        try:
            result = self.dispatcher(action["rpc"], params)
        except Exception as exc:
            self.log(f"[{self.panel_id}] {action['rpc']} failed: {exc}")
            return
        if isinstance(result, dict) and result.get("error"):
            self.log(f"[{self.panel_id}] {action['rpc']}: {result['error']}")

    def _collect_params(self, section_ids: list[str]) -> dict:
        """Gather field values from the listed sections into a flat dict.

        Static form fields go to top-level keys. Dynamic form fields are
        grouped by item key under a nested dict at top level (the section id
        is used as the key, e.g. "datasets": {group_name: {path, ...}}).
        """
        sections_by_id = {s["id"]: s for s in self.spec.get("sections", [])}
        params: dict[str, Any] = {}
        for sid in section_ids:
            sec = sections_by_id.get(sid)
            if not sec:
                continue
            if sec.get("kind") == "dynamic_form":
                nested: dict[str, dict] = {}
                for (s_id, key, fid), tag in self._field_tags.items():
                    if s_id != sid or not key:
                        continue
                    val = self._read_field(sec, fid, tag)
                    nested.setdefault(key, {})[fid] = val
                params[sid] = nested
            else:
                for f in sec.get("fields", []):
                    tag = self._field_tags.get((sid, "", f["id"]))
                    if tag is None:
                        continue
                    params[f["id"]] = self._read_field(sec, f["id"], tag)
        return params

    def _read_field(self, sec: dict, fid: str, tag: str) -> Any:
        if not dpg.does_item_exist(tag):
            return None
        try:
            v = dpg.get_value(tag)
        except Exception:
            return None
        # Coerce by declared type
        spec_field = next(
            (f for f in sec.get("fields", []) if f["id"] == fid),
            None,
        )
        if spec_field is None:
            return v
        t = spec_field.get("type", "str")
        try:
            if t == "int":   return int(v)
            if t == "float": return float(v)
            if t == "bool":  return bool(v)
        except Exception:
            return v
        return v


def render_panel(spec: dict, parent_tag: str, dispatcher: Dispatcher,
                 panel_id: str, log: Callable[[str], None] | None = None
                 ) -> PanelRuntime:
    """Entry point: build the panel and return its runtime for polling."""
    runtime = PanelRuntime(spec, dispatcher, panel_id, log=log)
    runtime.build(parent_tag)
    return runtime
