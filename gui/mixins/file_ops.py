"""FileOpsMixin — save/load graph JSON and export Python scripts."""
from __future__ import annotations
import dearpygui.dearpygui as dpg
from core.serializer import Serializer


def _prompt_port_names(ext_in: list, ext_out: list) -> tuple[list, list] | None:
    """Show a Tk dialog letting the user rename each external port before save.

    Builds a small Toplevel with one Entry widget per port (initialized with
    the auto-detected name) plus OK/Cancel buttons. Returns (ext_in, ext_out)
    with the renamed `name` fields, or None if the user cancelled.

    Defined as a module-level helper rather than a method so the closure
    captures are explicit and the dialog can be unit-tested in isolation.
    """
    import tkinter as tk
    from dataclasses import replace

    root = tk.Tk()
    root.title("Pack as Subgraph — name external ports")
    root.geometry("420x500")
    root.attributes("-topmost", True)

    in_entries: list[tk.Entry] = []
    out_entries: list[tk.Entry] = []
    cancelled = {"value": True}  # mutable closure cell

    pad = {"padx": 8, "pady": 2}
    tk.Label(root, text="External input ports:", font=("Segoe UI", 10, "bold")).pack(
        anchor="w", padx=8, pady=(8, 2)
    )
    if not ext_in:
        tk.Label(root, text="  (none)", fg="gray").pack(anchor="w", **pad)
    for ep in ext_in:
        frame = tk.Frame(root); frame.pack(fill="x", **pad)
        tk.Label(frame, text=f"  {ep.type:<10s}", font=("Consolas", 9), width=12, anchor="w").pack(side="left")
        e = tk.Entry(frame, font=("Segoe UI", 10))
        e.insert(0, ep.name)
        e.pack(side="left", fill="x", expand=True)
        in_entries.append(e)

    tk.Label(root, text="External output ports:", font=("Segoe UI", 10, "bold")).pack(
        anchor="w", padx=8, pady=(12, 2)
    )
    if not ext_out:
        tk.Label(root, text="  (none)", fg="gray").pack(anchor="w", **pad)
    for ep in ext_out:
        frame = tk.Frame(root); frame.pack(fill="x", **pad)
        tk.Label(frame, text=f"  {ep.type:<10s}", font=("Consolas", 9), width=12, anchor="w").pack(side="left")
        e = tk.Entry(frame, font=("Segoe UI", 10))
        e.insert(0, ep.name)
        e.pack(side="left", fill="x", expand=True)
        out_entries.append(e)

    def on_ok():
        cancelled["value"] = False
        root.quit()

    def on_cancel():
        root.quit()

    btn_frame = tk.Frame(root); btn_frame.pack(side="bottom", fill="x", pady=10)
    tk.Button(btn_frame, text="Cancel", width=10, command=on_cancel).pack(side="right", padx=8)
    tk.Button(btn_frame, text="Save",   width=10, command=on_ok).pack(side="right")

    root.protocol("WM_DELETE_WINDOW", on_cancel)
    root.mainloop()

    if cancelled["value"]:
        try:
            root.destroy()
        except Exception:
            pass
        return None

    # Pull the renamed values out before destroying the window
    new_in  = [replace(ep, name=in_entries[i].get().strip() or ep.name)
               for i, ep in enumerate(ext_in)]
    new_out = [replace(ep, name=out_entries[i].get().strip() or ep.name)
               for i, ep in enumerate(ext_out)]
    try:
        root.destroy()
    except Exception:
        pass
    return new_in, new_out


class FileOpsMixin:
    """Graph file operations: save, load, code export."""

    def _save_graph(self) -> None:
        """Ctrl+S — save graph to JSON."""
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
        """Ctrl+O — load graph from JSON."""
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
            self._clear_editor()
            for node in graph.nodes.values():
                self.graph.add_node(node)
            for conn in graph.connections:
                self.graph.add_connection(
                    conn.from_node_id, conn.from_port,
                    conn.to_node_id, conn.to_port
                )
            for node_id, node in graph.nodes.items():
                pos = positions.get(node_id, [100, 100])
                self.add_node_to_editor(node, tuple(pos))
            for conn in graph.connections:
                from_attr = f"attr_out_{conn.from_node_id}_{conn.from_port}"
                to_attr = f"attr_in_{conn.to_node_id}_{conn.to_port}"
                if dpg.does_item_exist(from_attr) and dpg.does_item_exist(to_attr):
                    lt = dpg.add_node_link(from_attr, to_attr, parent="NodeEditor")
                    self.dpg_link_to_conn[lt] = (
                        conn.from_node_id, conn.from_port,
                        conn.to_node_id, conn.to_port,
                    )
            self._log(f"Graph loaded from {path}")
        except Exception as e:
            import traceback
            self._log(f"Load failed: {traceback.format_exc()}")

    def _load_template(self, builder, label: str = "") -> None:
        """Clear the editor and populate it from a template builder function.

        `builder(graph)` should add nodes/connections to `graph` (an empty Graph)
        and return a dict {node_id: (x, y)} for layout. Visual nodes and links
        are then created in DPG to mirror the just-built graph.
        """
        try:
            self._clear_editor()
            positions = builder(self.graph) or {}

            # Create DPG nodes
            for node_id, node in self.graph.nodes.items():
                pos = positions.get(node_id, (100, 100))
                self.add_node_to_editor(node, tuple(pos))

            # Create visual links for every connection
            for conn in self.graph.connections:
                from_attr = f"attr_out_{conn.from_node_id}_{conn.from_port}"
                to_attr = f"attr_in_{conn.to_node_id}_{conn.to_port}"
                if dpg.does_item_exist(from_attr) and dpg.does_item_exist(to_attr):
                    lt = dpg.add_node_link(from_attr, to_attr, parent="NodeEditor")
                    self.dpg_link_to_conn[lt] = (
                        conn.from_node_id, conn.from_port,
                        conn.to_node_id, conn.to_port,
                    )
            self._save_path = None  # template is unsaved by definition
            self._log(f"Template loaded: {label or 'unnamed'}")
            if hasattr(self, "_rebuild_dataset_panel"):
                self._rebuild_dataset_panel()
        except Exception as e:
            import traceback
            self._log(f"Template load failed: {traceback.format_exc()}")

    def _refresh_code_panel(self) -> None:
        """Regenerate the export script and show it in the Code tab."""
        try:
            from core.exporter import GraphExporter
            script = GraphExporter().export(self.graph)
            dpg.set_value("code_text", script)
        except Exception as e:
            try:
                dpg.set_value("code_text", f"# Code generation error: {e}")
            except Exception:
                pass

    def _copy_code_panel(self) -> None:
        """Copy the current code panel content to the clipboard."""
        try:
            text = dpg.get_value("code_text") or ""
            dpg.set_clipboard_text(text)
            if dpg.does_item_exist("code_status"):
                dpg.set_value("code_status", "  Copied to clipboard")
        except Exception as e:
            if dpg.does_item_exist("code_status"):
                dpg.set_value("code_status", f"  Copy failed: {e}")

    def _pack_as_subgraph(self) -> None:
        """Pack the currently selected nodes as a reusable subgraph file.

        Walks the selection's boundary connections to determine which inner
        ports are exposed externally, then writes a .subgraph.json to the
        project's subgraphs/ directory. After saving, the subgraph appears
        in the palette under the Subgraphs category on next app start (or
        immediately if you call _reload_subgraphs from the menu).
        """
        import tkinter as tk
        from tkinter import simpledialog
        try:
            # Resolve selected node IDs (DPG returns integer ids; map to node uuids)
            selected_dpg = []
            try:
                selected_dpg = list(dpg.get_selected_nodes("NodeEditor"))
            except Exception:
                pass
            selected_node_ids: set[str] = set()
            for tag in selected_dpg:
                node_id = self.dpg_node_to_node_id.get(tag)
                if node_id:
                    selected_node_ids.add(node_id)
            if not selected_node_ids:
                self._log("[Pack] No nodes selected — select nodes in the editor first.")
                return

            # Ask for a name (also defines the file stem)
            root = tk.Tk(); root.withdraw()
            name = simpledialog.askstring(
                "Pack as Subgraph",
                f"Subgraph name ({len(selected_node_ids)} nodes selected):",
                initialvalue="MyBlock",
                parent=root,
            )
            root.destroy()
            if not name:
                return

            # Detect boundary ports
            from core.subgraph import SubgraphFile, SUBGRAPHS_DIR, detect_boundary_ports
            ext_in, ext_out = detect_boundary_ports(self.graph, selected_node_ids)

            # Let the user rename the auto-detected port names before saving
            renamed = _prompt_port_names(ext_in, ext_out)
            if renamed is None:
                self._log("[Pack] Cancelled.")
                return
            ext_in, ext_out = renamed

            # Serialize the selected nodes + their internal connections
            nodes_json = []
            for node_id in selected_node_ids:
                node = self.graph.nodes.get(node_id)
                if node is None:
                    continue
                pos = [100, 100]
                try:
                    tag = self.node_id_to_dpg.get(node_id)
                    if tag is not None:
                        pos = list(dpg.get_item_pos(tag))
                except Exception:
                    pass
                nodes_json.append({
                    "id": node.id,
                    "type_name": node.type_name,
                    "pos": pos,
                    "inputs": {
                        k: (p.default_value if isinstance(p.default_value, (int, float, bool, str, type(None))) else None)
                        for k, p in node.inputs.items()
                    },
                })
            conns_json = []
            for c in self.graph.connections:
                if c.from_node_id in selected_node_ids and c.to_node_id in selected_node_ids:
                    conns_json.append({
                        "from_node": c.from_node_id, "from_port": c.from_port,
                        "to_node":   c.to_node_id,   "to_port":   c.to_port,
                    })

            sf = SubgraphFile(
                name=name,
                description=f"Packed from {len(selected_node_ids)} nodes",
                external_inputs=ext_in,
                external_outputs=ext_out,
                nodes=nodes_json,
                connections=conns_json,
            )
            # File stem: lowercase, underscores
            stem = "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")
            out_path = SUBGRAPHS_DIR / f"{stem}.subgraph.json"
            sf.save(out_path)
            self._log(f"[Pack] Saved subgraph to {out_path}")
            self._log(f"[Pack] External inputs:  {[p.name for p in ext_in]}")
            self._log(f"[Pack] External outputs: {[p.name for p in ext_out]}")

            # Force the reloader to scan now (bypass its 1s throttle) so the new
            # subgraph node appears in the palette immediately
            try:
                self._subgraph_reloader._last_check = 0.0
                self._poll_subgraph_reload()
            except Exception:
                self._reload_subgraphs()  # fall back to legacy registry refresh
        except Exception as e:
            import traceback
            self._log(f"[Pack] failed: {traceback.format_exc()}")

    def _expand_subgraph_inline(self) -> None:
        """Expand the selected SubgraphNode(s) inline: replace each with its inner nodes.

        The inverse of _pack_as_subgraph. For each selected subgraph node:
        - Inner nodes are added to the parent graph with fresh UUIDs.
        - Internal connections are re-created.
        - External boundary connections are rewired to the corresponding inner ports.
        - The original subgraph node is removed.
        DPG calls that require an active context are wrapped in try/except so
        the graph-mutation logic can run headlessly in tests.
        """
        import uuid as _uuid_mod

        # Gather selected DPG node tags
        selected_dpg: list = []
        try:
            selected_dpg = list(dpg.get_selected_nodes("NodeEditor"))
        except Exception:
            pass

        # Resolve to graph node objects, filter to subgraph nodes
        subgraph_items: list[tuple[str, object]] = []  # (node_id, node)
        for tag in selected_dpg:
            node_id = self.dpg_node_to_node_id.get(tag)
            if node_id is None:
                continue
            node = self.graph.nodes.get(node_id)
            if node is not None and getattr(node, "type_name", "").startswith("subgraph_"):
                subgraph_items.append((node_id, node))

        if not subgraph_items:
            self._log("[Expand] No subgraph node selected — select a SubgraphNode first.")
            return

        expanded = 0
        for subgraph_id, sg_node in subgraph_items:
            try:
                self._expand_single_subgraph(subgraph_id, sg_node, _uuid_mod)
                expanded += 1
            except Exception:
                import traceback
                self._log(f"[Expand] Failed expanding {subgraph_id}: {traceback.format_exc()}")

        self._log(f"[Expand] Expanded {expanded} subgraph(s) inline.")

    def _expand_single_subgraph(self, subgraph_id: str, sg_node, _uuid_mod) -> None:
        """Core expansion logic for one subgraph node (separated for testability)."""
        # Get the SubgraphFile from the class attribute
        sf = type(sg_node)._subgraph_file

        # Get canvas position of the subgraph node (best-effort)
        base_pos = [100.0, 100.0]
        try:
            dpg_tag = self.node_id_to_dpg.get(subgraph_id)
            if dpg_tag is not None:
                base_pos = list(dpg.get_item_pos(dpg_tag))
        except Exception:
            pass

        # Build the inner graph
        inner_graph = sf.build_inner_graph()

        # Build a mapping old inner node id -> new uuid
        id_map: dict[str, str] = {}
        for old_id in inner_graph.nodes:
            id_map[old_id] = str(_uuid_mod.uuid4())

        # Add inner nodes to parent graph with fresh IDs
        for old_id, inner_node in inner_graph.nodes.items():
            new_id = id_map[old_id]
            inner_node.id = new_id
            self.graph.add_node(inner_node)

            # Compute position: saved pos in subgraph + subgraph's canvas pos
            saved_pos = sf.positions.get(old_id, None)
            if saved_pos is not None:
                pos = (float(base_pos[0]) + float(saved_pos[0]),
                       float(base_pos[1]) + float(saved_pos[1]))
            else:
                # Fallback grid: spread nodes 200px apart
                idx = list(inner_graph.nodes.keys()).index(old_id)
                pos = (float(base_pos[0]) + idx * 200, float(base_pos[1]))

            try:
                self.add_node_to_editor(inner_node, pos)
            except Exception:
                pass  # headless / no DPG context

        # Add internal connections
        for c in inner_graph.connections:
            new_from = id_map.get(c.from_node_id, c.from_node_id)
            new_to = id_map.get(c.to_node_id, c.to_node_id)
            self.graph.add_connection(new_from, c.from_port, new_to, c.to_port)
            try:
                from_attr = f"attr_out_{new_from}_{c.from_port}"
                to_attr = f"attr_in_{new_to}_{c.to_port}"
                lt = dpg.add_node_link(from_attr, to_attr, parent="NodeEditor")
                self.dpg_link_to_conn[lt] = (new_from, c.from_port, new_to, c.to_port)
            except Exception:
                pass

        # Build lookup dicts for external ports
        ext_in_by_name = {ep.name: ep for ep in sf.external_inputs}
        ext_out_by_name = {ep.name: ep for ep in sf.external_outputs}

        # Rewire parent-graph connections that touch the subgraph node.
        # Snapshot connections to avoid mutation-during-iteration issues.
        parent_conns = list(self.graph.connections)

        for c in parent_conns:
            if c.to_node_id == subgraph_id:
                # Incoming connection into the subgraph — rewire to inner input port
                ep = ext_in_by_name.get(c.to_port)
                if ep is None:
                    continue
                new_inner_id = id_map.get(ep.inner_node)
                if new_inner_id is None:
                    continue
                self._remove_connection_and_link(c.from_node_id, c.from_port,
                                                 c.to_node_id, c.to_port)
                self.graph.add_connection(c.from_node_id, c.from_port,
                                          new_inner_id, ep.inner_port)
                try:
                    from_attr = f"attr_out_{c.from_node_id}_{c.from_port}"
                    to_attr = f"attr_in_{new_inner_id}_{ep.inner_port}"
                    lt = dpg.add_node_link(from_attr, to_attr, parent="NodeEditor")
                    self.dpg_link_to_conn[lt] = (c.from_node_id, c.from_port,
                                                  new_inner_id, ep.inner_port)
                except Exception:
                    pass

            elif c.from_node_id == subgraph_id:
                # Outgoing connection from the subgraph — rewire from inner output port
                ep = ext_out_by_name.get(c.from_port)
                if ep is None:
                    continue
                new_inner_id = id_map.get(ep.inner_node)
                if new_inner_id is None:
                    continue
                self._remove_connection_and_link(c.from_node_id, c.from_port,
                                                 c.to_node_id, c.to_port)
                self.graph.add_connection(new_inner_id, ep.inner_port,
                                          c.to_node_id, c.to_port)
                try:
                    from_attr = f"attr_out_{new_inner_id}_{ep.inner_port}"
                    to_attr = f"attr_in_{c.to_node_id}_{c.to_port}"
                    lt = dpg.add_node_link(from_attr, to_attr, parent="NodeEditor")
                    self.dpg_link_to_conn[lt] = (new_inner_id, ep.inner_port,
                                                  c.to_node_id, c.to_port)
                except Exception:
                    pass

        # Delete the original subgraph node from tracking dicts and graph
        subgraph_dpg_tag = self.node_id_to_dpg.get(subgraph_id)

        # Remove any remaining DPG links touching the subgraph node
        for lt in [lt for lt, conn in list(self.dpg_link_to_conn.items())
                   if conn[0] == subgraph_id or conn[2] == subgraph_id]:
            self.dpg_link_to_conn.pop(lt, None)
            try:
                if dpg.does_item_exist(lt):
                    dpg.delete_item(lt)
            except Exception:
                pass

        # Clean up tracking dicts
        for k in [k for k in list(self.input_widgets) if k[0] == subgraph_id]:
            self.input_widgets.pop(k, None)
        for k in [k for k in list(self.output_displays) if k[0] == subgraph_id]:
            self.output_displays.pop(k, None)
        self.node_id_to_dpg.pop(subgraph_id, None)
        if subgraph_dpg_tag is not None:
            self.dpg_node_to_node_id.pop(subgraph_dpg_tag, None)
        self._node_base_pos.pop(subgraph_id, None)

        self.graph.remove_node(subgraph_id)

        if subgraph_dpg_tag is not None:
            try:
                if dpg.does_item_exist(subgraph_dpg_tag):
                    dpg.delete_item(subgraph_dpg_tag)
            except Exception:
                pass

    def _remove_connection_and_link(self, from_node_id, from_port, to_node_id, to_port) -> None:
        """Remove a graph connection and its corresponding DPG link (if any)."""
        self.graph.remove_connection(from_node_id, from_port, to_node_id, to_port)
        for lt, conn in list(self.dpg_link_to_conn.items()):
            if conn == (from_node_id, from_port, to_node_id, to_port):
                self.dpg_link_to_conn.pop(lt, None)
                try:
                    if dpg.does_item_exist(lt):
                        dpg.delete_item(lt)
                except Exception:
                    pass
                break

    def _reload_subgraphs(self) -> None:
        """Re-scan subgraphs/ and re-register dynamic SubgraphNode classes.

        Note: this updates NODE_REGISTRY but doesn't refresh the palette UI in
        place (palette is built once at startup). For v1 the user sees the new
        subgraph after restarting the app or via the right-click context menu.
        """
        try:
            from nodes import subgraphs as sg_mod, NODE_REGISTRY
            new_types = sg_mod.reload_subgraphs()
            # Re-register
            for t in new_types:
                cls = sg_mod._GENERATED.get(t)
                if cls is not None:
                    NODE_REGISTRY[t] = cls
            self._log(f"[Pack] Subgraphs registered: {new_types}")
        except Exception as e:
            self._log(f"[Pack] reload failed: {e}")

    def _export_class(self) -> None:
        """Export graph as a reusable nn.Module class to a .py file."""
        import tkinter as tk
        from tkinter import filedialog, simpledialog
        try:
            root = tk.Tk()
            root.withdraw()
            class_name = simpledialog.askstring(
                "Export As Class",
                "Class name:",
                initialvalue="GraphModel",
                parent=root,
            )
            if not class_name:
                root.destroy()
                return
            path = filedialog.asksaveasfilename(
                title="Export Class",
                defaultextension=".py",
                initialfile=f"{class_name.lower()}.py",
                filetypes=[("Python files", "*.py"), ("All files", "*.*")],
            )
            root.destroy()
            if not path:
                return
            from core.exporter import GraphExporter
            script = GraphExporter().export(self.graph, mode="class", class_name=class_name)
            with open(path, "w", encoding="utf-8") as f:
                f.write(script)
            self._log(f"Class exported to {path}")
            self._log(f"Use it: from {class_name.lower()} import {class_name}; m = {class_name}()")
        except Exception as e:
            import traceback
            self._log(f"Class export failed: {traceback.format_exc()}")

    def _export_script(self) -> None:
        """Ctrl+E — export graph to a runnable Python script."""
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
            from core.exporter import GraphExporter
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
