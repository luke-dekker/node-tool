"""FileOpsMixin — save/load graph JSON and export Python scripts."""
from __future__ import annotations
import dearpygui.dearpygui as dpg
from core.serializer import Serializer


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
