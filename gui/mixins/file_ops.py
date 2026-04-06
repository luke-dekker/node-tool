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
