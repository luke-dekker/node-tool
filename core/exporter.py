"""GraphExporter — turn a Graph into a runnable Python script.

Design principle: the exported script mirrors the graph 1:1. Every node becomes
an explicit, named Python statement (or small block). No hidden Sequentials, no
helper wrappers, no abstractions on top of what the graph itself does. Reading
the .py top-to-bottom should feel the same as reading the canvas left-to-right.

Output structure:
    \"\"\"<docstring with graph stats>\"\"\"
    from __future__ import annotations
    # === imports ===
    import ...

    def main():
        # === <Category> ===
        # <Node label>
        var_0 = ...
        ...

    if __name__ == "__main__":
        main()
"""
from __future__ import annotations
from collections import defaultdict
from core.graph import Graph
from core.node import BaseNode


_PREFIX_STRIP = ("pt_", "np_", "pd_", "sk_", "sp_", "viz_")

_STDLIB_HINTS = (
    "import math", "import random", "import json", "import os",
    "import sys", "import time", "import pathlib", "from pathlib",
    "from typing", "from collections", "from dataclasses",
    "from __future__",
)


def _short(type_name: str) -> str:
    """Turn a type_name into a clean variable prefix."""
    s = type_name
    for p in _PREFIX_STRIP:
        if s.startswith(p):
            s = s[len(p):]
            break
    s = s.replace("__terminal__", "terminal")
    return s or type_name


def _is_stdlib(imp: str) -> bool:
    return any(imp.startswith(h) for h in _STDLIB_HINTS)


class GraphExporter:
    """Export a Graph to a runnable Python script."""

    INDENT = "    "

    def export(self, graph: Graph) -> str:
        order = graph.topological_order()
        if not order:
            return self._empty_script()

        # Build connection lookups
        conn_map: dict[tuple[str, str], tuple[str, str]] = {}
        used_outputs: set[tuple[str, str]] = set()  # outputs with >=1 downstream consumer
        nodes_with_outgoing: set[str] = set()
        for c in graph.connections:
            conn_map[(c.to_node_id, c.to_port)] = (c.from_node_id, c.from_port)
            used_outputs.add((c.from_node_id, c.from_port))
            nodes_with_outgoing.add(c.from_node_id)

        # Variable assignment — only emit vars for outputs that something consumes,
        # so a 7-port BatchInput with one downstream image link doesn't dump 7 unused
        # tensor unpack lines. Exception: a "leaf" node (no outgoing connections at
        # all) still needs vars — its outputs ARE the result the user cares about.
        var_map: dict[tuple[str, str], str] = {}
        port_counters: dict[str, int] = defaultdict(int)

        def assign_out_vars(node: BaseNode) -> dict[str, str]:
            out_vars: dict[str, str] = {}
            prefix = _short(node.type_name)
            is_leaf = node.id not in nodes_with_outgoing
            for port_name in node.outputs:
                if port_name == "__terminal__":
                    continue
                if not is_leaf and (node.id, port_name) not in used_outputs:
                    continue
                if len(node.outputs) <= 1:
                    vname = f"{prefix}_{port_counters[prefix]}"
                else:
                    key = f"{prefix}_{port_name}"
                    vname = f"{prefix}_{port_name}_{port_counters[key]}"
                    port_counters[key] += 1
                port_counters[prefix] += 1
                out_vars[port_name] = vname
                var_map[(node.id, port_name)] = vname
            return out_vars

        # Walk topo order, collect imports + per-category code blocks
        all_imports: list[str] = []
        by_category: dict[str, list[str]] = defaultdict(list)
        category_order: list[str] = []  # preserves first-seen order
        missing_export: list[str] = []  # nodes whose export() falls back to default

        for node_id in order:
            node = graph.nodes[node_id]

            # Build in_vars: port_name -> upstream variable name (or None for unconnected)
            in_vars: dict[str, str | None] = {}
            for port_name in node.inputs:
                key = (node_id, port_name)
                if key in conn_map:
                    from_node_id, from_port = conn_map[key]
                    in_vars[port_name] = var_map.get((from_node_id, from_port))
                else:
                    in_vars[port_name] = None

            out_vars = assign_out_vars(node)

            try:
                imports, lines = node.export(in_vars, out_vars)
            except Exception as exc:
                imports, lines = [], [f"# [{node.label}] export error: {exc}"]

            # Detect default-stub fallthrough so we can list it in the TODO header
            if (len(lines) == 1
                    and lines[0].startswith("# [")
                    and "export not supported" in lines[0]):
                missing_export.append(f"{node.label} ({node.type_name})")
                # Emit a clean placeholder so downstream var refs at least exist
                stub_lines = [f"# TODO: implement export() for {node.label}"]
                for port_name, vname in out_vars.items():
                    stub_lines.append(f"{vname} = None  # output port: {port_name}")
                lines = stub_lines

            all_imports.extend(imports)

            cat = node.category or "Misc"
            if cat not in by_category:
                category_order.append(cat)
            by_category[cat].append(f"# {node.label}")
            by_category[cat].extend(lines)
            by_category[cat].append("")  # blank line between nodes

        # ── Render ──────────────────────────────────────────────────────────
        return self._render(graph, all_imports, category_order, by_category, missing_export)

    # ────────────────────────────────────────────────────────────────────────
    # Rendering
    # ────────────────────────────────────────────────────────────────────────

    def _render(
        self,
        graph: Graph,
        all_imports: list[str],
        category_order: list[str],
        by_category: dict[str, list[str]],
        missing_export: list[str],
    ) -> str:
        n_nodes = len(graph.nodes)
        n_conns = len(graph.connections)

        # Docstring header
        head: list[str] = [
            '"""Generated by Node Tool — runnable export of the current graph.',
            "",
            f"Graph: {n_nodes} node(s), {n_conns} connection(s).",
            f"Run: python {{this_file}}.py",
            "",
            "Each node in the canvas corresponds 1:1 to a code block below.",
            "There is no nn.Sequential or hidden wrapper — the script reads",
            "top-to-bottom in the same order the graph executes.",
            '"""',
            "from __future__ import annotations",
            "",
        ]

        # Imports — dedupe, sort stdlib first then third-party
        seen: set[str] = set()
        deduped: list[str] = []
        for imp in all_imports:
            imp = imp.strip()
            if imp and imp not in seen:
                seen.add(imp)
                deduped.append(imp)
        stdlib = sorted(i for i in deduped if _is_stdlib(i))
        third  = sorted(i for i in deduped if not _is_stdlib(i))

        import_block: list[str] = []
        if stdlib:
            import_block.extend(stdlib)
        if stdlib and third:
            import_block.append("")
        if third:
            import_block.extend(third)
        import_block.append("")

        # TODO summary for missing exports
        todo_block: list[str] = []
        if missing_export:
            todo_block = [
                "# --- TODO: nodes still missing export() ---",
                "# These nodes do not yet emit runnable code. Their outputs are stubbed",
                "# as None below; downstream code will need them implemented before",
                "# the script will run end-to-end.",
            ]
            for label in missing_export:
                todo_block.append(f"#   - {label}")
            todo_block.append("")

        # Body — wrapped in def main():
        body: list[str] = ["def main() -> None:"]
        for cat in category_order:
            body.append(f"{self.INDENT}# --- {cat} ---")
            for line in by_category[cat]:
                body.append(f"{self.INDENT}{line}" if line else "")
        # If main is empty for some reason
        if len(body) == 1:
            body.append(f"{self.INDENT}pass")
        body.append("")
        body.append('if __name__ == "__main__":')
        body.append(f"{self.INDENT}main()")
        body.append("")

        return "\n".join(head + import_block + todo_block + body)

    def _empty_script(self) -> str:
        return (
            '"""Generated by Node Tool — Empty graph (no nodes yet)."""\n'
            "from __future__ import annotations\n\n"
            "def main() -> None:\n"
            "    pass\n\n"
            'if __name__ == "__main__":\n'
            "    main()\n"
        )
