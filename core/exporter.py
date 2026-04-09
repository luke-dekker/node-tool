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
import re
from collections import defaultdict
from core.graph import Graph
from core.node import BaseNode

# Detects "var = nn.Something(" or "var = M.something(" — the start of a module
# constructor block. Used by class-mode export to split init lines from forward
# lines without forcing every node to expose a separate export_module() method.
_MODULE_CTOR_RE = re.compile(r"^\s*(\w+)\s*=\s*(nn|M|torchvision\.models)\.\w+\(")
_CLASS_DEF_RE  = re.compile(r"^\s*class\s+(\w+)\s*\(.*\)\s*:\s*$")


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

    def export(self, graph: Graph, mode: str = "script", class_name: str = "GraphModel") -> str:
        """Export the graph as Python.

        mode='script' (default) emits a procedural `def main(): ...` script — same
        behavior as before. mode='class' emits an `nn.Module` subclass with the
        layer constructors in __init__ and the forward chain in forward(). The
        class form is the right artifact when you want to drop the model into
        another graph as a single reusable building block.
        """
        if mode == "class":
            return self._export_class(graph, class_name)
        return self._export_script(graph)

    def _walk_graph(self, graph: Graph) -> dict:
        """Single source of truth for walking a graph and producing per-node
        export output. Both script and class modes consume this; differences
        between the two live in their respective renderers.

        Returns a dict with:
            order:               list of node_ids in topological order
            blocks:              list of (node, out_vars, lines) for each
                                 successfully exported node
            imports:             flat list (NOT deduped) of imports collected
                                 from all nodes
            var_map:             {(node_id, port_name): var_name} for every
                                 emitted output variable
            conn_map:            {(to_node_id, to_port): (from_node_id, from_port)}
            nodes_with_outgoing: set of node ids that have at least one
                                 outgoing connection (used to identify leaves)
            missing_export:      list of "label (type_name)" for nodes whose
                                 export() fell back to the default stub
        """
        order = graph.topological_order()

        conn_map: dict[tuple[str, str], tuple[str, str]] = {}
        used_outputs: set[tuple[str, str]] = set()
        nodes_with_outgoing: set[str] = set()
        for c in graph.connections:
            conn_map[(c.to_node_id, c.to_port)] = (c.from_node_id, c.from_port)
            used_outputs.add((c.from_node_id, c.from_port))
            nodes_with_outgoing.add(c.from_node_id)

        # Variable assignment — only emit vars for outputs that something
        # consumes, with a leaf-node exception so the final tensor in a chain
        # still gets a proper name.
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

        all_imports: list[str] = []
        blocks: list[tuple[BaseNode, dict[str, str], list[str]]] = []
        missing_export: list[str] = []

        for node_id in order:
            node = graph.nodes[node_id]

            in_vars: dict[str, str | None] = {}
            for port_name in node.inputs:
                key = (node_id, port_name)
                if key in conn_map:
                    in_vars[port_name] = var_map.get(conn_map[key])
                else:
                    in_vars[port_name] = None

            out_vars = assign_out_vars(node)

            try:
                imports, lines = node.export(in_vars, out_vars)
            except Exception as exc:
                imports, lines = [], [f"# [{node.label}] export error: {exc}"]

            # Detect default-stub fallthrough
            if (len(lines) == 1
                    and lines[0].startswith("# [")
                    and "export not supported" in lines[0]):
                missing_export.append(f"{node.label} ({node.type_name})")
                stub_lines = [f"# TODO: implement export() for {node.label}"]
                for port_name, vname in out_vars.items():
                    stub_lines.append(f"{vname} = None  # output port: {port_name}")
                lines = stub_lines

            all_imports.extend(imports)
            blocks.append((node, out_vars, lines))

        return {
            "order":               order,
            "blocks":              blocks,
            "imports":             all_imports,
            "var_map":             var_map,
            "conn_map":            conn_map,
            "nodes_with_outgoing": nodes_with_outgoing,
            "missing_export":      missing_export,
        }

    def _export_script(self, graph: Graph) -> str:
        if not graph.nodes:
            return self._empty_script()

        walk = self._walk_graph(graph)

        # Group blocks by node category, preserving first-seen order
        by_category: dict[str, list[str]] = defaultdict(list)
        category_order: list[str] = []
        for node, _out_vars, lines in walk["blocks"]:
            cat = node.category or "Misc"
            if cat not in by_category:
                category_order.append(cat)
            by_category[cat].append(f"# {node.label}")
            by_category[cat].extend(lines)
            by_category[cat].append("")

        return self._render(
            graph, walk["imports"], category_order, by_category, walk["missing_export"]
        )

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

    # ────────────────────────────────────────────────────────────────────────
    # Class export
    # ────────────────────────────────────────────────────────────────────────

    def _export_class(self, graph: Graph, class_name: str) -> str:
        """Emit graph as an nn.Module subclass.

        Uses the shared _walk_graph() helper for the topological walk and
        per-node export call. The class-specific work is:

          - BatchInputNode → forward(self, x) arg (its lines are dropped, its
            output vars are renamed to 'x' in downstream lines)
          - Each node's lines split into __init__ (constructors) vs forward()
            via _split_init_forward() heuristic
          - Layer vars (matched by `var = nn.X(...)`) get a `self.` prefix
          - Leaf-node tensor outputs become the forward() return value
        """
        if not graph.nodes:
            return self._empty_class(class_name)

        walk = self._walk_graph(graph)

        all_imports: list[str] = []
        init_lines: list[str] = []
        forward_lines: list[str] = []
        layer_vars: set[str] = set()       # vars to prefix with `self.`
        rename_to_x: set[str] = set()      # batch_input output vars → forward arg
        leaf_outputs: list[str] = []        # final tensor vars (return values)

        nodes_with_outgoing = walk["nodes_with_outgoing"]

        for node, out_vars, lines in walk["blocks"]:
            # BatchInputNode is the model's forward arg, not a code block.
            # Skip its lines but record its output var names so downstream
            # references get renamed to `x`.
            if node.type_name == "batch_input":
                for vname in out_vars.values():
                    rename_to_x.add(vname)
                continue

            # Skip stub-fallthrough lines — _walk_graph already replaced them
            # with a TODO comment + None assignments. In class mode we want a
            # cleaner single TODO comment in forward and no init pollution.
            if lines and lines[0].startswith("# TODO: implement export()"):
                forward_lines.append(lines[0])
                continue

            init_block, fwd_block, found_layer_vars = self._split_init_forward(lines)
            init_lines.extend(init_block)
            forward_lines.extend(fwd_block)
            layer_vars.update(found_layer_vars)

            # Track leaf-node tensor outputs as candidates for the return statement
            if node.id not in nodes_with_outgoing:
                for port_name in ("tensor_out", "tensor", "output", "model", "result"):
                    if port_name in out_vars:
                        leaf_outputs.append(out_vars[port_name])
                        break

        all_imports = walk["imports"]

        # Apply self. prefix to layer var refs in forward, init self.assignments
        init_lines = self._prefix_self(init_lines, layer_vars, in_init=True)
        forward_lines = self._prefix_self(forward_lines, layer_vars, in_init=False)
        # Rename batch input vars to `x`
        for old in rename_to_x:
            forward_lines = [self._rename_var(l, old, "x") for l in forward_lines]
            init_lines    = [self._rename_var(l, old, "x") for l in init_lines]

        return self._render_class(graph, class_name, all_imports,
                                  init_lines, forward_lines, leaf_outputs)

    def _split_init_forward(self, lines: list[str]) -> tuple[list[str], list[str], set[str]]:
        """Classify each line as init (constructor) or forward.

        Detects single- and multi-line constructors of the form
        `var = nn.X(...)` / `var = M.x(...)`. Returns (init, forward, var_names).
        """
        init: list[str] = []
        forward: list[str] = []
        layer_vars: set[str] = set()

        i = 0
        while i < len(lines):
            line = lines[i]
            m = _MODULE_CTOR_RE.match(line)
            if m:
                var = m.group(1)
                # Walk forward until paren depth balances (multi-line ctor)
                depth = line.count("(") - line.count(")")
                block = [line]
                j = i + 1
                while depth > 0 and j < len(lines):
                    block.append(lines[j])
                    depth += lines[j].count("(") - lines[j].count(")")
                    j += 1
                init.extend(block)
                layer_vars.add(var)
                i = j
                continue
            # Class definition (Autoencoder/VAE inline class)
            cls_m = _CLASS_DEF_RE.match(line)
            if cls_m:
                # Capture the class def + all subsequent indented lines into init
                init.append(line)
                j = i + 1
                while j < len(lines) and (lines[j].startswith("    ") or lines[j].startswith("\t") or not lines[j].strip()):
                    init.append(lines[j])
                    j += 1
                i = j
                continue
            forward.append(line)
            i += 1
        return init, forward, layer_vars

    def _prefix_self(self, lines: list[str], layer_vars: set[str], in_init: bool) -> list[str]:
        """Prefix layer var refs with `self.`. In init, only the LHS of an
        assignment becomes `self.var`; in forward, every reference does."""
        out: list[str] = []
        for line in lines:
            if in_init:
                # Match leading var = ... and rewrite the LHS only
                m = re.match(r"^(\s*)(\w+)(\s*=)", line)
                if m and m.group(2) in layer_vars:
                    line = f"{m.group(1)}self.{m.group(2)}{m.group(3)}{line[m.end():]}"
            else:
                for var in layer_vars:
                    line = self._rename_var(line, var, f"self.{var}")
            out.append(line)
        return out

    @staticmethod
    def _rename_var(line: str, old: str, new: str) -> str:
        """Replace whole-word occurrences of `old` with `new`."""
        return re.sub(rf"\b{re.escape(old)}\b", new, line)

    def _render_class(
        self,
        graph: Graph,
        class_name: str,
        all_imports: list[str],
        init_lines: list[str],
        forward_lines: list[str],
        leaf_outputs: list[str],
    ) -> str:
        n_nodes = len(graph.nodes)
        n_conns = len(graph.connections)

        head = [
            f'"""Generated by Node Tool — {class_name} module from graph.',
            "",
            f"Graph: {n_nodes} node(s), {n_conns} connection(s).",
            "",
            "Drop this file next to your code and:",
            f"    from {{this_file}} import {class_name}",
            f"    model = {class_name}()",
            '"""',
            "from __future__ import annotations",
            "",
        ]

        seen: set[str] = set()
        deduped: list[str] = []
        # Always include nn for the class subclass declaration
        for imp in ["import torch", "import torch.nn as nn", *all_imports]:
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

        # __init__
        init_body = [f"{self.INDENT * 2}{l}" if l else "" for l in init_lines]
        if not any(l.strip() for l in init_body):
            init_body = [f"{self.INDENT * 2}pass"]

        # forward
        fwd_body = [f"{self.INDENT * 2}{l}" if l else "" for l in forward_lines]

        # Return statement
        if not leaf_outputs:
            ret = f"{self.INDENT * 2}return None  # no leaf-tensor outputs found"
        elif len(leaf_outputs) == 1:
            ret = f"{self.INDENT * 2}return {leaf_outputs[0]}"
        else:
            ret = f"{self.INDENT * 2}return ({', '.join(leaf_outputs)})"

        body = [
            f"class {class_name}(nn.Module):",
            f"{self.INDENT}def __init__(self) -> None:",
            f"{self.INDENT * 2}super().__init__()",
            *init_body,
            "",
            f"{self.INDENT}def forward(self, x):",
            *fwd_body,
            ret,
            "",
        ]

        return "\n".join(head + import_block + body)

    def _empty_class(self, class_name: str) -> str:
        return (
            f'"""Generated by Node Tool — {class_name} (empty graph)."""\n'
            "from __future__ import annotations\n"
            "import torch\n"
            "import torch.nn as nn\n\n"
            f"class {class_name}(nn.Module):\n"
            "    def __init__(self) -> None:\n"
            "        super().__init__()\n\n"
            "    def forward(self, x):\n"
            "        return x\n"
        )
