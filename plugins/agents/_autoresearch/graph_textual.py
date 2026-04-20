"""Textual graph format consumed by MutatorNode's LLM prompt.

Example output for a 3-node line graph:

    Node a3f2 (InputMarkerNode) "Data In (A)"
      inputs: group="task_1", modality="x"
      outputs: tensor -> [LinearNode b1c4 .x]
    Node b1c4 (LinearNode) "Linear"
      inputs: x=<InputMarkerNode a3f2 .tensor>, out_features=64
      outputs: y -> [TrainMarkerNode d8e1 .tensor_in]
    Node d8e1 (TrainMarkerNode) "Data Out (B)"
      inputs: tensor_in=<LinearNode b1c4 .y>, group="task_1"
      outputs: config -> (none)

Compact enough to fit a 4–8k-token context for small graphs. The LLM is
asked to propose ONE typed mutation op against this representation; the
orchestrator executes the op via `apply_mutation`.
"""
from __future__ import annotations
from typing import Any


def serialize_graph_textual(
    graph,
    *,
    region: list[str] | None = None,
    max_chars: int = 4000,
) -> str:
    """Render the graph (or a subset) as a plain-text block.

    `region` (optional): node ids to include. None = every node in the graph.
    Typical value: `graph.subgraph_between(a_id, b_id)` for the autoresearch
    mutable region.

    `max_chars` caps the output so large graphs don't overflow the LLM's
    context window. Truncation adds a `... (N nodes omitted)` trailer.

    Nodes are ordered topologically. Connection refs use short 6-char ids.
    Scalar defaults (str, int, float, bool, None) are inlined; complex
    values collapse to `<non-scalar>` — the LLM doesn't need tensor shapes
    to propose a swap_node_class mutation.
    """
    included = set(region) if region is not None else set(graph.nodes.keys())
    order = [nid for nid in graph.topological_order() if nid in included]

    # Build incoming / outgoing connection maps keyed by (node_id, port).
    incoming: dict[tuple[str, str], tuple[str, str]] = {}
    outgoing: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for c in graph.connections:
        if c.to_node_id in included and c.from_node_id in included:
            incoming[(c.to_node_id, c.to_port)] = (c.from_node_id, c.from_port)
            outgoing.setdefault((c.from_node_id, c.from_port), []).append(
                (c.to_node_id, c.to_port)
            )

    lines: list[str] = []
    for nid in order:
        node = graph.nodes[nid]
        header = _node_header(node)
        in_line = _inputs_line(node, incoming, graph)
        out_line = _outputs_line(node, outgoing, graph)
        lines.append(header)
        lines.append(in_line)
        lines.append(out_line)

    out = "\n".join(lines)
    if len(out) <= max_chars:
        return out
    # Truncate at a node boundary, suffix a summary of how many we dropped.
    truncated_lines: list[str] = []
    running = 0
    included_nodes = 0
    for i, ln in enumerate(lines):
        running += len(ln) + 1
        if running > max_chars and i % 3 == 0:   # stop only at a header boundary
            break
        truncated_lines.append(ln)
        if i % 3 == 2:
            included_nodes += 1
    dropped = len(order) - included_nodes
    if dropped > 0:
        truncated_lines.append(f"... ({dropped} nodes omitted, output truncated)")
    return "\n".join(truncated_lines)


# ── line builders ──────────────────────────────────────────────────────────

def _node_header(node) -> str:
    cls = type(node).__name__
    short = _short(node.id)
    label = getattr(node, "label", "") or ""
    return f'Node {short} ({cls}) "{label}"'


def _inputs_line(node, incoming: dict, graph) -> str:
    if not node.inputs:
        return "  inputs: (none)"
    parts = []
    for name in sorted(node.inputs):
        port = node.inputs[name]
        src = incoming.get((node.id, name))
        if src:
            src_node = graph.nodes.get(src[0])
            src_cls = type(src_node).__name__ if src_node else "Unknown"
            parts.append(f"{name}=<{src_cls} {_short(src[0])} .{src[1]}>")
        else:
            parts.append(f"{name}={_scalar(port.default_value)}")
    return "  inputs: " + ", ".join(parts)


def _outputs_line(node, outgoing: dict, graph) -> str:
    if not node.outputs:
        return "  outputs: (none)"
    parts = []
    for name in sorted(node.outputs):
        dsts = outgoing.get((node.id, name), [])
        if not dsts:
            parts.append(f"{name} -> (none)")
            continue
        refs = []
        for dst in dsts:
            dst_node = graph.nodes.get(dst[0])
            dst_cls = type(dst_node).__name__ if dst_node else "Unknown"
            refs.append(f"[{dst_cls} {_short(dst[0])} .{dst[1]}]")
        parts.append(f"{name} -> " + ", ".join(refs))
    return "  outputs: " + "; ".join(parts)


# ── helpers ────────────────────────────────────────────────────────────────

def _short(node_id: str) -> str:
    return (node_id or "")[:6]


def _scalar(value: Any) -> str:
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return repr(value)
    if isinstance(value, (list, tuple)) and len(value) <= 4 and all(
        isinstance(x, (int, float, bool, str, type(None))) for x in value
    ):
        return repr(value)
    return "<non-scalar>"
