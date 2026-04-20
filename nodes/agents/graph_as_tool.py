"""GraphAsToolNode — wrap an A/B-bounded subgraph as a callable TOOL.

The wrapped region is the topological cone between an `INPUT` marker (A) and
a `TRAIN_TARGET` marker (B) that share the same `group` — same pair the
training panel uses. No new marker role is needed (Decision 5 of DESIGN.md).

Tool input schema: one property per A-marker output port. Callable kwargs
override A's outputs; the cone executes in topological order; the caller
receives B's output dict.

Graph access: at execute() time, the node reads `self._graph` which
`Graph.execute()` sets before each node's execute(). If the node is run
standalone (e.g., in a unit test without a Graph), it raises a clear error.
"""
from __future__ import annotations
from typing import Any

from core.node import BaseNode, MarkerRole, PortType


class GraphAsToolNode(BaseNode):
    type_name   = "ag_graph_as_tool"
    label       = "Graph-as-Tool"
    category    = "Agents"
    subcategory = "Tools"
    description = ("Wrap the subgraph between an A (INPUT) marker and a B "
                   "(TRAIN_TARGET) marker of the same `group` as a callable "
                   "tool the agent can invoke.")

    def _setup_ports(self) -> None:
        self.add_input("name", PortType.STRING, default="subgraph_tool",
                       description="Tool name the LLM will reference")
        self.add_input("description", PortType.STRING,
                       default="Invoke the wrapped subgraph.",
                       description="Drives the LLM's choice of when to call this tool")
        self.add_input("group", PortType.STRING, default="task_1",
                       description=("A/B marker group to wrap. Uses the first "
                                    "matching INPUT marker and TRAIN_TARGET "
                                    "marker in the graph."))
        self.add_input("side_effect", PortType.BOOL, default=False,
                       description=("True if the wrapped graph has side effects "
                                    "(file/network/hardware)."))
        self.add_output("tool", "TOOL")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        from plugins.agents._llm.protocol import ToolDef  # deferred

        graph = getattr(self, "_graph", None)
        if graph is None:
            raise RuntimeError(
                "GraphAsToolNode needs a live Graph to discover A/B markers — "
                "was this node executed standalone? Add it to a Graph and run "
                "graph.execute()."
            )

        group = (inputs.get("group") or "task_1").strip() or "task_1"
        a, b = _find_markers(graph, group)
        if a is None:
            raise RuntimeError(
                f"GraphAsToolNode: no INPUT marker (A) with group={group!r}"
            )
        if b is None:
            raise RuntimeError(
                f"GraphAsToolNode: no TRAIN_TARGET marker (B) with group={group!r}"
            )

        cone = graph.subgraph_between(a.id, b.id)
        if not cone:
            raise RuntimeError(
                f"GraphAsToolNode: no path from A to B for group={group!r}"
            )

        # Tool's JSON schema: one required property per A-output port.
        a_output_ports = list(a.outputs.keys())
        properties = {p: {"type": "object",
                          "description": f"Value for marker A's {p!r} output port"}
                      for p in a_output_ports}
        schema = {
            "type":       "object",
            "properties": properties,
            "required":   a_output_ports,
        }

        name = (inputs.get("name") or "subgraph_tool").strip() or "subgraph_tool"
        description = ((inputs.get("description") or "").strip()
                       or f"Run subgraph between A/B markers of group {group!r}.")

        # Snapshot ids + cone — closure captures these for later invocations.
        def _call(**kwargs) -> dict:
            overrides = {p: kwargs[p] for p in a_output_ports if p in kwargs}
            # Back-compat with InputMarkerNode's stash pattern.
            if hasattr(a, "_probe_tensor") and "tensor" in overrides:
                a._probe_tensor = overrides["tensor"]
            return _run_cone(graph, cone, a.id, b.id, overrides)

        return {"tool": ToolDef(
            name=name, description=description, input_schema=schema,
            callable=_call, side_effect=bool(inputs.get("side_effect")),
        )}

    def export(self, iv, ov):
        return [], [f"# GraphAsToolNode export pending Phase D"]


# ── Helpers ────────────────────────────────────────────────────────────────

def _find_markers(graph, group: str):
    """Return (a_node, b_node) for the given group, or (None, None) pieces
    missing. Picks the first match of each role.
    """
    a = b = None
    for n in graph.nodes_by_role(MarkerRole.INPUT):
        if _marker_group(n) == group:
            a = n
            break
    for n in graph.nodes_by_role(MarkerRole.TRAIN_TARGET):
        if _marker_group(n) == group:
            b = n
            break
    return a, b


def _marker_group(node) -> str:
    port = node.inputs.get("group") if hasattr(node, "inputs") else None
    if port is None:
        return ""
    return str(port.default_value or "")


def _run_cone(graph, cone: list[str], a_id: str, b_id: str,
              a_overrides: dict) -> dict:
    """Execute every node in `cone` in topological order, seeding A's outputs
    from `a_overrides`. Returns B's output dict.
    """
    stored: dict[str, dict] = {a_id: dict(a_overrides)}
    conn_map = {
        (c.to_node_id, c.to_port): (c.from_node_id, c.from_port)
        for c in graph.connections
    }
    for nid in cone:
        if nid == a_id:
            continue
        node = graph.nodes[nid]
        inputs = {}
        for pname, port in node.inputs.items():
            src = conn_map.get((nid, pname))
            if src and src[0] in stored and src[1] in stored[src[0]]:
                inputs[pname] = stored[src[0]][src[1]]
            else:
                inputs[pname] = port.default_value
        try:
            out = node.execute(inputs) or {}
        except Exception as exc:
            raise RuntimeError(
                f"GraphAsTool: subgraph node {node.label!r} ({nid[:6]}) raised: {exc}"
            ) from exc
        stored[nid] = out
    return dict(stored.get(b_id, {}))
