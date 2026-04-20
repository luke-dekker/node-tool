"""Mutation op dataclass + safe parser + apply_mutation(graph, op).

Five primitive op kinds mirror Karpathy's edit categories:

    {op:"swap_node_class",  node_id, new_class_name}
    {op:"add_node",         class_name, connections=[(from_id, from_port, in_port), ...]}
    {op:"remove_node",      node_id}
    {op:"set_input",        node_id, port, value}
    {op:"add_connection",   from_id, from_port, to_id, to_port}

apply_mutation() returns (ok: bool, message: str). On failure the graph is
left unchanged — callers should snapshot before mutating so they can revert
when an op is rejected or the subsequent evaluation scores poorly.

The ALLOWED_CLASSES gate (in the node, not here) bounds which type_names
are acceptable for swap_node_class / add_node — autoresearch cannot turn a
training graph into an agent graph.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Any


OP_KINDS = {"swap_node_class", "add_node", "remove_node", "set_input",
            "add_connection"}


@dataclass
class MutationOp:
    op:         str
    node_id:    str = ""
    new_class_name: str = ""
    class_name: str = ""
    port:       str = ""
    value:      Any = None
    from_id:    str = ""
    from_port:  str = ""
    to_id:      str = ""
    to_port:    str = ""
    # For add_node: list of (from_id, from_port, in_port_on_new_node)
    connections: list[list[str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v not in ("", None, [])} | {"op": self.op}

    @classmethod
    def from_dict(cls, d: dict) -> "MutationOp":
        kind = str(d.get("op") or "").strip()
        if kind not in OP_KINDS:
            raise ValueError(
                f"Unknown op kind {kind!r}; expected one of {sorted(OP_KINDS)}"
            )
        return cls(
            op=kind,
            node_id=str(d.get("node_id") or ""),
            new_class_name=str(d.get("new_class_name") or ""),
            class_name=str(d.get("class_name") or ""),
            port=str(d.get("port") or ""),
            value=d.get("value"),
            from_id=str(d.get("from_id") or ""),
            from_port=str(d.get("from_port") or ""),
            to_id=str(d.get("to_id") or ""),
            to_port=str(d.get("to_port") or ""),
            connections=list(d.get("connections") or []),
        )


def parse_mutation_json(raw: str) -> MutationOp:
    """Parse an LLM-emitted JSON blob into a MutationOp.

    Tolerates common emissions:
      - a fenced code block (```json ... ```)
      - extra prose before/after the object
      - object wrapped in a {"mutation": {...}} envelope
    """
    text = _strip_fences(raw)
    # Find the first balanced '{ ... }' block
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise ValueError(f"parse_mutation_json: no JSON object in {raw!r}")
    obj = json.loads(text[start:end + 1])
    if "mutation" in obj and isinstance(obj["mutation"], dict):
        obj = obj["mutation"]
    return MutationOp.from_dict(obj)


def apply_mutation(
    graph, op: MutationOp, *,
    allowlist: set[str] | None = None,
    cone: set[str] | None = None,
) -> tuple[bool, str]:
    """Apply `op` to `graph` in place. Returns (ok, message).

    `allowlist`: permitted class type_names for swap_node_class / add_node.
    None = any registered class is allowed.
    `cone`: permitted node ids for mutations that target a node. None = any.
    """
    from nodes import NODE_REGISTRY

    def _in_cone(nid: str) -> bool:
        return cone is None or nid in cone

    if op.op == "swap_node_class":
        if not _in_cone(op.node_id):
            return False, f"node {op.node_id[:6]} outside mutable cone"
        if op.node_id not in graph.nodes:
            return False, f"node {op.node_id[:6]} not in graph"
        new_cls = NODE_REGISTRY.get(op.new_class_name)
        if new_cls is None:
            return False, f"unknown class {op.new_class_name!r}"
        if allowlist is not None and op.new_class_name not in allowlist:
            return False, f"class {op.new_class_name!r} not in allowlist"
        # Preserve id and connections; just swap the node instance.
        old = graph.nodes[op.node_id]
        new_node = new_cls()
        new_node.id = old.id
        graph.nodes[op.node_id] = new_node
        # Port defaults that exist on both old and new are copied across.
        for pname in new_node.inputs:
            if pname in old.inputs:
                new_node.inputs[pname].default_value = old.inputs[pname].default_value
        return True, f"swapped {op.node_id[:6]} -> {op.new_class_name}"

    if op.op == "remove_node":
        if not _in_cone(op.node_id):
            return False, f"node {op.node_id[:6]} outside mutable cone"
        if op.node_id not in graph.nodes:
            return False, f"node {op.node_id[:6]} not in graph"
        graph.remove_node(op.node_id)
        return True, f"removed {op.node_id[:6]}"

    if op.op == "add_node":
        new_cls = NODE_REGISTRY.get(op.class_name)
        if new_cls is None:
            return False, f"unknown class {op.class_name!r}"
        if allowlist is not None and op.class_name not in allowlist:
            return False, f"class {op.class_name!r} not in allowlist"
        new_node = new_cls()
        graph.add_node(new_node)
        for triple in op.connections:
            if len(triple) != 3:
                continue
            from_id, from_port, in_port = triple
            graph.add_connection(from_id, from_port, new_node.id, in_port)
        return True, f"added {new_node.id[:6]} ({op.class_name})"

    if op.op == "set_input":
        if not _in_cone(op.node_id):
            return False, f"node {op.node_id[:6]} outside mutable cone"
        node = graph.nodes.get(op.node_id)
        if node is None:
            return False, f"node {op.node_id[:6]} not in graph"
        if op.port not in node.inputs:
            return False, f"port {op.port!r} not on node"
        # Reject file paths that escape cwd
        if isinstance(op.value, str) and _escapes_cwd(op.value):
            return False, f"value {op.value!r} escapes cwd"
        node.inputs[op.port].default_value = op.value
        return True, f"set {op.node_id[:6]}.{op.port} = {op.value!r}"

    if op.op == "add_connection":
        if not (_in_cone(op.from_id) and _in_cone(op.to_id)):
            return False, "connection endpoints outside mutable cone"
        c = graph.add_connection(op.from_id, op.from_port,
                                  op.to_id, op.to_port)
        if c is None:
            return False, "connection rejected (cycle or missing port)"
        return True, f"connected {op.from_id[:6]}.{op.from_port} -> " \
                     f"{op.to_id[:6]}.{op.to_port}"

    return False, f"unknown op: {op.op!r}"


def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` markdown fences if present."""
    t = text.strip()
    if t.startswith("```"):
        # Drop first line (```json) and trailing ```
        lines = t.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines)
    return t


def _escapes_cwd(path: str) -> bool:
    """True if a string looks like a filesystem path that escapes cwd."""
    import os
    if not path:
        return False
    if os.path.isabs(path):
        return True
    norm = os.path.normpath(path)
    return norm.startswith("..") or "/../" in path or "\\..\\" in path
