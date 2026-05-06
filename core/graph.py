"""Graph, Connection, topological executor, and undo/redo command stack."""

from __future__ import annotations
import hashlib
import json
import re
from typing import Any
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from core.node import BaseNode
from core.port_types import PortTypeRegistry


_PLUGIN_PREFIX_RE = re.compile(r"^[a-z]{2,3}_")


def _compact_type_name(type_name: str) -> str:
    """Derive a CamelCase alias prefix from a type_name.

    Strips a leading plugin prefix like `pt_` / `ag_` / `io_` (2–3 lowercase
    letters + underscore) and turns snake_case into CamelCase so the result
    reads naturally as a node alias: `pt_linear` → `Linear`,
    `ag_ollama_client` → `OllamaClient`, `pt_input_marker` → `InputMarker`.
    Falls back to a titlecased version of the raw type_name on odd input.
    """
    stripped = _PLUGIN_PREFIX_RE.sub("", type_name or "") or (type_name or "node")
    parts = [p for p in stripped.split("_") if p]
    return "".join(p[:1].upper() + p[1:] for p in parts) or "Node"


# ── Command stack (undo/redo) ─────────────────────────────────────────────────

class Command(ABC):
    @abstractmethod
    def execute(self) -> None: ...

    @abstractmethod
    def undo(self) -> None: ...


class CommandStack:
    def __init__(self, max_depth: int = 100):
        self._undo: deque[Command] = deque(maxlen=max_depth)
        self._redo: deque[Command] = deque(maxlen=max_depth)

    def push(self, cmd: Command) -> None:
        cmd.execute()
        self._undo.append(cmd)
        self._redo.clear()

    def undo(self) -> None:
        if self._undo:
            cmd = self._undo.pop()
            cmd.undo()
            self._redo.append(cmd)

    def redo(self) -> None:
        if self._redo:
            cmd = self._redo.pop()
            cmd.execute()
            self._undo.append(cmd)

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)


def _canonical_value(v: Any) -> Any:
    """Return a JSON-safe representation of a port default value.

    Non-serializable values (tensors, nn.Modules, arbitrary objects) collapse
    to None — the snapshot captures graph structure, not runtime caches.
    """
    if isinstance(v, (int, float, bool, str, type(None))):
        return v
    if isinstance(v, (list, tuple)):
        return [_canonical_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _canonical_value(v[k]) for k in sorted(v)}
    return None


def _hash_payload(payload: dict) -> str:
    """sha1 of the canonical JSON of `payload`. Stable across processes."""
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


class Connection:
    """A directed link from one node's output port to another's input port."""

    def __init__(self, from_node_id: str, from_port: str,
                 to_node_id: str, to_port: str) -> None:
        self.from_node_id = from_node_id
        self.from_port = from_port
        self.to_node_id = to_node_id
        self.to_port = to_port

    def __repr__(self) -> str:
        return (f"<Connection {self.from_node_id[:8]}.{self.from_port}"
                f" -> {self.to_node_id[:8]}.{self.to_port}>")


class Graph:
    """
    Manages a directed acyclic graph of nodes and connections.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, BaseNode] = {}
        self.connections: list[Connection] = []
        # Snapshot cache: hash → serialized payload. Populated by snapshot();
        # consulted by revert_to(). In-memory only; NOT persisted to disk.
        self._snapshots: dict[str, dict] = {}

    def clear(self) -> None:
        """Remove all nodes and connections in place (preserves object identity)."""
        self.nodes.clear()
        self.connections.clear()

    # ── Node management ────────────────────────────────────────────────────

    def add_node(self, node: BaseNode) -> BaseNode:
        self.nodes[node.id] = node
        if not node.alias:
            node.alias = self._alloc_alias(node.type_name)
        return node

    def _alloc_alias(self, type_name: str) -> str:
        """Assign a stable, human-readable per-instance alias like 'Linear2'
        or 'InputMarker1'. Counter is per-type and per-graph; counts nodes
        of the same type already present. Stays fixed for the node's life —
        even if other nodes of the same type are added/removed later — so
        the user can memorize 'Linear2' and have it keep pointing at the
        same instance. Autoresearch uses this as the target-id prefix so
        the history table mirrors what's on the canvas.
        """
        base = _compact_type_name(type_name)
        used = {n.alias for n in self.nodes.values() if n.alias}
        # Count by matching prefix, then scan forward to the first free slot.
        # Re-use of 'N' is fine: aliases are released on node removal (the
        # gap would otherwise grow forever over long edit sessions).
        n = 1
        while f"{base}{n}" in used:
            n += 1
        return f"{base}{n}"

    def remove_node(self, node_id: str) -> None:
        self.nodes.pop(node_id, None)
        self.connections = [
            c for c in self.connections
            if c.from_node_id != node_id and c.to_node_id != node_id
        ]

    def get_node(self, node_id: str) -> BaseNode | None:
        return self.nodes.get(node_id)

    def nodes_by_role(self, role: str) -> list[BaseNode]:
        """Return all nodes whose class-level marker_role matches `role`.

        Lets GUIs and training backends discover special nodes (input markers,
        training targets, etc.) without hardcoding type_name strings. See
        core.node.MarkerRole for the canonical role constants.
        """
        return [n for n in self.nodes.values() if n.marker_role == role]

    # ── Connection management ───────────────────────────────────────────────

    def add_connection(self, from_node_id: str, from_port: str,
                       to_node_id: str, to_port: str) -> Connection | None:
        """Add a connection. Returns None if it would create a cycle or ports don't exist."""
        from_node = self.nodes.get(from_node_id)
        to_node = self.nodes.get(to_node_id)
        if from_node is None or to_node is None:
            return None
        if from_port not in from_node.outputs:
            return None
        if to_port not in to_node.inputs:
            return None

        # Remove any existing connection to the same input port
        self.connections = [
            c for c in self.connections
            if not (c.to_node_id == to_node_id and c.to_port == to_port)
        ]

        conn = Connection(from_node_id, from_port, to_node_id, to_port)

        # Check for cycles before committing
        self.connections.append(conn)
        if self._has_cycle():
            self.connections.pop()
            return None

        return conn

    def remove_connection(self, from_node_id: str, from_port: str,
                          to_node_id: str, to_port: str) -> None:
        self.connections = [
            c for c in self.connections
            if not (c.from_node_id == from_node_id and c.from_port == from_port
                    and c.to_node_id == to_node_id and c.to_port == to_port)
        ]

    def remove_connection_obj(self, conn: Connection) -> None:
        try:
            self.connections.remove(conn)
        except ValueError:
            pass

    # ── Topology ────────────────────────────────────────────────────────────

    def _build_adjacency(self) -> tuple[dict[str, set[str]], dict[str, int]]:
        """Build adjacency list and in-degree map for topological sort."""
        adj: dict[str, set[str]] = defaultdict(set)
        in_degree: dict[str, int] = {nid: 0 for nid in self.nodes}
        for c in self.connections:
            if c.from_node_id in self.nodes and c.to_node_id in self.nodes:
                if c.to_node_id not in adj[c.from_node_id]:
                    adj[c.from_node_id].add(c.to_node_id)
                    in_degree[c.to_node_id] += 1
        return adj, in_degree

    def _has_cycle(self) -> bool:
        adj, in_degree = self._build_adjacency()
        queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
        visited = 0
        while queue:
            nid = queue.popleft()
            visited += 1
            for neighbour in adj[nid]:
                in_degree[neighbour] -= 1
                if in_degree[neighbour] == 0:
                    queue.append(neighbour)
        return visited != len(self.nodes)

    def topological_order(self) -> list[str]:
        """Return node IDs in topological order (Kahn's algorithm)."""
        adj, in_degree = self._build_adjacency()
        queue = deque(sorted(nid for nid, deg in in_degree.items() if deg == 0))
        order: list[str] = []
        while queue:
            nid = queue.popleft()
            order.append(nid)
            for neighbour in sorted(adj[nid]):
                in_degree[neighbour] -= 1
                if in_degree[neighbour] == 0:
                    queue.append(neighbour)
        return order  # may be shorter than nodes if there are cycles

    def subgraph_between(self, a_id: str, b_id: str) -> list[str]:
        """Topologically-ordered node ids in the cone between A (inclusive)
        and B (inclusive).

        Cone = descendants(A) ∩ ancestors(B), both inclusive of the endpoints.
        Used by `GraphAsToolNode` to scope subgraph execution and by
        autoresearch's mutator to bound the mutable region. Returns [] if B
        is unreachable from A. Raises `KeyError` if either id is missing.
        """
        if a_id not in self.nodes:
            raise KeyError(f"subgraph_between: node not found: {a_id}")
        if b_id not in self.nodes:
            raise KeyError(f"subgraph_between: node not found: {b_id}")

        adj: dict[str, set[str]]  = {nid: set() for nid in self.nodes}
        radj: dict[str, set[str]] = {nid: set() for nid in self.nodes}
        for c in self.connections:
            if c.from_node_id in adj and c.to_node_id in adj:
                adj[c.from_node_id].add(c.to_node_id)
                radj[c.to_node_id].add(c.from_node_id)

        descendants: set[str] = {a_id}
        stack = [a_id]
        while stack:
            cur = stack.pop()
            for nxt in adj[cur]:
                if nxt not in descendants:
                    descendants.add(nxt)
                    stack.append(nxt)

        ancestors: set[str] = {b_id}
        stack = [b_id]
        while stack:
            cur = stack.pop()
            for prv in radj[cur]:
                if prv not in ancestors:
                    ancestors.add(prv)
                    stack.append(prv)

        cone = descendants & ancestors
        if not cone:
            return []
        return [nid for nid in self.topological_order() if nid in cone]

    # ── Snapshot / revert (transactional graph state) ──────────────────────

    def snapshot(self) -> str:
        """Capture current graph state, return a content-addressed hash.

        State captured:
          - Each node's id, type_name, and JSON-serializable input default values
          - All connections (from_id, from_port, to_id, to_port)

        Runtime caches (_layer, _probe_tensor, etc.) are NOT captured — run
        graph.execute() again after a revert to re-materialize them.

        The hash is stable across snapshot() calls on an unmodified graph,
        so two calls return the same hash iff nothing has changed. The full
        payload is retained in `self._snapshots` so `revert_to(hash)` works
        in the same process; not persisted to disk.

        Used by autoresearch's mutate→eval→keep/revert loop and, via the
        `CommandStack`-free path, as ad-hoc undo in the GUI.
        """
        payload = self._build_snapshot_payload()
        key = _hash_payload(payload)
        self._snapshots[key] = payload
        return key

    def revert_to(self, snapshot_hash: str) -> None:
        """Restore graph state to a previous snapshot. Mutates in place.

        Strategy (surgical, not full rebuild):
          1. Remove any node present now but not in the snapshot.
          2. Recreate any node in the snapshot but missing now (via NODE_REGISTRY).
          3. For nodes in both, reset input defaults to the snapshot values.
          4. Clear connections and re-add all from the snapshot.

        WARNING: external callers that held references to nodes created AFTER
        the snapshot will end up with stale objects that no longer belong to
        the graph — treat the returned Graph as authoritative. Nodes in both
        sides retain object identity.
        """
        payload = self._snapshots.get(snapshot_hash)
        if payload is None:
            raise KeyError(f"Unknown snapshot hash: {snapshot_hash!r}")
        self._restore_from_payload(payload)

    def _build_snapshot_payload(self) -> dict:
        nodes = []
        for nid in sorted(self.nodes):
            node = self.nodes[nid]
            inputs = {}
            for pname in sorted(node.inputs):
                inputs[pname] = _canonical_value(node.inputs[pname].default_value)
            nodes.append({
                "id":        nid,
                "type_name": node.type_name,
                "alias":     node.alias,
                "inputs":    inputs,
            })
        connections = sorted([
            [c.from_node_id, c.from_port, c.to_node_id, c.to_port]
            for c in self.connections
        ])
        return {"nodes": nodes, "connections": connections}

    def _restore_from_payload(self, payload: dict) -> None:
        from nodes import NODE_REGISTRY  # lazy: avoids circular import at module top
        snap_nodes = {n["id"]: n for n in payload.get("nodes", [])}

        # 1. Remove nodes not in snapshot
        for nid in list(self.nodes.keys()):
            if nid not in snap_nodes:
                self.nodes.pop(nid, None)

        # 2. Recreate missing nodes; 3. reset defaults on surviving nodes
        for nid, snap in snap_nodes.items():
            node = self.nodes.get(nid)
            if node is None:
                cls_ = NODE_REGISTRY.get(snap["type_name"])
                if cls_ is None:
                    raise RuntimeError(
                        f"revert_to: node type {snap['type_name']!r} not in registry; "
                        "cannot recreate"
                    )
                node = cls_()
                node.id = nid
                self.nodes[nid] = node
            # Restore alias from snapshot; backfill for pre-alias snapshots.
            node.alias = snap.get("alias", "") or node.alias or self._alloc_alias(node.type_name)
            for pname, val in snap.get("inputs", {}).items():
                if pname in node.inputs:
                    node.inputs[pname].default_value = val

        # 4. Reset connections
        self.connections = [
            Connection(c[0], c[1], c[2], c[3])
            for c in payload.get("connections", [])
        ]

    # ── Execution ────────────────────────────────────────────────────────────

    def execute(self) -> tuple[dict[str, dict[str, Any]], list[str], dict[str, dict[str, str]]]:
        """
        Execute all reachable nodes in topological order.
        Returns (node_outputs, terminal_lines, errors).
        node_outputs: {node_id: {port_name: value}}
        errors: {node_id: {"message", "type", "label"}}
        """
        order = self.topological_order()
        stored: dict[str, dict[str, Any]] = {}  # node_id -> {port: value}
        terminal_lines: list[str] = []
        errors: dict[str, dict[str, str]] = {}

        # Build a lookup: (to_node_id, to_port) -> (from_node_id, from_port)
        conn_map: dict[tuple[str, str], tuple[str, str]] = {}
        for c in self.connections:
            conn_map[(c.to_node_id, c.to_port)] = (c.from_node_id, c.from_port)

        for node_id in order:
            node = self.nodes[node_id]
            # Expose a graph reference on the node so graph-aware nodes (e.g.
            # GraphAsToolNode) can introspect peers and sub-cones. Most nodes
            # ignore this attribute; it's a non-breaking opt-in hook.
            node._graph = self
            inputs: dict[str, Any] = {}

            for port_name, port in node.inputs.items():
                key = (node_id, port_name)
                if key in conn_map:
                    from_node_id, from_port = conn_map[key]
                    if from_node_id in stored and from_port in stored[from_node_id]:
                        raw = stored[from_node_id][from_port]
                        if raw is None:
                            # Upstream produced None — fall back to this port's
                            # default value so live-preview workflows (where the
                            # user sets a port default to test a subgraph in
                            # isolation) don't get clobbered by an upstream that
                            # hasn't been provided real data yet. Matches the
                            # "no connection" branch below.
                            inputs[port_name] = port.default_value
                        else:
                            inputs[port_name] = PortTypeRegistry.coerce_value(port.port_type, raw)
                    else:
                        inputs[port_name] = port.default_value
                else:
                    inputs[port_name] = port.default_value

            try:
                outputs = node.execute(inputs)
            except Exception as exc:
                outputs = {}
                terminal_lines.append(f"[ERROR] Node {node.label} ({node_id[:8]}): {exc}")
                errors[node_id] = {
                    "message": str(exc),
                    "type": type(exc).__name__,
                    "label": node.label,
                }

            stored[node_id] = outputs or {}

            # Collect terminal output from any node that produces __terminal__
            if "__terminal__" in (outputs or {}):
                terminal_lines.append(str(outputs["__terminal__"]))

        return stored, terminal_lines, errors
