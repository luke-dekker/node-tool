"""Graph, Connection, topological executor, and undo/redo command stack."""

from __future__ import annotations
from typing import Any
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from core.node import BaseNode
from core.port_types import PortTypeRegistry


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

    def clear(self) -> None:
        """Remove all nodes and connections in place (preserves object identity)."""
        self.nodes.clear()
        self.connections.clear()

    # ── Node management ────────────────────────────────────────────────────

    def add_node(self, node: BaseNode) -> BaseNode:
        self.nodes[node.id] = node
        return node

    def remove_node(self, node_id: str) -> None:
        self.nodes.pop(node_id, None)
        self.connections = [
            c for c in self.connections
            if c.from_node_id != node_id and c.to_node_id != node_id
        ]

    def get_node(self, node_id: str) -> BaseNode | None:
        return self.nodes.get(node_id)

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
