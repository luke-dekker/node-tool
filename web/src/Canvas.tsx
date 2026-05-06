// The graph canvas. Uses React Flow for pan/zoom/minimap and our custom
// GraphNode for rendering. Connection and disconnection are round-tripped
// to the server so that graph state stays authoritative there.

import { useCallback, useMemo } from "react";
import ReactFlow, {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
  type Connection,
  type Edge,
  type EdgeChange,
  type NodeChange,
} from "reactflow";
import "reactflow/dist/style.css";

import { GraphNode } from "./GraphNode";
import { useStore } from "./store";
import { theme } from "./theme";

export function Canvas() {
  const nodes = useStore((s) => s.nodes);
  const edges = useStore((s) => s.edges);
  const setNodes = useStore((s) => s.setNodes);
  const setEdges = useStore((s) => s.setEdges);
  const setSelected = useStore((s) => s.setSelected);
  const client = useStore((s) => s.client);

  const nodeTypes = useMemo(() => ({ nodetool: GraphNode }), []);

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => {
      setNodes((prev) => applyNodeChanges(changes, prev));
      for (const c of changes) {
        if (c.type === "select" && c.selected) setSelected(c.id);
        if (c.type === "remove") {
          client.call("remove_node", { node_id: c.id }).catch(console.error);
        }
      }
    },
    [setNodes, setSelected, client],
  );

  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      setEdges((prev) => applyEdgeChanges(changes, prev));
      for (const c of changes) {
        if (c.type === "remove") {
          const edge = edges.find((e) => e.id === c.id);
          if (edge) {
            client
              .call("disconnect", {
                from_node: edge.source,
                from_port: edge.sourceHandle,
                to_node: edge.target,
                to_port: edge.targetHandle,
              })
              .catch(console.error);
          }
        }
      }
    },
    [setEdges, edges, client],
  );

  const onConnect = useCallback(
    (conn: Connection) => {
      if (!conn.source || !conn.target || !conn.sourceHandle || !conn.targetHandle) return;
      const newEdge: Edge = {
        id: `${conn.source}:${conn.sourceHandle}->${conn.target}:${conn.targetHandle}`,
        source: conn.source,
        sourceHandle: conn.sourceHandle,
        target: conn.target,
        targetHandle: conn.targetHandle,
        style: { stroke: theme.accent, strokeWidth: 2 },
      };
      setEdges((prev) => addEdge(newEdge, prev));
      client
        .call("connect", {
          from_node: conn.source,
          from_port: conn.sourceHandle,
          to_node: conn.target,
          to_port: conn.targetHandle,
        })
        .catch((err) => {
          console.error("connect failed:", err);
          // Roll back the optimistic edge if server rejected (e.g. type mismatch)
          setEdges((prev) => prev.filter((e) => e.id !== newEdge.id));
        });
    },
    [setEdges, client],
  );

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      onConnect={onConnect}
      nodeTypes={nodeTypes}
      fitView={false}
      proOptions={{ hideAttribution: true }}
      defaultEdgeOptions={{ style: { stroke: theme.accent, strokeWidth: 2 } }}
      deleteKeyCode={["Delete", "Backspace"]}
    >
      <Background variant={BackgroundVariant.Dots} gap={24} size={1} color="#222a3a" />
      <MiniMap
        style={{ background: theme.bgMid, border: `1px solid ${theme.border}` }}
        maskColor="rgba(0,0,0,0.5)"
        nodeColor={() => theme.accentDim}
        pannable
        zoomable
      />
      <Controls style={{ background: theme.bgMid, border: `1px solid ${theme.border}` }} />
    </ReactFlow>
  );
}
