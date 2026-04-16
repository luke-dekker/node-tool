// App-wide state: RPC client, connection status, registry, palette search,
// and the live graph (nodes + edges mirrored from React Flow). Canvas nodes
// carry the full NodeInstance payload returned by the server so the
// inspector can read port descriptions and defaults without another RPC.

import { create } from "zustand";
import type { Edge, Node } from "reactflow";
import { RpcClient, type ConnState } from "./rpc";
import type { NodeInstance, Registry } from "./types";

const SERVER_URL = "ws://127.0.0.1:9800";

// React Flow stores arbitrary data on each node; we stash the full NodeInstance.
export type FlowNode = Node<{ instance: NodeInstance }>;
export type FlowEdge = Edge;

export interface TemplateInfo {
  label: string;
  description: string;
}

interface Store {
  conn: ConnState;
  client: RpcClient;
  registry: Registry | null;
  search: string;
  nodes: FlowNode[];
  edges: FlowEdge[];
  selectedId: string | null;
  errors: Record<string, { message: string; type: string; label: string }>;
  templates: TemplateInfo[];
  pluginPanels: string[];

  setConn: (s: ConnState) => void;
  setRegistry: (r: Registry) => void;
  setSearch: (q: string) => void;
  setNodes: (fn: (prev: FlowNode[]) => FlowNode[]) => void;
  setEdges: (fn: (prev: FlowEdge[]) => FlowEdge[]) => void;
  setSelected: (id: string | null) => void;
  setErrors: (e: Record<string, { message: string; type: string; label: string }>) => void;

  terminal: string[];

  connect: () => void;
  spawnNode: (typeName: string) => Promise<void>;
  appendLog: (line: string) => void;
  clearGraph: () => Promise<void>;
  runGraph: () => Promise<void>;
  exportCode: () => Promise<string>;
  saveGraph: () => Promise<void>;
  loadGraph: (file: File) => Promise<void>;
  loadTemplate: (label: string) => Promise<void>;
}

// Stagger newly spawned nodes so they don't pile on top of each other.
let spawnCount = 0;
function nextSpawnPos(): { x: number; y: number } {
  const col = spawnCount % 4;
  const row = Math.floor(spawnCount / 4);
  spawnCount += 1;
  return { x: 120 + col * 260, y: 80 + row * 140 + col * 20 };
}

export const useStore = create<Store>((set, get) => {
  const client = new RpcClient(SERVER_URL, (s) => {
    set({ conn: s });
    if (s === "open") {
      const c = get().client;
      c.call<Registry>("get_registry")
        .then((reg) => set({ registry: reg }))
        .catch((err) => console.error("get_registry failed:", err));
      c.call<{ templates: TemplateInfo[] }>("get_templates")
        .then((res) => set({ templates: res.templates ?? [] }))
        .catch((err) => console.error("get_templates failed:", err));
      c.call<{ panels: string[] }>("get_plugin_panels")
        .then((res) => set({ pluginPanels: res.panels ?? [] }))
        .catch((err) => console.error("get_plugin_panels failed:", err));
    }
  });

  return {
    conn: "idle",
    client,
    registry: null,
    search: "",
    nodes: [],
    edges: [],
    selectedId: null,
    errors: {},
    templates: [],
    pluginPanels: [],

    setConn: (s) => set({ conn: s }),
    setRegistry: (r) => set({ registry: r }),
    setSearch: (q) => set({ search: q }),
    setNodes: (fn) => set((st) => ({ nodes: fn(st.nodes) })),
    setEdges: (fn) => set((st) => ({ edges: fn(st.edges) })),
    setSelected: (id) => set({ selectedId: id }),
    setErrors: (e) => set({ errors: e }),

    connect: () => client.connect(),

    terminal: [],

    spawnNode: async (typeName: string) => {
      try {
        const instance = await client.call<NodeInstance>("add_node", {
          type_name: typeName,
        });
        const pos = nextSpawnPos();
        const flowNode: FlowNode = {
          id: instance.id,
          type: "nodetool",
          position: pos,
          data: { instance },
        };
        set((st) => ({ nodes: [...st.nodes, flowNode] }));
      } catch (err) {
        console.error("add_node failed:", err);
      }
    },

    appendLog: (line: string) =>
      set((st) => ({ terminal: [...st.terminal, line].slice(-500) })),

    clearGraph: async () => {
      try {
        await client.call("clear");
        spawnCount = 0;
        set({ nodes: [], edges: [], selectedId: null, errors: {} });
      } catch (err) {
        console.error("clear failed:", err);
      }
    },

    runGraph: async () => {
      const append = get().appendLog;
      append("> Running graph...");
      try {
        const res = await client.call<{
          outputs?: Record<string, Record<string, string>>;
          terminal?: string[];
          errors?: Record<string, { message: string; type: string; label: string }>;
          error?: string;
        }>("execute");
        if (res.error) {
          append(`[FATAL] ${res.error}`);
          return;
        }
        for (const line of res.terminal ?? []) append(line);
        set({ errors: res.errors ?? {} });
        const outCount = Object.keys(res.outputs ?? {}).length;
        const errCount = Object.keys(res.errors ?? {}).length;
        append(
          errCount > 0
            ? `Done with ${errCount} error(s) across ${outCount} node(s).`
            : `Done. ${outCount} node(s) produced output.`,
        );
      } catch (err) {
        append(`[ERROR] execute: ${String(err)}`);
      }
    },

    exportCode: async () => {
      try {
        const res = await client.call<{ code: string }>("export_code");
        return res.code;
      } catch (err) {
        console.error("export_code failed:", err);
        return `# export failed: ${String(err)}`;
      }
    },

    saveGraph: async () => {
      // Collect canvas positions and ask the server to serialize the graph,
      // then trigger a browser download of the JSON.
      const positions: Record<string, [number, number]> = {};
      for (const n of get().nodes) {
        positions[n.id] = [n.position.x, n.position.y];
      }
      try {
        const data = await client.call("serialize_graph", { positions });
        const blob = new Blob([JSON.stringify(data, null, 2)], {
          type: "application/json",
        });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "graph.json";
        a.click();
        URL.revokeObjectURL(url);
        get().appendLog(`> Graph saved (${get().nodes.length} nodes)`);
      } catch (err) {
        get().appendLog(`[ERROR] save: ${String(err)}`);
      }
    },

    loadGraph: async (file: File) => {
      try {
        const text = await file.text();
        const data = JSON.parse(text);
        const res = await client.call<{
          nodes: Record<string, NodeInstance>;
          connections: { from_node: string; from_port: string; to_node: string; to_port: string }[];
          positions: Record<string, [number, number]>;
        }>("deserialize_graph", { data });
        applyServerGraph(set, res);
        spawnCount = Object.keys(res.nodes).length;
        get().appendLog(`> Loaded ${file.name} (${Object.keys(res.nodes).length} nodes)`);
      } catch (err) {
        get().appendLog(`[ERROR] load: ${String(err)}`);
      }
    },

    loadTemplate: async (label: string) => {
      try {
        const res = await client.call<{
          nodes: Record<string, NodeInstance>;
          connections: { from_node: string; from_port: string; to_node: string; to_port: string }[];
          positions: Record<string, [number, number]>;
        }>("load_template", { label });
        applyServerGraph(set, res);
        spawnCount = Object.keys(res.nodes).length;
        get().appendLog(`> Template loaded: ${label}`);
      } catch (err) {
        get().appendLog(`[ERROR] template: ${String(err)}`);
      }
    },
  };
});

// Replace the local node/edge state with what the server returned, preserving
// positions reported by the server (templates and graph files include them).
function applyServerGraph(
  set: (partial: Partial<Store> | ((s: Store) => Partial<Store>)) => void,
  res: {
    nodes: Record<string, NodeInstance>;
    connections: { from_node: string; from_port: string; to_node: string; to_port: string }[];
    positions: Record<string, [number, number]>;
  },
) {
  const flowNodes: FlowNode[] = Object.values(res.nodes).map((instance) => {
    const p = res.positions[instance.id];
    return {
      id: instance.id,
      type: "nodetool",
      position: { x: p ? p[0] : 100, y: p ? p[1] : 100 },
      data: { instance },
    };
  });
  const flowEdges: FlowEdge[] = res.connections.map((c) => ({
    id: `${c.from_node}:${c.from_port}->${c.to_node}:${c.to_port}`,
    source: c.from_node,
    sourceHandle: c.from_port,
    target: c.to_node,
    targetHandle: c.to_port,
  }));
  set({ nodes: flowNodes, edges: flowEdges, selectedId: null, errors: {} });
}
