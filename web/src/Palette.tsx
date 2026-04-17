// Left sidebar: searchable, categorized list of available node types. Clicking
// a node will (once the canvas exists) spawn it on the graph. For now, clicks
// log a placeholder.

import { useMemo } from "react";
import { useStore } from "./store";
import { CATEGORY_COLORS, theme } from "./theme";
import type { NodeDef } from "./types";

export function Palette() {
  const registry = useStore((s) => s.registry);
  const search = useStore((s) => s.search);
  const setSearch = useStore((s) => s.setSearch);
  const spawnNode = useStore((s) => s.spawnNode);

  const filtered = useMemo(() => {
    if (!registry) return [];
    const q = search.trim().toLowerCase();
    const out: [string, NodeDef[]][] = [];
    for (const cat of registry.category_order) {
      const nodes = registry.categories[cat];
      if (!nodes) continue;
      const hits = q
        ? nodes.filter((n) => n.label.toLowerCase().includes(q) || n.type_name.includes(q))
        : nodes;
      if (hits.length) out.push([cat, hits]);
    }
    return out;
  }, [registry, search]);

  const spawn = (typeName: string) => {
    spawnNode(typeName);
  };

  return (
    <div style={styles.panel}>
      <div style={styles.header}>Nodes</div>
      <input
        style={styles.search}
        placeholder="Search nodes..."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
      />
      <div style={styles.scroll}>
        {!registry && <div style={styles.hint}>Connecting to server…</div>}
        {registry && filtered.length === 0 && (
          <div style={styles.hint}>No nodes match "{search}"</div>
        )}
        {filtered.map(([cat, nodes]) => {
          const color = CATEGORY_COLORS[cat] ?? theme.textDim;
          return (
            <div key={cat} style={styles.category}>
              <div style={{ ...styles.catHeader, borderLeft: `3px solid ${color}`, color }}>
                {cat}
              </div>
              {nodes.map((n) => (
                <button
                  key={n.type_name}
                  style={styles.nodeBtn}
                  title={n.description}
                  onClick={() => spawn(n.type_name)}
                >
                  {n.label}
                </button>
              ))}
            </div>
          );
        })}
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  panel: {
    width: 230,
    minWidth: 230,
    background: theme.bgMid,
    borderRight: `1px solid ${theme.border}`,
    display: "flex",
    flexDirection: "column",
    height: "100%",
  },
  header: {
    padding: "10px 12px",
    background: theme.bgHeader,
    color: theme.textBright,
    fontSize: 13,
    fontWeight: 600,
    borderBottom: `1px solid ${theme.border}`,
  },
  search: {
    margin: 8,
    padding: "6px 10px",
    background: theme.bgDark,
    color: theme.text,
    border: `1px solid ${theme.border}`,
    borderRadius: 4,
    fontSize: 13,
    outline: "none",
  },
  scroll: {
    flex: 1,
    overflowY: "auto",
    padding: "0 4px 8px",
  },
  hint: {
    padding: "12px",
    color: theme.textDim,
    fontSize: 12,
  },
  category: {
    marginTop: 4,
    marginBottom: 6,
  },
  catHeader: {
    padding: "4px 8px",
    background: theme.bgDark,
    fontSize: 12,
    fontWeight: 600,
    letterSpacing: 0.3,
  },
  nodeBtn: {
    display: "block",
    width: "100%",
    textAlign: "left",
    padding: "5px 10px 5px 14px",
    background: "transparent",
    color: theme.textDim,
    border: "none",
    fontSize: 13,
    cursor: "pointer",
    borderRadius: 3,
  },
};
