// Custom React Flow node. Uses natural flow layout: a header row, optional
// config summary, then a port grid (inputs on the left, outputs on the right).
// Each port row has the Handle absolutely positioned to the row's vertical
// midpoint so the connection wires line up exactly with the label they're
// next to.

import { Handle, Position, type NodeProps } from "reactflow";
import { memo } from "react";
import type { NodeInstance, PortDef } from "./types";
import { CATEGORY_COLORS, theme } from "./theme";
import { useStore } from "./store";

function rgbaToCss(c: [number, number, number, number]): string {
  return `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${c[3] / 255})`;
}

function dataInputs(instance: NodeInstance): [string, PortDef][] {
  // Editable ports = config (live in inspector, no handle on the node).
  return Object.entries(instance.inputs).filter(([, p]) => !p.editable);
}

function configSummary(instance: NodeInstance): string {
  const parts: string[] = [];
  for (const [, p] of Object.entries(instance.inputs)) {
    if (!p.editable) continue;
    const v = p.default_value;
    if (v === null || v === undefined) continue;
    const s = String(v);
    if (!s || s === "false" || s === "False" || s === "0" || s === "0.0") continue;
    parts.push(s);
    if (parts.length >= 4) break;
  }
  return parts.join(", ");
}

function GraphNodeComponent({ id, data, selected }: NodeProps<{ instance: NodeInstance }>) {
  const instance = data.instance;
  const error = useStore((s) => s.errors[id]);

  const headerColor = CATEGORY_COLORS[instance.category] ?? theme.accentDim;
  const inputs = dataInputs(instance);
  const outputs = Object.entries(instance.outputs);
  const summary = configSummary(instance);

  // We need the same number of rows on both sides to keep handles aligned.
  const rowCount = Math.max(inputs.length, outputs.length);

  const borderColor = error
    ? theme.err
    : selected
      ? theme.accent
      : "rgba(52, 62, 84, 0.85)";

  return (
    <div
      style={{
        ...styles.root,
        borderColor,
        borderWidth: selected || error ? 2 : 1,
      }}
    >
      <div style={{ ...styles.header, background: headerColor }}>{instance.label}</div>

      {summary && <div style={styles.summary}>{summary}</div>}

      {rowCount > 0 && (
        <div style={styles.portGrid}>
          {Array.from({ length: rowCount }).map((_, i) => {
            const inPort = inputs[i];
            const outPort = outputs[i];
            return (
              <div key={i} style={styles.portRow}>
                <div style={styles.portCell}>
                  {inPort && (
                    <>
                      <Handle
                        id={inPort[0]}
                        type="target"
                        position={Position.Left}
                        style={{
                          ...styles.handle,
                          background: rgbaToCss(inPort[1].color),
                          left: -6,
                        }}
                      />
                      <span
                        style={{ ...styles.portLabel, color: rgbaToCss(inPort[1].color) }}
                      >
                        {inPort[0]}
                      </span>
                    </>
                  )}
                </div>
                <div style={{ ...styles.portCell, justifyContent: "flex-end" }}>
                  {outPort && (
                    <>
                      <span
                        style={{
                          ...styles.portLabel,
                          color: rgbaToCss(outPort[1].color),
                          textAlign: "right",
                        }}
                      >
                        {outPort[0]}
                      </span>
                      <Handle
                        id={outPort[0]}
                        type="source"
                        position={Position.Right}
                        style={{
                          ...styles.handle,
                          background: rgbaToCss(outPort[1].color),
                          right: -6,
                        }}
                      />
                    </>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {error && <div style={styles.error}>{error.message}</div>}
    </div>
  );
}

export const GraphNode = memo(GraphNodeComponent);

const styles: Record<string, React.CSSProperties> = {
  root: {
    minWidth: 200,
    background: "rgba(24, 28, 40, 0.97)",
    borderStyle: "solid",
    borderRadius: 8,
    fontSize: 12,
    color: theme.text,
    boxShadow: "0 4px 14px rgba(0,0,0,0.35)",
    overflow: "visible", // handles stick out by 6px on each side
  },
  header: {
    padding: "6px 12px",
    color: "#fff",
    fontWeight: 600,
    fontSize: 12,
    letterSpacing: 0.2,
    borderTopLeftRadius: 7,
    borderTopRightRadius: 7,
  },
  summary: {
    padding: "4px 12px 2px",
    color: theme.textDim,
    fontSize: 11,
    fontStyle: "italic",
    borderBottom: "1px solid rgba(60, 70, 92, 0.3)",
    marginBottom: 2,
  },
  portGrid: {
    padding: "6px 0",
    display: "flex",
    flexDirection: "column",
    gap: 4,
  },
  portRow: {
    display: "flex",
    alignItems: "center",
    minHeight: 20,
  },
  portCell: {
    position: "relative",
    flex: 1,
    display: "flex",
    alignItems: "center",
    minWidth: 0,
    padding: "0 12px",
  },
  handle: {
    width: 11,
    height: 11,
    border: "1.5px solid rgba(18, 20, 28, 0.9)",
  },
  portLabel: {
    fontSize: 11,
    fontWeight: 500,
    whiteSpace: "nowrap",
    lineHeight: 1.4,
  },
  error: {
    margin: "2px 0 0",
    padding: "4px 12px",
    background: "rgba(235, 90, 90, 0.12)",
    color: theme.err,
    fontSize: 11,
    borderTop: "1px solid rgba(235, 90, 90, 0.3)",
    borderBottomLeftRadius: 7,
    borderBottomRightRadius: 7,
    maxWidth: 240,
    whiteSpace: "normal",
    wordBreak: "break-word",
  },
};
