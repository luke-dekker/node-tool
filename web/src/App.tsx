// Root component. Connects to the server on mount and composes the layout:
//   ┌─ menu bar ──────────────────────────────────────────────────────┐
//   │ Palette │        Canvas (React Flow)        │     Inspector     │
//   │         │────────────────────────────────── │                   │
//   │         │          Bottom Panel             │                   │
//   └─────────┴───────────────────────────────────┴───────────────────┘

import { useEffect } from "react";
import { BottomPanel } from "./BottomPanel";
import { Canvas } from "./Canvas";
import { FileMenu } from "./FileMenu";
import { Inspector } from "./Inspector";
import { Palette } from "./Palette";
import { useStore } from "./store";
import { theme } from "./theme";

export function App() {
  const connect = useStore((s) => s.connect);
  const conn = useStore((s) => s.conn);
  const runGraph = useStore((s) => s.runGraph);
  const clearGraph = useStore((s) => s.clearGraph);
  const exportCode = useStore((s) => s.exportCode);
  const appendLog = useStore((s) => s.appendLog);

  useEffect(() => {
    connect();
  }, [connect]);

  const onExport = async () => {
    const code = await exportCode();
    try {
      await navigator.clipboard.writeText(code);
      appendLog(`> Exported ${code.length} chars of Python to clipboard.`);
    } catch {
      appendLog(`> Exported ${code.length} chars (clipboard unavailable).`);
    }
  };

  return (
    <div style={styles.root}>
      <div style={styles.menuBar}>
        <span style={styles.title}>NodeTool</span>
        <FileMenu />
        <div style={styles.btnGroup}>
          <button style={{ ...styles.btn, ...styles.runBtn }} onClick={runGraph} disabled={conn !== "open"}>
            Run Graph
          </button>
          <button style={styles.btn} onClick={onExport} disabled={conn !== "open"}>
            Export .py
          </button>
          <button style={{ ...styles.btn, ...styles.clearBtn }} onClick={clearGraph} disabled={conn !== "open"}>
            Clear All
          </button>
        </div>
        <span style={styles.spacer} />
        <ConnBadge state={conn} />
      </div>
      <div style={styles.body}>
        <Palette />
        <div style={styles.centerCol}>
          <div style={styles.canvasArea}>
            {conn === "open" ? (
              <Canvas />
            ) : (
              <div style={styles.canvasHint}>
                {conn === "connecting"
                  ? "Connecting to ws://127.0.0.1:9800…"
                  : "Disconnected. Start the server: python server.py"}
              </div>
            )}
          </div>
          <BottomPanel />
        </div>
        <Inspector />
      </div>
    </div>
  );
}

function ConnBadge({ state }: { state: string }) {
  const color =
    state === "open" ? theme.ok : state === "connecting" ? theme.accent : theme.err;
  const label =
    state === "open" ? "Connected" : state === "connecting" ? "Connecting…" : "Disconnected";
  return <span style={{ color, fontSize: 13, marginRight: 12 }}>● {label}</span>;
}

const styles: Record<string, React.CSSProperties> = {
  root: {
    height: "100vh",
    width: "100vw",
    display: "flex",
    flexDirection: "column",
    background: theme.bgDark,
    color: theme.text,
    fontFamily:
      "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
  },
  menuBar: {
    display: "flex",
    alignItems: "center",
    padding: "0 12px",
    height: 40,
    background: theme.bgHeader,
    borderBottom: `1px solid ${theme.border}`,
    gap: 12,
    flexShrink: 0,
  },
  title: {
    color: theme.textBright,
    fontWeight: 600,
    fontSize: 14,
  },
  btnGroup: { display: "flex", gap: 6 },
  btn: {
    padding: "5px 14px",
    fontSize: 12,
    fontWeight: 500,
    background: theme.bgLight,
    color: theme.text,
    border: `1px solid ${theme.border}`,
    borderRadius: 4,
    cursor: "pointer",
    fontFamily: "inherit",
  },
  runBtn: {
    background: "#2e9c60",
    borderColor: "#2e9c60",
    color: "#fff",
  },
  clearBtn: {
    background: "#963c46",
    borderColor: "#963c46",
    color: "#fff",
  },
  spacer: { flex: 1 },
  body: {
    flex: 1,
    display: "flex",
    minHeight: 0,
  },
  centerCol: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    minWidth: 0,
  },
  canvasArea: {
    flex: 1,
    background: theme.bgDark,
    position: "relative",
    minHeight: 0,
  },
  canvasHint: {
    color: theme.textDim,
    fontSize: 13,
    position: "absolute",
    inset: 0,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
};
