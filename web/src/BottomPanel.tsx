// Bottom panel: tabs for Output (terminal log), Code (exported Python), and
// hooks for plugin-registered tabs (Training, Robotics) once we port them
// over from the Godot frontend.

import { useEffect, useRef, useState } from "react";
import { PANEL_BUILDERS } from "./panels";
import { useStore } from "./store";
import { theme } from "./theme";

export function BottomPanel() {
  const terminal = useStore((s) => s.terminal);
  const exportCode = useStore((s) => s.exportCode);
  const pluginPanels = useStore((s) => s.pluginPanels);
  const [code, setCode] = useState<string>("# Run the graph to generate code preview.");

  // Built-in tabs are always present; plugin tabs are appended in the order
  // the server reports them, but only if we have a React builder for them.
  const tabs = [
    { id: "output", label: "Output" },
    { id: "code", label: "Code" },
    ...pluginPanels
      .filter((name) => PANEL_BUILDERS[name])
      .map((name) => ({ id: `plugin:${name}`, label: name })),
  ];

  const [tab, setTab] = useState<string>("output");

  // Re-export whenever we switch to the Code tab so it reflects the current graph.
  useEffect(() => {
    if (tab === "code") {
      exportCode().then(setCode).catch(console.error);
    }
  }, [tab, exportCode]);

  const outRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (tab === "output" && outRef.current) {
      outRef.current.scrollTop = outRef.current.scrollHeight;
    }
  }, [terminal, tab]);

  // Resolve a plugin tab id (`plugin:Training`) to its builder component.
  let pluginPanel: React.ReactNode = null;
  if (tab.startsWith("plugin:")) {
    const name = tab.slice("plugin:".length);
    const Comp = PANEL_BUILDERS[name];
    if (Comp) pluginPanel = <Comp />;
  }

  return (
    <div style={styles.panel}>
      <div style={styles.tabBar}>
        {tabs.map((t) => (
          <TabButton key={t.id} active={tab === t.id} onClick={() => setTab(t.id)}>
            {t.label}
          </TabButton>
        ))}
      </div>
      <div style={styles.body}>
        {tab === "output" && (
          <div ref={outRef} style={styles.terminal}>
            {terminal.length === 0 ? (
              <span style={{ color: theme.textDim }}>
                Execute the graph to see output here.
              </span>
            ) : (
              terminal.map((line, i) => (
                <div
                  key={i}
                  style={{
                    color: line.startsWith("[ERROR]") || line.startsWith("[FATAL]")
                      ? theme.err
                      : theme.text,
                  }}
                >
                  {line}
                </div>
              ))
            )}
          </div>
        )}
        {tab === "code" && <pre style={styles.code}>{code}</pre>}
        {pluginPanel}
      </div>
    </div>
  );
}

function TabButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      style={{
        ...styles.tab,
        color: active ? theme.textBright : theme.textDim,
        borderBottom: active ? `2px solid ${theme.accent}` : "2px solid transparent",
        background: active ? theme.bgLight : "transparent",
      }}
      onClick={onClick}
    >
      {children}
    </button>
  );
}

const styles: Record<string, React.CSSProperties> = {
  panel: {
    height: 220,
    background: theme.bgMid,
    borderTop: `1px solid ${theme.border}`,
    display: "flex",
    flexDirection: "column",
  },
  tabBar: {
    display: "flex",
    background: theme.bgDark,
    borderBottom: `1px solid ${theme.border}`,
    flexShrink: 0,
  },
  tab: {
    padding: "6px 16px",
    border: "none",
    fontSize: 12,
    fontWeight: 500,
    cursor: "pointer",
    fontFamily: "inherit",
  },
  body: {
    flex: 1,
    minHeight: 0,
  },
  terminal: {
    height: "100%",
    overflowY: "auto",
    padding: "8px 12px",
    fontFamily: "'Consolas', 'Monaco', monospace",
    fontSize: 12,
    lineHeight: 1.5,
    whiteSpace: "pre-wrap",
  },
  code: {
    margin: 0,
    height: "100%",
    overflow: "auto",
    padding: "8px 12px",
    fontFamily: "'Consolas', 'Monaco', monospace",
    fontSize: 12,
    color: theme.text,
    background: theme.bgDark,
  },
};
