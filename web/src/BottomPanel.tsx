// Bottom panel: tabs for Output (terminal log), Code (exported Python),
// and every plugin-registered spec panel. Tabs stay mounted after first
// visit — inactive ones are hidden with display:none so form state (typed
// dataset paths, hyperparams) and poll timers survive tab switches.

import { useEffect, useRef, useState } from "react";
import { PANEL_BUILDERS } from "./panels";
import { useStore } from "./store";
import { theme } from "./theme";

export function BottomPanel() {
  const terminal = useStore((s) => s.terminal);
  const exportCode = useStore((s) => s.exportCode);
  const pluginPanels = useStore((s) => s.pluginPanels);
  const [code, setCode] = useState<string>("# Run the graph to generate code preview.");

  // Built-in tabs are always present; plugin tabs appended in the order
  // the server reports them, but only if we have a React builder.
  const tabs = [
    { id: "output", label: "Output" },
    { id: "code",   label: "Code"   },
    ...pluginPanels
      .filter((name) => PANEL_BUILDERS[name])
      .map((name) => ({ id: `plugin:${name}`, label: name })),
  ];

  const [tab, setTab] = useState<string>("output");
  // Track which tabs have been visited so we only pay the mount cost for
  // panels the user has actually opened.
  const [visited, setVisited] = useState<Set<string>>(new Set(["output", "code"]));
  useEffect(() => {
    setVisited((prev) => (prev.has(tab) ? prev : new Set(prev).add(tab)));
  }, [tab]);

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

  const pluginTabs = tabs.filter((t) => t.id.startsWith("plugin:"));

  const show = (id: string): React.CSSProperties =>
    tab === id ? { display: "block", height: "100%" } : { display: "none" };

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
        <div style={show("output")}>
          <div ref={outRef} style={styles.terminal}>
            {terminal.length === 0 ? (
              <span style={{ color: theme.textDim }}>
                Execute the graph or start training to see output here.
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
        </div>
        <div style={show("code")}>
          <pre style={styles.code}>{code}</pre>
        </div>
        {pluginTabs.map((t) => {
          if (!visited.has(t.id)) return null;
          const name = t.id.slice("plugin:".length);
          const Comp = PANEL_BUILDERS[name];
          if (!Comp) return null;
          return (
            <div key={t.id} style={show(t.id)}>
              <Comp />
            </div>
          );
        })}
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
    height: 240,
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
    position: "relative",
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
