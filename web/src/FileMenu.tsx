// File menu — dropdown with New, Open, Save, Templates, Export. Templates
// expands inline since web menus don't have nested submenu chrome by default.

import { useEffect, useRef, useState } from "react";
import { useStore } from "./store";
import { theme } from "./theme";

export function FileMenu() {
  const [open, setOpen] = useState(false);
  const [showTemplates, setShowTemplates] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const templates = useStore((s) => s.templates);
  const clearGraph = useStore((s) => s.clearGraph);
  const saveGraph = useStore((s) => s.saveGraph);
  const loadGraph = useStore((s) => s.loadGraph);
  const loadTemplate = useStore((s) => s.loadTemplate);
  const exportCode = useStore((s) => s.exportCode);
  const appendLog = useStore((s) => s.appendLog);

  // Close the menu when clicking outside.
  useEffect(() => {
    const onDocClick = (e: MouseEvent) => {
      if (open && ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false);
        setShowTemplates(false);
      }
    };
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [open]);

  const close = () => {
    setOpen(false);
    setShowTemplates(false);
  };

  const onNew = () => {
    clearGraph();
    close();
  };
  const onOpen = () => {
    fileInputRef.current?.click();
    close();
  };
  const onSave = () => {
    saveGraph();
    close();
  };
  const onExport = async () => {
    const code = await exportCode();
    const blob = new Blob([code], { type: "text/x-python" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "graph.py";
    a.click();
    URL.revokeObjectURL(url);
    appendLog(`> Exported ${code.length} chars to graph.py`);
    close();
  };

  return (
    <div ref={ref} style={styles.wrapper}>
      <button style={styles.menuBtn} onClick={() => setOpen((v) => !v)}>
        File
      </button>
      <input
        ref={fileInputRef}
        type="file"
        accept=".json,application/json"
        style={{ display: "none" }}
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) loadGraph(f);
          e.target.value = "";
        }}
      />
      {open && (
        <div style={styles.dropdown}>
          <MenuItem onClick={onNew}>New</MenuItem>
          <MenuItem onClick={onOpen}>Open Graph...</MenuItem>
          <MenuItem onClick={onSave}>Save Graph</MenuItem>
          <Separator />
          <MenuItem
            onClick={() => setShowTemplates((v) => !v)}
            chevron={showTemplates ? "▾" : "▸"}
          >
            Templates
          </MenuItem>
          {showTemplates && (
            <div style={styles.submenu}>
              {templates.length === 0 && (
                <div style={styles.emptyHint}>(no templates)</div>
              )}
              {templates.map((t) => (
                <MenuItem
                  key={t.label}
                  onClick={() => {
                    loadTemplate(t.label);
                    close();
                  }}
                  title={t.description}
                >
                  {t.label}
                </MenuItem>
              ))}
            </div>
          )}
          <Separator />
          <MenuItem onClick={onExport}>Export .py...</MenuItem>
        </div>
      )}
    </div>
  );
}

function MenuItem({
  children,
  onClick,
  chevron,
  title,
}: {
  children: React.ReactNode;
  onClick: () => void;
  chevron?: string;
  title?: string;
}) {
  return (
    <button style={styles.item} onClick={onClick} title={title}>
      <span>{children}</span>
      {chevron && <span style={styles.chevron}>{chevron}</span>}
    </button>
  );
}

function Separator() {
  return <div style={styles.separator} />;
}

const styles: Record<string, React.CSSProperties> = {
  wrapper: { position: "relative" },
  menuBtn: {
    padding: "6px 12px",
    background: "transparent",
    color: theme.text,
    border: "none",
    borderRadius: 4,
    fontSize: 13,
    cursor: "pointer",
    fontFamily: "inherit",
  },
  dropdown: {
    position: "absolute",
    top: "100%",
    left: 0,
    minWidth: 180,
    background: "#10121a",
    border: `1px solid ${theme.border}`,
    borderRadius: 4,
    boxShadow: "0 8px 24px rgba(0,0,0,0.5)",
    padding: "4px 0",
    zIndex: 1000,
  },
  submenu: {
    background: theme.bgDark,
    borderTop: `1px solid ${theme.border}`,
    borderBottom: `1px solid ${theme.border}`,
    maxHeight: 280,
    overflowY: "auto",
  },
  item: {
    display: "flex",
    width: "100%",
    padding: "6px 14px",
    background: "transparent",
    color: theme.text,
    border: "none",
    fontSize: 12,
    textAlign: "left",
    cursor: "pointer",
    fontFamily: "inherit",
    alignItems: "center",
    justifyContent: "space-between",
    gap: 12,
  },
  chevron: { color: theme.textDim, fontSize: 11 },
  separator: { height: 1, background: theme.border, margin: "4px 0" },
  emptyHint: { padding: "6px 14px", color: theme.textDim, fontSize: 11 },
};
