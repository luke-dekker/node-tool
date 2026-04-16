// Robotics panel — registered by the robotics plugin. Same UI shell as the
// DPG/Godot version: serial monitor on the left, sensor plot placeholder on
// the right. No hardware wiring yet (mirrors what the other frontends ship).

import { useState } from "react";
import { theme } from "../theme";

export function RoboticsPanel() {
  const [port, setPort] = useState("");
  const [baud, setBaud] = useState("115200");
  const [log, setLog] = useState<string[]>(["(serial output will appear here)"]);
  const [send, setSend] = useState("");

  const sendCmd = () => {
    if (!send) return;
    setLog((prev) => [...prev, `> ${send}`].slice(-200));
    setSend("");
  };

  return (
    <div style={styles.root}>
      <div style={styles.col}>
        <div style={styles.section}>Serial Monitor</div>

        <div style={styles.row}>
          <label style={styles.lbl}>Port</label>
          <select value={port} onChange={(e) => setPort(e.target.value)} style={styles.input}>
            <option value="">(none)</option>
          </select>
          <select value={baud} onChange={(e) => setBaud(e.target.value)} style={styles.smInput}>
            {["9600", "19200", "38400", "57600", "115200"].map((b) => (
              <option key={b}>{b}</option>
            ))}
          </select>
        </div>

        <div style={styles.btnRow}>
          <button style={styles.btn}>Connect</button>
          <button style={styles.btn}>Disconnect</button>
          <button style={styles.btn}>Refresh</button>
        </div>

        <div style={styles.terminal}>
          {log.map((l, i) => (
            <div key={i}>{l}</div>
          ))}
        </div>

        <div style={styles.row}>
          <input
            type="text"
            placeholder="Send command..."
            value={send}
            onChange={(e) => setSend(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") sendCmd();
            }}
            style={styles.input}
          />
          <button style={styles.btn} onClick={sendCmd}>
            Send
          </button>
        </div>
      </div>

      <div style={styles.col}>
        <div style={styles.section}>Live Sensor Plot</div>
        <div style={styles.hint}>Wire sensor nodes to see live data here.</div>
        <div style={styles.plotPlaceholder} />
      </div>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  root: {
    display: "flex",
    height: "100%",
    padding: "8px 12px",
    gap: 12,
  },
  col: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    gap: 6,
    minWidth: 0,
    fontSize: 12,
  },
  section: { color: theme.accent, fontWeight: 600, fontSize: 12 },
  hint: { color: theme.textDim, fontSize: 11 },
  row: { display: "flex", gap: 6, alignItems: "center" },
  btnRow: { display: "flex", gap: 6 },
  lbl: { color: theme.textDim, width: 40, fontSize: 12 },
  input: {
    flex: 1,
    padding: "3px 6px",
    background: theme.bgDark,
    color: theme.text,
    border: `1px solid ${theme.border}`,
    borderRadius: 3,
    fontSize: 12,
    outline: "none",
    fontFamily: "inherit",
    minWidth: 0,
  },
  smInput: {
    width: 90,
    padding: "3px 6px",
    background: theme.bgDark,
    color: theme.text,
    border: `1px solid ${theme.border}`,
    borderRadius: 3,
    fontSize: 12,
    outline: "none",
    fontFamily: "inherit",
  },
  btn: {
    padding: "4px 10px",
    background: theme.bgLight,
    color: theme.text,
    border: `1px solid ${theme.border}`,
    borderRadius: 3,
    fontSize: 12,
    cursor: "pointer",
    fontFamily: "inherit",
  },
  terminal: {
    flex: 1,
    overflowY: "auto",
    padding: "6px 8px",
    background: theme.bgDark,
    border: `1px solid ${theme.border}`,
    borderRadius: 3,
    fontFamily: "'Consolas', 'Monaco', monospace",
    fontSize: 11,
    color: theme.text,
    minHeight: 60,
  },
  plotPlaceholder: {
    flex: 1,
    background: theme.bgDark,
    border: `1px solid ${theme.border}`,
    borderRadius: 4,
  },
};
