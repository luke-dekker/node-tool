// Training panel — shown when the PyTorch plugin's "Training" panel is
// registered server-side. UI mirrors the DPG/Godot version: status + loss
// info on the left, dataset config in the middle, hyperparams on the right.
// Wiring to actual train_start RPC will come once we build that out;
// for now it's the same UI shell.

import { useState } from "react";
import { theme } from "../theme";

export function TrainingPanel() {
  const [epochs, setEpochs] = useState(10);
  const [lr, setLr] = useState(0.001);
  const [optim, setOptim] = useState("adam");
  const [loss, setLoss] = useState("crossentropy");
  const [device, setDevice] = useState("cpu");

  return (
    <div style={styles.root}>
      <div style={styles.col}>
        <div style={styles.statusRow}>
          <span style={{ color: theme.textDim }}>●</span>
          <span style={{ color: theme.textDim }}>Idle</span>
        </div>
        <div>Epoch 0 / 0</div>
        <div>Best loss —</div>
        <div style={styles.plotPlaceholder}>[Loss plot — coming soon]</div>
      </div>

      <div style={styles.col}>
        <div style={styles.section}>Datasets</div>
        <div style={styles.hint}>
          Load a template or add Data In (A) markers to configure datasets.
        </div>
      </div>

      <div style={styles.col}>
        <Field label="epochs">
          <input
            type="number"
            min={1}
            value={epochs}
            onChange={(e) => setEpochs(parseInt(e.target.value || "1", 10))}
            style={styles.input}
          />
        </Field>
        <Field label="lr">
          <input
            type="number"
            step={0.0001}
            value={lr}
            onChange={(e) => setLr(parseFloat(e.target.value || "0"))}
            style={styles.input}
          />
        </Field>
        <Field label="optim">
          <select value={optim} onChange={(e) => setOptim(e.target.value)} style={styles.input}>
            {["adam", "adamw", "sgd", "rmsprop"].map((o) => (
              <option key={o}>{o}</option>
            ))}
          </select>
        </Field>
        <Field label="loss">
          <select value={loss} onChange={(e) => setLoss(e.target.value)} style={styles.input}>
            {["crossentropy", "mse", "bce", "bcewithlogits", "l1"].map((o) => (
              <option key={o}>{o}</option>
            ))}
          </select>
        </Field>
        <Field label="device">
          <select value={device} onChange={(e) => setDevice(e.target.value)} style={styles.input}>
            {["cpu", "cuda", "cuda:0", "cuda:1"].map((o) => (
              <option key={o}>{o}</option>
            ))}
          </select>
        </Field>
        <div style={styles.btnRow}>
          <button style={styles.btn}>Start</button>
          <button style={styles.btn}>Pause</button>
          <button style={styles.btn}>Stop</button>
        </div>
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={styles.field}>
      <label style={styles.fieldLabel}>{label}</label>
      <div style={{ flex: 1 }}>{children}</div>
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
    gap: 4,
    minWidth: 0,
    fontSize: 12,
  },
  statusRow: { display: "flex", gap: 6, alignItems: "center" },
  plotPlaceholder: {
    flex: 1,
    marginTop: 6,
    background: theme.bgDark,
    border: `1px solid ${theme.border}`,
    borderRadius: 4,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    color: theme.textDim,
    fontSize: 11,
  },
  section: { color: theme.accent, fontWeight: 600, fontSize: 12 },
  hint: { color: theme.textDim, fontSize: 11, marginTop: 4 },
  field: { display: "flex", alignItems: "center", gap: 8 },
  fieldLabel: { width: 56, color: theme.textDim },
  input: {
    width: "100%",
    padding: "3px 6px",
    background: theme.bgDark,
    color: theme.text,
    border: `1px solid ${theme.border}`,
    borderRadius: 3,
    fontSize: 12,
    outline: "none",
    fontFamily: "inherit",
  },
  btnRow: { display: "flex", gap: 6, marginTop: 4 },
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
};
