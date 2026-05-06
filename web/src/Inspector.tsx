// Right panel: shows the selected node's details and editable config inputs.
// Editable ports are the ones the server marks with editable=true (primitives
// like INT/FLOAT/BOOL/STRING). Data ports show up as read-only labels.
// Edits round-trip through set_input, and the local NodeInstance is updated
// so the canvas config summary reflects the new value immediately.

import { useState } from "react";
import { useStore } from "./store";
import { CATEGORY_COLORS, theme } from "./theme";
import type { NodeInstance, PortDef } from "./types";

export function Inspector() {
  const selectedId = useStore((s) => s.selectedId);
  const node = useStore((s) => s.nodes.find((n) => n.id === selectedId));

  return (
    <div style={styles.panel}>
      <div style={styles.header}>Inspector</div>
      <div style={styles.scroll}>
        {!node ? (
          <div style={styles.hint}>Select a node to inspect.</div>
        ) : (
          <InspectorBody instance={node.data.instance} />
        )}
      </div>
    </div>
  );
}

function InspectorBody({ instance }: { instance: NodeInstance }) {
  const client = useStore((s) => s.client);
  const setNodes = useStore((s) => s.setNodes);

  const update = (portName: string, value: unknown) => {
    // Optimistic local patch first so the widget feels instant.
    setNodes((prev) =>
      prev.map((n) =>
        n.id === instance.id
          ? {
              ...n,
              data: {
                ...n.data,
                instance: {
                  ...n.data.instance,
                  inputs: {
                    ...n.data.instance.inputs,
                    [portName]: { ...n.data.instance.inputs[portName], default_value: value },
                  },
                },
              },
            }
          : n,
      ),
    );
    // Round-trip to server. set_input returns the full updated node dict so
    // we can refresh `relevant_inputs` after a kind/op/mode change — the
    // visible field set re-narrows live without a second RPC.
    client
      .call<NodeInstance>("set_input", { node_id: instance.id, port_name: portName, value })
      .then((updated) => {
        setNodes((prev) =>
          prev.map((n) =>
            n.id === instance.id
              ? { ...n, data: { ...n.data, instance: updated } }
              : n,
          ),
        );
      })
      .catch((err) => console.error("set_input failed:", err));
  };

  const cat = instance.category;
  const catColor = CATEGORY_COLORS[cat] ?? theme.textDim;

  // Mega-consolidated nodes override BaseNode.relevant_inputs to hide ports
  // that don't apply to the current kind/op/mode. When the server returns a
  // non-null list, we show ONLY those editable ports; otherwise show all.
  // Wired data ports are always shown — they represent connections, not config.
  const relevant = instance.relevant_inputs;
  const isRelevant = (name: string) =>
    relevant === null || relevant === undefined ? true : relevant.includes(name);

  const editable = Object.entries(instance.inputs).filter(
    ([name, p]) => p.editable && isRelevant(name),
  );
  const dataPorts = Object.entries(instance.inputs).filter(([, p]) => !p.editable);
  const outputs = Object.entries(instance.outputs);

  return (
    <div style={styles.body}>
      <div style={styles.title}>{instance.label}</div>
      <div style={{ color: catColor, fontSize: 12, fontWeight: 600 }}>{cat}</div>
      {instance.description && (
        <div style={styles.desc}>{instance.description}</div>
      )}

      {editable.length > 0 && (
        <>
          <SectionHeader label="Config" />
          {editable.map(([name, port]) => (
            <ConfigRow key={name} name={name} port={port} onChange={(v) => update(name, v)} />
          ))}
        </>
      )}

      {dataPorts.length > 0 && (
        <>
          <SectionHeader label="Inputs" />
          {dataPorts.map(([name, port]) => (
            <PortRow key={name} name={name} port={port} />
          ))}
        </>
      )}

      {outputs.length > 0 && (
        <>
          <SectionHeader label="Outputs" />
          {outputs.map(([name, port]) => (
            <PortRow key={name} name={name} port={port} />
          ))}
        </>
      )}
    </div>
  );
}

function SectionHeader({ label }: { label: string }) {
  return <div style={styles.section}>{label}</div>;
}

function PortRow({ name, port }: { name: string; port: PortDef }) {
  const [r, g, b] = port.color;
  const color = `rgb(${r}, ${g}, ${b})`;
  return (
    <div style={styles.portRow} title={port.description || port.port_type}>
      <span style={{ color, fontSize: 11, fontFamily: "monospace" }}>●</span>
      <span style={{ color: theme.text }}>{name}</span>
      <span style={{ color: theme.textDim, marginLeft: "auto", fontSize: 11 }}>
        {port.port_type}
      </span>
    </div>
  );
}

function ConfigRow({
  name,
  port,
  onChange,
}: {
  name: string;
  port: PortDef;
  onChange: (value: unknown) => void;
}) {
  return (
    <div style={styles.configRow}>
      <label style={styles.configLabel} title={port.description}>
        {name}
      </label>
      <div style={styles.configInput}>
        <ConfigWidget port={port} onChange={onChange} />
      </div>
    </div>
  );
}

// Typeable combobox that fetches suggestions from the server the first
// time the user focuses it. Falls back to a plain text input if the RPC
// returns nothing or errors (e.g. Ollama daemon not running).
function DynamicChoiceInput({
  port,
  onChange,
}: {
  port: PortDef;
  onChange: (v: unknown) => void;
}) {
  const client = useStore((s) => s.client);
  const v = port.default_value;
  const [items, setItems] = useState<string[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const str = v === null || v === undefined ? "" : String(v);
  const listId = `dyn-choices-${port.dynamic_choices}`;

  const loadIfNeeded = () => {
    if (items !== null || loading) return;
    setLoading(true);
    client
      .call<{ items?: Array<{ name?: string; label?: string }>; error?: string }>(
        port.dynamic_choices!,
      )
      .then((res) => {
        const names: string[] = [];
        for (const it of res.items ?? []) {
          const n = it.name ?? it.label;
          if (typeof n === "string") names.push(n);
        }
        setItems(names);
        if (res.error) setErr(res.error);
      })
      .catch((e) => {
        setItems([]);
        setErr(String(e?.message ?? e));
      })
      .finally(() => setLoading(false));
  };

  return (
    <>
      <input
        type="text"
        list={listId}
        style={styles.textInput}
        value={str}
        placeholder={loading ? "loading…" : "type or pick from list"}
        onFocus={loadIfNeeded}
        onChange={(e) => onChange(e.target.value)}
      />
      {items && items.length > 0 && (
        <datalist id={listId}>
          {items.map((n) => (
            <option key={n} value={n} />
          ))}
        </datalist>
      )}
      {err && (
        <div style={{ fontSize: 10, color: theme.textDim, marginTop: 2 }}>
          {err}
        </div>
      )}
    </>
  );
}

function ConfigWidget({ port, onChange }: { port: PortDef; onChange: (v: unknown) => void }) {
  const t = port.port_type;
  const v = port.default_value;

  if (t === "BOOL") {
    return (
      <input
        type="checkbox"
        checked={Boolean(v)}
        onChange={(e) => onChange(e.target.checked)}
      />
    );
  }
  if (t === "INT") {
    return (
      <input
        type="number"
        step={1}
        style={styles.numInput}
        value={v === null || v === undefined ? "" : Number(v)}
        onChange={(e) => onChange(parseInt(e.target.value || "0", 10))}
      />
    );
  }
  if (t === "FLOAT") {
    return (
      <input
        type="number"
        step={0.001}
        style={styles.numInput}
        value={v === null || v === undefined ? "" : Number(v)}
        onChange={(e) => onChange(parseFloat(e.target.value || "0"))}
      />
    );
  }
  if (t === "STRING" && port.choices && port.choices.length > 0) {
    return (
      <select
        style={styles.select}
        value={String(v ?? "")}
        onChange={(e) => onChange(e.target.value)}
      >
        {port.choices.map((c) => (
          <option key={c} value={c}>
            {c}
          </option>
        ))}
      </select>
    );
  }
  // Port declared a `dynamic_choices` RPC — render a typeable combobox
  // whose suggestions are fetched from the server on focus. Keeps the
  // "node is canonical" principle while giving the user discovery UX
  // (e.g. installed Ollama models) where it belongs: on the node itself.
  if (t === "STRING" && port.dynamic_choices) {
    return <DynamicChoiceInput port={port} onChange={onChange} />;
  }
  // Long / multi-line strings (prompts, code bodies, playbooks) get a
  // resizable textarea instead of a cramped single-line input. Heuristic:
  // any string with newlines, or over 60 chars, is treated as multi-line.
  const str = v === null || v === undefined ? "" : String(v);
  const isMultiline = str.includes("\n") || str.length > 60;
  if (isMultiline) {
    // Show enough rows to see ~10 lines at once; user can resize via the
    // textarea handle if they want more. Monospace for code-shaped fields.
    const looksLikeCode = /^\s*(from\s|import\s|def\s|#|return\s)/m.test(str);
    return (
      <textarea
        style={{
          ...styles.textInput,
          minHeight: 160,
          fontFamily: looksLikeCode ? "monospace" : "inherit",
          resize: "vertical",
          lineHeight: 1.4,
        }}
        value={str}
        onChange={(e) => onChange(e.target.value)}
      />
    );
  }
  return (
    <input
      type="text"
      style={styles.textInput}
      value={str}
      onChange={(e) => onChange(e.target.value)}
    />
  );
}

const styles: Record<string, React.CSSProperties> = {
  panel: {
    width: 300,
    minWidth: 300,
    background: theme.bgMid,
    borderLeft: `1px solid ${theme.border}`,
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
  scroll: {
    flex: 1,
    overflowY: "auto",
  },
  hint: {
    padding: 16,
    color: theme.textDim,
    fontSize: 13,
  },
  body: {
    padding: "12px 14px 32px",
  },
  title: {
    color: theme.textBright,
    fontSize: 16,
    fontWeight: 600,
    marginBottom: 2,
  },
  desc: {
    color: theme.textDim,
    fontSize: 12,
    marginTop: 6,
    lineHeight: 1.4,
  },
  section: {
    marginTop: 16,
    marginBottom: 6,
    color: theme.accent,
    fontSize: 11,
    fontWeight: 600,
    textTransform: "uppercase",
    letterSpacing: 0.8,
  },
  portRow: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    padding: "4px 0",
    fontSize: 12,
  },
  configRow: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    marginBottom: 6,
  },
  configLabel: {
    width: 90,
    color: theme.textDim,
    fontSize: 12,
    flexShrink: 0,
  },
  configInput: {
    flex: 1,
  },
  numInput: {
    width: "100%",
    padding: "4px 6px",
    background: theme.bgDark,
    color: theme.text,
    border: `1px solid ${theme.border}`,
    borderRadius: 3,
    fontSize: 12,
    outline: "none",
  },
  textInput: {
    width: "100%",
    padding: "4px 6px",
    background: theme.bgDark,
    color: theme.text,
    border: `1px solid ${theme.border}`,
    borderRadius: 3,
    fontSize: 12,
    outline: "none",
    fontFamily: "inherit",
  },
  select: {
    width: "100%",
    padding: "4px 6px",
    background: theme.bgDark,
    color: theme.text,
    border: `1px solid ${theme.border}`,
    borderRadius: 3,
    fontSize: 12,
    outline: "none",
  },
};
