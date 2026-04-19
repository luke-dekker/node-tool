// Generic renderer for PanelSpec (core/panel.py). Every plugin's side
// panel is described as a spec; this file renders it natively in React.
// Zero plugin-specific code — add a new plugin with a new spec and it
// shows up here without edits, same as DPG (gui/panel_renderer.py).
//
// Section kinds: form / dynamic_form / status / plot / buttons / custom.
// Custom kinds live in the `customKinds` registry at the bottom.

import { useEffect, useMemo, useRef, useState } from "react";
import { theme } from "../theme";
import { useStore } from "../store";
import type {
  PanelSpec, Section, SpecField, SpecAction,
  FormSection, DynamicFormSection, StatusSection,
  ButtonsSection, CustomSection,
} from "../types";

type Dispatcher = (method: string, params?: Record<string, unknown>) => Promise<any>;

// Keyed store for every field's current value. Static forms:
//   formValues[section.id][field.id] = value
// Dynamic forms:
//   dynValues[section.id][itemKey][field.id] = value
type FormValues = Record<string, Record<string, unknown>>;
type DynValues  = Record<string, Record<string, Record<string, unknown>>>;

export function SpecRenderer({ label }: { label: string }) {
  const spec = useStore((s) => s.panelSpecs[label]) as PanelSpec | undefined;
  const client = useStore((s) => s.client);
  const appendLog = useStore((s) => s.appendLog);

  const dispatch: Dispatcher = useMemo(
    () => (method, params = {}) => client.call(method, params),
    [client],
  );

  const [formValues, setFormValues] = useState<FormValues>({});
  const [dynValues, setDynValues]   = useState<DynValues>({});

  // Seed defaults from the spec on first load / spec change.
  useEffect(() => {
    if (!spec) return;
    const next: FormValues = {};
    for (const sec of spec.sections) {
      if (sec.kind === "form") {
        next[sec.id] = {};
        for (const f of sec.fields) next[sec.id][f.id] = f.default ?? null;
      }
    }
    setFormValues(next);
  }, [spec]);

  if (!spec) {
    return (
      <div style={styles.loading}>
        Loading <b>{label}</b> panel spec...
      </div>
    );
  }

  const setFormValue = (sectionId: string, fieldId: string, value: unknown) => {
    setFormValues((v) => ({
      ...v,
      [sectionId]: { ...(v[sectionId] ?? {}), [fieldId]: value },
    }));
  };
  const setDynValue = (
    sectionId: string,
    itemKey: string,
    fieldId: string,
    value: unknown,
  ) => {
    setDynValues((v) => ({
      ...v,
      [sectionId]: {
        ...(v[sectionId] ?? {}),
        [itemKey]: { ...((v[sectionId] ?? {})[itemKey] ?? {}), [fieldId]: value },
      },
    }));
  };

  // Gather params referenced by an action's `collect` list.
  const collect = (sectionIds: string[]): Record<string, unknown> => {
    const out: Record<string, unknown> = {};
    for (const sid of sectionIds) {
      const sec = spec.sections.find((s) => s.id === sid);
      if (!sec) continue;
      if (sec.kind === "dynamic_form") {
        out[sec.id] = dynValues[sec.id] ?? {};
      } else if (sec.kind === "form") {
        Object.assign(out, formValues[sec.id] ?? {});
      }
    }
    return out;
  };

  const onAction = async (action: SpecAction) => {
    try {
      const params = collect(action.collect ?? []);
      const result = await dispatch(action.rpc, params);
      if (result && typeof result === "object" && "error" in result && result.error) {
        appendLog(`[${label}] ${action.rpc}: ${result.error}`);
      }
    } catch (err) {
      appendLog(`[${label}] ${action.rpc} failed: ${String(err)}`);
    }
  };

  return (
    <div style={styles.root}>
      {spec.sections.map((sec) => (
        <SectionBlock
          key={sec.id}
          section={sec}
          dispatch={dispatch}
          formValues={formValues}
          dynValues={dynValues}
          setFormValue={setFormValue}
          setDynValue={setDynValue}
          onAction={onAction}
        />
      ))}
    </div>
  );
}

// ── Section dispatcher ─────────────────────────────────────────────────────

interface SectionProps {
  section: Section;
  dispatch: Dispatcher;
  formValues: FormValues;
  dynValues: DynValues;
  setFormValue: (sectionId: string, fieldId: string, value: unknown) => void;
  setDynValue: (sectionId: string, itemKey: string, fieldId: string, value: unknown) => void;
  onAction: (action: SpecAction) => void;
}

function SectionBlock(props: SectionProps) {
  const { section } = props;
  switch (section.kind) {
    case "form":         return <FormSectionView    {...props} section={section} />;
    case "dynamic_form": return <DynFormSectionView {...props} section={section} />;
    case "status":       return <StatusSectionView  {...props} section={section} />;
    case "plot":         return <PlotPlaceholder    section={section} />;
    case "buttons":      return <ButtonsSectionView {...props} section={section} />;
    case "custom":       return <CustomSectionView  {...props} section={section} />;
  }
}

// ── Form ──────────────────────────────────────────────────────────────────

function FormSectionView({
  section, formValues, setFormValue,
}: SectionProps & { section: FormSection }) {
  return (
    <div style={styles.section}>
      {section.label && <div style={styles.sectionLabel}>{section.label}</div>}
      {section.fields.map((f) => (
        <FieldInput
          key={f.id}
          field={f}
          value={formValues[section.id]?.[f.id] ?? f.default ?? null}
          onChange={(v) => setFormValue(section.id, f.id, v)}
        />
      ))}
    </div>
  );
}

// ── Dynamic form ─────────────────────────────────────────────────────────

function DynFormSectionView({
  section, dispatch, dynValues, setDynValue,
}: SectionProps & { section: DynamicFormSection }) {
  const [items, setItems] = useState<Record<string, Record<string, any>>>({});

  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const resp: any = await dispatch(section.source_rpc, {});
        const groups = resp?.groups ?? {};
        if (!alive) return;
        setItems(groups);
      } catch {
        // ignore — retry on next poll
      }
    };
    poll();
    const handle = window.setInterval(poll, 500);
    return () => {
      alive = false;
      window.clearInterval(handle);
    };
  }, [dispatch, section.source_rpc]);

  const keys = Object.keys(items).sort();
  if (keys.length === 0) {
    return (
      <div style={styles.section}>
        {section.label && <div style={styles.sectionLabel}>{section.label}</div>}
        <div style={styles.hint}>
          {section.empty_hint ?? "Nothing to configure yet."}
        </div>
      </div>
    );
  }

  return (
    <div style={styles.section}>
      {section.label && <div style={styles.sectionLabel}>{section.label}</div>}
      {keys.map((key) => {
        const meta = items[key] ?? {};
        let title = section.item_label_template;
        try {
          title = section.item_label_template
            .replace("{key}", key)
            .replace(
              /\{(\w+)\}/g,
              (_, k) => (meta[k] !== undefined ? String(meta[k]) : ""),
            );
        } catch {
          title = key;
        }
        return (
          <div key={key} style={styles.dynItem}>
            <div style={styles.dynItemTitle}>{title}</div>
            {section.fields.map((f) => {
              const v = dynValues[section.id]?.[key]?.[f.id] ?? f.default ?? null;
              return (
                <FieldInput
                  key={f.id}
                  field={f}
                  value={v}
                  onChange={(val) => setDynValue(section.id, key, f.id, val)}
                />
              );
            })}
          </div>
        );
      })}
    </div>
  );
}

// ── Status ───────────────────────────────────────────────────────────────

function StatusSectionView({
  section, dispatch,
}: SectionProps & { section: StatusSection }) {
  const [data, setData] = useState<Record<string, any>>({});

  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const resp: any = await dispatch(section.source_rpc, {});
        if (!alive) return;
        setData(resp ?? {});
      } catch {
        // ignore
      }
    };
    poll();
    const handle = window.setInterval(poll, section.poll_ms || 500);
    return () => {
      alive = false;
      window.clearInterval(handle);
    };
  }, [dispatch, section.source_rpc, section.poll_ms]);

  const statusColor = (fieldId: string, value: string): string => {
    if (fieldId !== "status") return theme.text;
    const v = value.toLowerCase();
    if (v === "running") return theme.ok;
    if (v === "error")   return theme.err;
    if (v === "paused")  return "#e0b050";
    if (v === "done")    return theme.accent;
    return theme.textDim;
  };

  return (
    <div style={styles.section}>
      {section.label && <div style={styles.sectionLabel}>{section.label}</div>}
      {section.fields.map((f) => {
        const v = data[f.id];
        const display = v === undefined || v === "" ? "—" : String(v);
        return (
          <div key={f.id} style={styles.statusRow}>
            {f.label && <span style={styles.statusKey}>{f.label}:</span>}
            <span style={{ color: statusColor(f.id, display) }}>{display}</span>
          </div>
        );
      })}
    </div>
  );
}

// ── Buttons ──────────────────────────────────────────────────────────────

function ButtonsSectionView({
  section, onAction,
}: SectionProps & { section: ButtonsSection }) {
  return (
    <div style={styles.section}>
      <div style={styles.btnRow}>
        {section.actions.map((a) => (
          <button key={a.id} style={styles.btn} onClick={() => onAction(a)}>
            {a.label}
          </button>
        ))}
      </div>
    </div>
  );
}

// ── Custom section dispatcher ────────────────────────────────────────────

const customKinds: Record<
  string,
  (props: SectionProps & { section: CustomSection }) => JSX.Element
> = {
  loss_plot: LossPlot,
  log_tail:  LogTail,
};

function LogTail({
  section, dispatch,
}: SectionProps & { section: CustomSection }) {
  const srcRpc = String(section.params?.source_rpc ?? "");
  const pollMs = Number(section.params?.poll_ms ?? 500);
  const [lines, setLines] = useState<string[]>([]);

  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const resp: any = await dispatch(srcRpc, {});
        if (!alive) return;
        setLines(resp?.lines ?? []);
      } catch { /* ignore */ }
    };
    poll();
    const handle = window.setInterval(poll, pollMs);
    return () => {
      alive = false;
      window.clearInterval(handle);
    };
  }, [dispatch, srcRpc, pollMs]);

  return (
    <div style={styles.section}>
      {section.label && <div style={styles.sectionLabel}>{section.label}</div>}
      <div style={styles.logTail}>
        {lines.length === 0
          ? <span style={{ color: theme.textDim }}>(no output yet)</span>
          : lines.map((l, i) => <div key={i}>{l}</div>)}
      </div>
    </div>
  );
}

function CustomSectionView(props: SectionProps & { section: CustomSection }) {
  const Renderer = customKinds[props.section.custom_kind];
  if (!Renderer) {
    return (
      <div style={styles.section}>
        <div style={styles.hint}>
          [no renderer for custom kind '{props.section.custom_kind}']
        </div>
      </div>
    );
  }
  return <Renderer {...props} />;
}

// ── LossPlot — a SVG line chart driven by source_rpc ─────────────────────

function LossPlot({
  section, dispatch,
}: SectionProps & { section: CustomSection }) {
  const srcRpc = String(section.params?.source_rpc ?? "get_training_losses");
  const pollMs = Number(section.params?.poll_ms ?? 500);
  const seriesNames = (section.params?.series as string[] | undefined) ?? ["train", "val"];
  const [series, setSeries] = useState<Record<string, number[]>>({});

  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const resp: any = await dispatch(srcRpc, {});
        if (!alive) return;
        setSeries(resp?.series ?? {});
      } catch { /* ignore */ }
    };
    poll();
    const handle = window.setInterval(poll, pollMs);
    return () => {
      alive = false;
      window.clearInterval(handle);
    };
  }, [dispatch, srcRpc, pollMs]);

  const colors = [theme.accent, theme.ok];
  const W = 240, H = 100;
  let allValues: number[] = [];
  for (const n of seriesNames) allValues = allValues.concat(series[n] ?? []);
  const maxLen = Math.max(
    1,
    ...seriesNames.map((n) => (series[n] ?? []).length),
  );
  const yMin = Math.min(...allValues, 0);
  const yMax = Math.max(...allValues, 1);
  const toX = (i: number) => (i / Math.max(1, maxLen - 1)) * (W - 6) + 3;
  const toY = (v: number) => H - 3 - ((v - yMin) / Math.max(1e-9, yMax - yMin)) * (H - 6);

  return (
    <div style={styles.section}>
      {section.label && <div style={styles.sectionLabel}>{section.label}</div>}
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={styles.plot}>
        <rect x={0} y={0} width={W} height={H} fill={theme.bgDark} />
        {seriesNames.map((n, idx) => {
          const ys = series[n] ?? [];
          if (ys.length < 2) return null;
          const d = ys
            .map((v, i) => `${i === 0 ? "M" : "L"}${toX(i)},${toY(v)}`)
            .join(" ");
          return (
            <path key={n} d={d} stroke={colors[idx % colors.length]}
                  strokeWidth={1.5} fill="none" />
          );
        })}
      </svg>
      <div style={styles.legend}>
        {seriesNames.map((n, idx) => (
          <span key={n} style={{ marginRight: 10, color: colors[idx % colors.length] }}>
            ● {n}
          </span>
        ))}
      </div>
    </div>
  );
}

// Placeholder for the bare "plot" kind (not used by current specs).
function PlotPlaceholder({ section }: { section: Section }) {
  return (
    <div style={styles.section}>
      <div style={styles.hint}>[plot '{section.id}']</div>
    </div>
  );
}

// ── Field input ──────────────────────────────────────────────────────────

function FieldInput({
  field, value, onChange,
}: {
  field: SpecField;
  value: unknown;
  onChange: (v: unknown) => void;
}) {
  const label = field.label ?? field.id;
  const v: any = value;

  let input: React.ReactNode;
  if (field.type === "choice" || (field.choices && field.choices.length)) {
    input = (
      <select
        style={styles.input}
        value={String(v ?? "")}
        onChange={(e) => onChange(e.target.value)}
      >
        {(field.choices ?? []).map((c) => (
          <option key={c}>{c}</option>
        ))}
      </select>
    );
  } else if (field.type === "int") {
    input = (
      <input
        type="number"
        step={1}
        min={field.min ?? undefined}
        max={field.max ?? undefined}
        style={styles.input}
        value={typeof v === "number" ? v : parseInt(String(v ?? "0"), 10) || 0}
        onChange={(e) => onChange(parseInt(e.target.value || "0", 10))}
      />
    );
  } else if (field.type === "float") {
    input = (
      <input
        type="number"
        step={field.step ?? 0.0001}
        style={styles.input}
        value={typeof v === "number" ? v : parseFloat(String(v ?? "0")) || 0}
        onChange={(e) => onChange(parseFloat(e.target.value || "0"))}
      />
    );
  } else if (field.type === "bool") {
    input = (
      <input
        type="checkbox"
        checked={Boolean(v)}
        onChange={(e) => onChange(e.target.checked)}
      />
    );
  } else {
    input = (
      <input
        type="text"
        placeholder={field.hint ?? ""}
        style={styles.input}
        value={v ?? ""}
        onChange={(e) => onChange(e.target.value)}
      />
    );
  }

  return (
    <div style={styles.field}>
      <label style={styles.fieldLabel}>{label}</label>
      <div style={{ flex: 1 }}>{input}</div>
    </div>
  );
}

// ── Styles ───────────────────────────────────────────────────────────────

const styles: Record<string, React.CSSProperties> = {
  root: {
    height: "100%",
    padding: "8px 12px",
    overflowY: "auto",
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
    gap: 12,
    alignContent: "start",
  },
  loading: { padding: 12, color: theme.textDim, fontSize: 12 },
  section: { display: "flex", flexDirection: "column", gap: 4, fontSize: 12, minWidth: 0 },
  sectionLabel: {
    color: theme.accent,
    fontWeight: 600,
    fontSize: 12,
    marginBottom: 2,
  },
  hint: { color: theme.textDim, fontSize: 11, marginTop: 4 },
  dynItem: {
    display: "flex", flexDirection: "column", gap: 3,
    padding: "4px 0",
    borderTop: `1px dashed ${theme.border}`,
  },
  dynItemTitle: { color: theme.text, fontSize: 11, fontWeight: 500, margin: "2px 0" },
  statusRow: { display: "flex", gap: 6, alignItems: "baseline" },
  statusKey: { color: theme.textDim, minWidth: 70 },
  field: { display: "flex", alignItems: "center", gap: 8 },
  fieldLabel: { width: 64, color: theme.textDim, fontSize: 11 },
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
  btnRow: { display: "flex", gap: 6, marginTop: 4, flexWrap: "wrap" },
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
  plot: {
    background: theme.bgDark,
    border: `1px solid ${theme.border}`,
    borderRadius: 4,
    height: 100,
  },
  legend: { marginTop: 2, fontSize: 10, color: theme.textDim },
  logTail: {
    background: theme.bgDark,
    border: `1px solid ${theme.border}`,
    borderRadius: 3,
    padding: 6,
    fontFamily: "'Consolas', 'Monaco', monospace",
    fontSize: 11,
    maxHeight: 120,
    overflowY: "auto",
    whiteSpace: "pre-wrap",
  },
};
