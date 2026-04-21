// Types mirror the server payloads (server.py). Keep in sync when adding fields.

export type Color = [number, number, number, number]; // 0-255 RGBA

export interface PortDef {
  port_type: string;
  color: Color;
  description?: string;
  default_value?: unknown;
  editable?: boolean;
  choices?: string[] | null;
  // RPC name that returns {"items": [{"name": ..., "label"?: ...}, ...]}.
  // When set, the Inspector renders a dropdown populated on-demand from
  // this RPC — e.g. ollama's installed-model list for OllamaClient.model.
  dynamic_choices?: string;
}

export interface NodeDef {
  type_name: string;
  label: string;
  category: string;
  subcategory?: string;
  description: string;
  inputs: Record<string, PortDef>;
  outputs: Record<string, PortDef>;
}

export interface Registry {
  categories: Record<string, NodeDef[]>;
  category_order: string[];
}

// Full node instance returned by add_node / get_node — includes id and live
// default_value / description on every port.
export interface NodeInstance extends NodeDef {
  id: string;
}

// ── Panel spec (core/panel.py) ────────────────────────────────────────────

export type FieldType = "str" | "int" | "float" | "bool" | "choice";

export interface SpecField {
  id: string;
  type: FieldType;
  label?: string;
  default?: unknown;
  choices?: string[];
  hint?: string;
  min?: number | null;
  max?: number | null;
  step?: number | null;
}

export interface SpecAction {
  id: string;
  label: string;
  rpc: string;
  collect?: string[];
}

export type SectionKind =
  | "form"
  | "dynamic_form"
  | "status"
  | "plot"
  | "buttons"
  | "custom";

export interface BaseSection {
  id: string;
  kind: SectionKind;
  label?: string;
}

export interface FormSection extends BaseSection {
  kind: "form";
  fields: SpecField[];
}

export interface DynamicFormSection extends BaseSection {
  kind: "dynamic_form";
  source_rpc: string;
  item_label_template: string;
  fields: SpecField[];
  empty_hint?: string;
}

export interface StatusSection extends BaseSection {
  kind: "status";
  source_rpc: string;
  fields: SpecField[];
  poll_ms: number;
}

export interface PlotSection extends BaseSection {
  kind: "plot";
  source_rpc: string;
  y_label: string;
  x_label: string;
  poll_ms: number;
}

export interface ButtonsSection extends BaseSection {
  kind: "buttons";
  actions: SpecAction[];
}

export interface CustomSection extends BaseSection {
  kind: "custom";
  custom_kind: string;
  params: Record<string, unknown>;
  fields?: SpecField[];
}

export type Section =
  | FormSection
  | DynamicFormSection
  | StatusSection
  | PlotSection
  | ButtonsSection
  | CustomSection;

export interface PanelSpec {
  label: string;
  sections: Section[];
}

