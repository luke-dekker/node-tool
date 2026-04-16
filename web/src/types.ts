// Types mirror the server payloads (server.py). Keep in sync when adding fields.

export type Color = [number, number, number, number]; // 0-255 RGBA

export interface PortDef {
  port_type: string;
  color: Color;
  description?: string;
  default_value?: unknown;
  editable?: boolean;
  choices?: string[] | null;
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

