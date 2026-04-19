// Optional override registry for plugin panels.
//
// Default behavior: every panel reported by the server (`get_panel_specs`)
// auto-renders via SpecRenderer in BottomPanel. No per-plugin React file
// is required — plugins must stay GUI-clean, and the React frontend must
// stay plugin-agnostic.
//
// Register an entry here ONLY when a panel needs bespoke React beyond what
// the six PanelSpec section kinds (form, dynamic_form, status, plot,
// buttons, custom) can express. Most plugins should never need this.

import type { ComponentType } from "react";

export interface PluginPanelProps {
  active?: boolean;
}

export const PANEL_BUILDERS: Record<string, ComponentType<PluginPanelProps>> = {
  // Example (do not enable without a real reason):
  //   MyPlugin: MyPluginPanel,
};
