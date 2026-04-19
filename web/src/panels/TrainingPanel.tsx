// Training panel — rendered entirely by the spec from the pytorch plugin.
// Layout, fields, buttons, status display, loss plot: all come from
// plugins/pytorch/_panel_training.py via get_panel_specs RPC.

import { SpecRenderer } from "./SpecRenderer";

export function TrainingPanel() {
  return <SpecRenderer label="Training" />;
}
