// Robotics panel — rendered entirely by the spec from the robotics plugin.
// See plugins/robotics/_panel.py. No robotics-panel-specific React code.

import { SpecRenderer } from "./SpecRenderer";

export function RoboticsPanel({ active = true }: { active?: boolean }) {
  return <SpecRenderer label="Robotics" active={active} />;
}
