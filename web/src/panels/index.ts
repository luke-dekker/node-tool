// Registry of plugin panel builders, keyed by the name the plugin registers
// server-side. When the server reports a panel name we don't have a builder
// for, BottomPanel logs and skips it (same pattern as the Godot frontend).

import type { ComponentType } from "react";
import { TrainingPanel } from "./TrainingPanel";
import { RoboticsPanel } from "./RoboticsPanel";

export const PANEL_BUILDERS: Record<string, ComponentType> = {
  Training: TrainingPanel,
  Robotics: RoboticsPanel,
};
