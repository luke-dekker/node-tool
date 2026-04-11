"""LeRobot plugin — HuggingFace LeRobot integration for the SO-100/101 robot arms.

Wraps the lerobot Python library to provide visual-programming access to:
  - LeRobot demonstration datasets (load from HuggingFace Hub)
  - Feetech STS3215 servo bus control (the SO-101's motors)
  - ACT / Diffusion Policy wrappers for imitation learning
  - Camera input for vision-based policies

Requires: pip install lerobot
Hardware: SO-100 or SO-101 robot arm with Feetech servos + USB adapter

The robotics plugin provides the PID/filter/kinematics primitives;
this plugin provides the LeRobot-specific integration layer.
"""
from core.plugins import PluginContext


def register(ctx: PluginContext) -> None:
    # ── Port types ──────────────────────────────────────────────────────
    ctx.register_port_type("LEROBOT_OBS", default=None,
                           color=(80, 200, 255, 255), pin_shape="quad_filled",
                           description="LeRobot observation dict (joints + images)")
    ctx.register_port_type("LEROBOT_ACTION", default=None,
                           color=(255, 140, 80, 255), pin_shape="quad_filled",
                           description="LeRobot action (joint position targets)")

    # ── Nodes ───────────────────────────────────────────────────────────
    from plugins.lerobot import nodes
    ctx.discover_nodes(nodes)

    # ── Categories ──────────────────────────────────────────────────────
    ctx.add_categories(["LeRobot"])
