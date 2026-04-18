"""Robotics plugin — sensors, actuators, control, signal processing, kinematics.

Designed to work alongside the PyTorch plugin: train a model in PyTorch,
deploy it in a robotics graph with PID stabilization, sensor fusion, and
motor control. Same tool, same paradigm, same graph execution engine.
"""
from core.plugins import PluginContext


def register(ctx: PluginContext) -> None:
    # ── Port types ──────────────────────────────────────────────────────
    ctx.register_port_type("SENSOR_DATA", default=None,
                           color=(60, 220, 180, 255), pin_shape="circle_filled",
                           description="Timestamped sensor reading (float + metadata)")
    ctx.register_port_type("MOTOR_CMD", default=None,
                           color=(255, 160, 60, 255), pin_shape="quad_filled",
                           description="Motor command (PWM/velocity/position)")
    ctx.register_port_type("POSE", default=None,
                           color=(180, 120, 255, 255), pin_shape="triangle",
                           description="6DOF pose (x, y, z, qx, qy, qz, qw)")
    ctx.register_port_type("JOINT_STATE", default=None,
                           color=(120, 200, 255, 255), pin_shape="triangle_filled",
                           description="Joint angles/velocities/efforts")
    ctx.register_port_type("SERIAL_FRAME", default=None,
                           color=(200, 200, 80, 255), pin_shape="quad",
                           description="Framed serial packet (bytes + metadata)")

    # ── Nodes ───────────────────────────────────────────────────────────
    from plugins.robotics import nodes
    ctx.discover_nodes(nodes)

    # ── Categories ──────────────────────────────────────────────────────
    ctx.add_categories(["Control", "Sensors", "Actuators", "Signal", "Kinematics"])

    # ── Panel ───────────────────────────────────────────────────────────
    # DPG adapter for the robotics panel. Other frontends (React, Godot)
    # can implement their own panel against the same controller API once
    # we extract one — see FRONTEND_CHECKLIST.md "Plugin-registered panels".
    from plugins.robotics._panel_dpg import build_robotics_panel
    ctx.register_panel("Robotics", build_robotics_panel)
