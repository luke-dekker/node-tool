"""Trajectory planning and safety limiting nodes for robotics control."""
from __future__ import annotations
import math
from typing import Any
from core.node import BaseNode, PortType


class TrajectoryPlannerNode(BaseNode):
    """Trapezoidal velocity profile trajectory planner.

    Generates smooth position, velocity, and acceleration commands for
    waypoint-to-waypoint motion. Three phases: acceleration, cruise, deceleration.
    Stateful: each execute() is one time step.
    """
    type_name   = "rob_trajectory"
    label       = "Trajectory Planner"
    category    = "Robotics"
    subcategory = "Control"
    description = (
        "Trapezoidal velocity profile trajectory planner. Generates smooth "
        "position/velocity for waypoint-to-waypoint motion."
    )

    def __init__(self):
        self._time = 0.0
        self._total_time = 0.0
        self._start = 0.0
        self._dist = 0.0
        self._sign = 1
        self._max_v = 1.0
        self._max_a = 2.0
        self._t_accel = 0.0
        self._d_accel = 0.0
        self._t_cruise = 0.0
        self._is_triangle = False
        super().__init__()

    def _setup_ports(self) -> None:
        self.add_input("start_pos",        PortType.FLOAT, 0.0,
                       description="Starting position")
        self.add_input("end_pos",          PortType.FLOAT, 1.0,
                       description="Target position")
        self.add_input("max_velocity",     PortType.FLOAT, 1.0,
                       description="Max velocity during cruise phase")
        self.add_input("max_acceleration", PortType.FLOAT, 2.0,
                       description="Accel/decel rate")
        self.add_input("dt",              PortType.FLOAT, 0.01,
                       description="Time step per tick")
        self.add_input("reset",           PortType.BOOL, False,
                       description="Reset trajectory to beginning")

        self.add_output("position",     PortType.FLOAT, description="Commanded position")
        self.add_output("velocity",     PortType.FLOAT, description="Commanded velocity")
        self.add_output("acceleration", PortType.FLOAT, description="Commanded acceleration")
        self.add_output("done",         PortType.BOOL,  description="True when complete")
        self.add_output("progress",     PortType.FLOAT, description="0.0 to 1.0")

    def _plan(self, start, end, max_v, max_a):
        self._start = start
        self._dist = end - start
        self._sign = 1 if self._dist >= 0 else -1
        dist = abs(self._dist)
        self._max_a = max(1e-9, max_a)
        self._max_v = max(1e-9, max_v)
        self._time = 0.0

        t_accel = self._max_v / self._max_a
        d_accel = 0.5 * self._max_a * t_accel * t_accel

        if 2 * d_accel >= dist:
            # Triangle profile — never reaches max velocity
            self._max_v = math.sqrt(self._max_a * dist)
            t_accel = self._max_v / self._max_a if self._max_a > 0 else 0
            d_accel = 0.5 * self._max_a * t_accel * t_accel
            self._is_triangle = True
            self._total_time = 2 * t_accel
            self._t_cruise = 0.0
        else:
            self._is_triangle = False
            self._t_cruise = (dist - 2 * d_accel) / self._max_v
            self._total_time = 2 * t_accel + self._t_cruise

        self._t_accel = t_accel
        self._d_accel = d_accel

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        start = float(inputs.get("start_pos") or 0)
        end   = float(inputs.get("end_pos") or 1)
        max_v = abs(float(inputs.get("max_velocity") or 1))
        max_a = abs(float(inputs.get("max_acceleration") or 2))
        dt    = max(1e-6, float(inputs.get("dt") or 0.01))

        if bool(inputs.get("reset", False)) or self._total_time == 0.0:
            self._plan(start, end, max_v, max_a)

        self._time = min(self._time + dt, self._total_time)
        t = self._time
        ta = self._t_accel
        s = self._sign
        a = self._max_a
        v = self._max_v

        if self._total_time <= 0:
            return {"position": start, "velocity": 0.0, "acceleration": 0.0,
                    "done": True, "progress": 1.0}

        if self._is_triangle:
            if t <= ta:
                pos = self._start + s * 0.5 * a * t * t
                vel = s * a * t
                acc = s * a
            else:
                td = t - ta
                pos = self._start + s * (self._d_accel + v * td - 0.5 * a * td * td)
                vel = s * (v - a * td)
                acc = -s * a
        else:
            if t <= ta:
                pos = self._start + s * 0.5 * a * t * t
                vel = s * a * t
                acc = s * a
            elif t <= ta + self._t_cruise:
                pos = self._start + s * (self._d_accel + v * (t - ta))
                vel = s * v
                acc = 0.0
            else:
                td = t - (ta + self._t_cruise)
                pos = self._start + s * (self._d_accel + v * self._t_cruise + v * td - 0.5 * a * td * td)
                vel = s * (v - a * td)
                acc = -s * a

        progress = min(1.0, t / self._total_time)
        done = progress >= 1.0
        if done:
            pos = self._start + self._dist
            vel = 0.0
            acc = 0.0

        return {"position": pos, "velocity": vel, "acceleration": acc,
                "done": done, "progress": progress}


class SafetyLimiterNode(BaseNode):
    """Clamp joint commands within safe limits. Latching emergency stop.

    Rate limiting tracks previous output and clamps delta to rate_limit * dt.
    E-stop is latching: once triggered, stays active until reset_estop.
    """
    type_name   = "rob_safety_limiter"
    label       = "Safety Limiter"
    category    = "Robotics"
    subcategory = "Control"
    description = (
        "Clamp joint commands within safe limits. Triggers emergency stop "
        "on threshold breach."
    )

    def __init__(self):
        self._prev_output = 0.0
        self._estop_latched = False
        super().__init__()

    def _setup_ports(self) -> None:
        self.add_input("command",      PortType.FLOAT, 0.0, description="Raw command")
        self.add_input("min_limit",    PortType.FLOAT, -1.0)
        self.add_input("max_limit",    PortType.FLOAT, 1.0)
        self.add_input("rate_limit",   PortType.FLOAT, 10.0,
                       description="Max change per tick (0 = no limit)")
        self.add_input("dt",           PortType.FLOAT, 0.01)
        self.add_input("estop",        PortType.BOOL, False, description="Emergency stop input")
        self.add_input("reset_estop",  PortType.BOOL, False, description="Reset e-stop latch")

        self.add_output("output",       PortType.FLOAT, description="Safe clamped command")
        self.add_output("clamped",      PortType.BOOL,  description="True if output != command")
        self.add_output("estop_active", PortType.BOOL,  description="E-stop latched")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        command = float(inputs.get("command") or 0)
        lo = float(inputs.get("min_limit") or -1)
        hi = float(inputs.get("max_limit") or 1)
        rl = float(inputs.get("rate_limit") or 10)
        dt = max(1e-6, float(inputs.get("dt") or 0.01))

        if bool(inputs.get("reset_estop", False)):
            self._estop_latched = False
        if bool(inputs.get("estop", False)):
            self._estop_latched = True

        if self._estop_latched:
            self._prev_output = 0.0
            return {"output": 0.0, "clamped": command != 0, "estop_active": True}

        out = max(lo, min(hi, command))
        clamped = out != command

        if rl > 0:
            max_delta = rl * dt
            delta = out - self._prev_output
            if abs(delta) > max_delta:
                out = self._prev_output + (max_delta if delta > 0 else -max_delta)
                clamped = True

        self._prev_output = out
        return {"output": out, "clamped": clamped, "estop_active": False}
