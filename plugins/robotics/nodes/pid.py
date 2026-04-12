"""PID Controller node — the fundamental control building block.

Classic PID with anti-windup and output clamping. Maintains internal state
(integral accumulator, previous error) across graph executions — each
execute() call is one control loop tick.

Wire: setpoint + measurement → PID → motor command.
Tune: Kp, Ki, Kd, output_min, output_max, dt.
"""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


class PIDControllerNode(BaseNode):
    type_name   = "rob_pid"
    label       = "PID Controller"
    category    = "Robotics"
    subcategory = "Robotics"
    subcategory = "Control"
    description = (
        "Classic PID controller with anti-windup. Each graph execution is "
        "one control tick. Internal state persists across ticks."
    )

    def __init__(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_output = 0.0
        super().__init__()

    def _setup_ports(self) -> None:
        self.add_input("setpoint",    PortType.FLOAT, 0.0,
                       description="Desired value")
        self.add_input("measurement", PortType.FLOAT, 0.0,
                       description="Current measured value")
        self.add_input("Kp",          PortType.FLOAT, 1.0,
                       description="Proportional gain")
        self.add_input("Ki",          PortType.FLOAT, 0.0,
                       description="Integral gain")
        self.add_input("Kd",          PortType.FLOAT, 0.0,
                       description="Derivative gain")
        self.add_input("dt",          PortType.FLOAT, 0.01,
                       description="Time step (seconds)")
        self.add_input("output_min",  PortType.FLOAT, -1.0)
        self.add_input("output_max",  PortType.FLOAT, 1.0)
        self.add_input("reset",       PortType.BOOL, False,
                       description="Reset integral and derivative state")

        self.add_output("output",     PortType.FLOAT,
                        description="Control output (clamped)")
        self.add_output("error",      PortType.FLOAT,
                        description="Current error (setpoint - measurement)")
        self.add_output("P",          PortType.FLOAT, description="Proportional term")
        self.add_output("I",          PortType.FLOAT, description="Integral term")
        self.add_output("D",          PortType.FLOAT, description="Derivative term")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        sp   = float(inputs.get("setpoint") or 0)
        meas = float(inputs.get("measurement") or 0)
        kp   = float(inputs.get("Kp") or 1)
        ki   = float(inputs.get("Ki") or 0)
        kd   = float(inputs.get("Kd") or 0)
        dt   = max(1e-6, float(inputs.get("dt") or 0.01))
        lo   = float(inputs.get("output_min") or -1)
        hi   = float(inputs.get("output_max") or 1)

        if bool(inputs.get("reset", False)):
            self._integral = 0.0
            self._prev_error = 0.0

        error = sp - meas
        p_term = kp * error
        self._integral += error * dt
        i_term = ki * self._integral
        d_term = kd * (error - self._prev_error) / dt
        self._prev_error = error

        raw_output = p_term + i_term + d_term
        output = max(lo, min(hi, raw_output))

        # Anti-windup: if output is saturated, stop accumulating integral
        if output != raw_output:
            self._integral -= error * dt

        self._prev_output = output
        return {"output": output, "error": error,
                "P": p_term, "I": i_term, "D": d_term}

    def export(self, iv, ov):
        return [], [
            f"# PID Controller — stateful, runs once per tick",
            f"# TODO: implement PID state management for export",
            f"{ov.get('output', '_pid_out')} = 0.0",
        ]


class BangBangControllerNode(BaseNode):
    type_name   = "rob_bangbang"
    label       = "Bang-Bang"
    category    = "Robotics"
    subcategory = "Control"
    description = "On/off controller with hysteresis. Output is high or low."

    def _setup_ports(self) -> None:
        self.add_input("setpoint",    PortType.FLOAT, 0.0)
        self.add_input("measurement", PortType.FLOAT, 0.0)
        self.add_input("hysteresis",  PortType.FLOAT, 0.1,
                       description="Dead band around setpoint")
        self.add_input("output_high", PortType.FLOAT, 1.0)
        self.add_input("output_low",  PortType.FLOAT, 0.0)
        self.add_output("output",     PortType.FLOAT)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        sp   = float(inputs.get("setpoint") or 0)
        meas = float(inputs.get("measurement") or 0)
        hyst = float(inputs.get("hysteresis") or 0.1)
        hi   = float(inputs.get("output_high") or 1)
        lo   = float(inputs.get("output_low") or 0)
        error = sp - meas
        output = hi if error > hyst else (lo if error < -hyst else (hi + lo) / 2)
        return {"output": output}


class RampGeneratorNode(BaseNode):
    type_name   = "rob_ramp"
    label       = "Ramp Generator"
    category    = "Robotics"
    subcategory = "Control"
    description = "Smoothly ramp toward a target value at a configurable rate per tick."

    def __init__(self):
        self._current = 0.0
        super().__init__()

    def _setup_ports(self) -> None:
        self.add_input("target",    PortType.FLOAT, 0.0)
        self.add_input("rate",      PortType.FLOAT, 0.1,
                       description="Max change per tick")
        self.add_input("reset",     PortType.BOOL, False)
        self.add_output("output",   PortType.FLOAT)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        target = float(inputs.get("target") or 0)
        rate   = abs(float(inputs.get("rate") or 0.1))
        if bool(inputs.get("reset", False)):
            self._current = 0.0
        diff = target - self._current
        if abs(diff) <= rate:
            self._current = target
        else:
            self._current += rate if diff > 0 else -rate
        return {"output": self._current}
