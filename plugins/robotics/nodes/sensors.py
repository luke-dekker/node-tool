"""Sensor nodes — data input for robotics graphs.

These generate or read sensor data. The Sensor Simulator is useful for
testing control loops without hardware; the hardware nodes (IMU, Encoder,
Distance) read from serial/I2C when connected.
"""
from __future__ import annotations
import math
import random
from typing import Any
from core.node import BaseNode, PortType


class SensorSimulatorNode(BaseNode):
    """Generate synthetic sensor data — sine, noise, step, ramp — for testing."""
    type_name   = "rob_sensor_sim"
    label       = "Sensor Simulator"
    category    = "Robotics"
    subcategory = "Robotics"
    subcategory = "Sensors"
    description = (
        "Generate synthetic sensor data for testing control loops. "
        "Modes: sine, noise, step, ramp, constant."
    )

    def __init__(self):
        self._tick = 0
        super().__init__()

    def _setup_ports(self) -> None:
        self.add_input("mode",       PortType.STRING, "sine",
                       choices=["sine", "noise", "step", "ramp", "constant"])
        self.add_input("amplitude",  PortType.FLOAT, 1.0)
        self.add_input("frequency",  PortType.FLOAT, 1.0,
                       description="Hz (for sine mode)")
        self.add_input("offset",     PortType.FLOAT, 0.0)
        self.add_input("noise_std",  PortType.FLOAT, 0.0,
                       description="Additive Gaussian noise")
        self.add_input("dt",         PortType.FLOAT, 0.01)
        self.add_input("reset",      PortType.BOOL, False)
        self.add_output("value",     PortType.FLOAT)
        self.add_output("time",      PortType.FLOAT)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if bool(inputs.get("reset", False)):
            self._tick = 0

        mode  = str(inputs.get("mode") or "sine").lower()
        amp   = float(inputs.get("amplitude") or 1)
        freq  = float(inputs.get("frequency") or 1)
        off   = float(inputs.get("offset") or 0)
        noise = float(inputs.get("noise_std") or 0)
        dt    = float(inputs.get("dt") or 0.01)

        t = self._tick * dt
        self._tick += 1

        if mode == "sine":
            val = amp * math.sin(2 * math.pi * freq * t)
        elif mode == "noise":
            val = amp * random.gauss(0, 1)
        elif mode == "step":
            val = amp if t >= 1.0 / max(freq, 0.01) else 0.0
        elif mode == "ramp":
            val = amp * t * freq
        else:  # constant
            val = amp

        val += off
        if noise > 0:
            val += noise * random.gauss(0, 1)

        return {"value": val, "time": t}


class EncoderNode(BaseNode):
    """Rotary encoder — converts tick counts to angle and velocity."""
    type_name   = "rob_encoder"
    label       = "Encoder"
    category    = "Robotics"
    subcategory = "Sensors"
    description = "Rotary encoder: raw counts → angle (deg) and velocity (deg/s)."

    def __init__(self):
        self._prev_count = 0
        super().__init__()

    def _setup_ports(self) -> None:
        self.add_input("counts",     PortType.INT, 0,
                       description="Raw encoder tick count")
        self.add_input("cpr",        PortType.INT, 1024,
                       description="Counts per revolution")
        self.add_input("dt",         PortType.FLOAT, 0.01)
        self.add_output("angle",     PortType.FLOAT,
                        description="Angle in degrees")
        self.add_output("velocity",  PortType.FLOAT,
                        description="Angular velocity in deg/s")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        counts = int(inputs.get("counts") or 0)
        cpr    = max(1, int(inputs.get("cpr") or 1024))
        dt     = max(1e-6, float(inputs.get("dt") or 0.01))
        angle  = (counts / cpr) * 360.0
        delta  = counts - self._prev_count
        velocity = (delta / cpr) * 360.0 / dt
        self._prev_count = counts
        return {"angle": angle, "velocity": velocity}


class DistanceSensorNode(BaseNode):
    """Ultrasonic/LIDAR distance reading with min/max range filtering."""
    type_name   = "rob_distance"
    label       = "Distance Sensor"
    category    = "Robotics"
    subcategory = "Sensors"
    description = "Range sensor with min/max filtering. Out-of-range → None."

    def _setup_ports(self) -> None:
        self.add_input("raw_distance", PortType.FLOAT, 0.0,
                       description="Raw distance reading (m)")
        self.add_input("min_range",    PortType.FLOAT, 0.02)
        self.add_input("max_range",    PortType.FLOAT, 4.0)
        self.add_output("distance",    PortType.FLOAT)
        self.add_output("in_range",    PortType.BOOL)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        raw = float(inputs.get("raw_distance") or 0)
        lo  = float(inputs.get("min_range") or 0.02)
        hi  = float(inputs.get("max_range") or 4.0)
        ok  = lo <= raw <= hi
        return {"distance": raw if ok else 0.0, "in_range": ok}


class AnalogSensorNode(BaseNode):
    """Generic analog sensor — voltage to physical value with scaling."""
    type_name   = "rob_analog"
    label       = "Analog Sensor"
    category    = "Robotics"
    subcategory = "Sensors"
    description = "Voltage → scaled physical value. value = (voltage - offset) * scale"

    def _setup_ports(self) -> None:
        self.add_input("voltage",  PortType.FLOAT, 0.0)
        self.add_input("offset",   PortType.FLOAT, 0.0)
        self.add_input("scale",    PortType.FLOAT, 1.0)
        self.add_output("value",   PortType.FLOAT)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        v   = float(inputs.get("voltage") or 0)
        off = float(inputs.get("offset") or 0)
        s   = float(inputs.get("scale") or 1)
        return {"value": (v - off) * s}
