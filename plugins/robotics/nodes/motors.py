"""Motor/actuator nodes — command output for robotics graphs."""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


class MotorCommandNode(BaseNode):
    """Generate a motor command with speed + direction + clamping."""
    type_name   = "rob_motor_cmd"
    label       = "Motor Command"
    category    = "Actuators"
    description = "Generate a motor command. Clamps to [min, max], outputs speed + direction."

    def _setup_ports(self) -> None:
        self.add_input("speed",     PortType.FLOAT, 0.0,
                       description="Desired speed (-1 to 1 or absolute)")
        self.add_input("min_speed", PortType.FLOAT, -1.0)
        self.add_input("max_speed", PortType.FLOAT, 1.0)
        self.add_input("deadband",  PortType.FLOAT, 0.05,
                       description="Speeds below this are set to 0")
        self.add_output("command",   PortType.FLOAT,
                        description="Clamped speed command")
        self.add_output("direction", PortType.INT,
                        description="1=forward, -1=reverse, 0=stopped")
        self.add_output("pwm",       PortType.INT,
                        description="PWM value 0-255 (absolute speed mapped)")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        speed = float(inputs.get("speed") or 0)
        lo    = float(inputs.get("min_speed") or -1)
        hi    = float(inputs.get("max_speed") or 1)
        dead  = float(inputs.get("deadband") or 0.05)

        if abs(speed) < dead:
            speed = 0.0
        speed = max(lo, min(hi, speed))

        direction = 1 if speed > 0 else (-1 if speed < 0 else 0)
        pwm = int(min(255, abs(speed) * 255))
        return {"command": speed, "direction": direction, "pwm": pwm}


class ServoNode(BaseNode):
    """Servo motor — angle command with limits and speed constraint."""
    type_name   = "rob_servo"
    label       = "Servo"
    category    = "Actuators"
    description = "Servo angle command. Clamps to [min_angle, max_angle]."

    def __init__(self):
        self._current_angle = 90.0
        super().__init__()

    def _setup_ports(self) -> None:
        self.add_input("target_angle", PortType.FLOAT, 90.0)
        self.add_input("min_angle",    PortType.FLOAT, 0.0)
        self.add_input("max_angle",    PortType.FLOAT, 180.0)
        self.add_input("max_speed",    PortType.FLOAT, 60.0,
                       description="Max degrees per tick")
        self.add_output("angle",       PortType.FLOAT,
                        description="Current angle (after speed limiting)")
        self.add_output("at_target",   PortType.BOOL)
        self.add_output("pulse_us",    PortType.INT,
                        description="Servo pulse width in microseconds (500-2500)")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        target = float(inputs.get("target_angle") or 90)
        lo     = float(inputs.get("min_angle") or 0)
        hi     = float(inputs.get("max_angle") or 180)
        rate   = abs(float(inputs.get("max_speed") or 60))

        target = max(lo, min(hi, target))
        diff = target - self._current_angle
        if abs(diff) <= rate:
            self._current_angle = target
        else:
            self._current_angle += rate if diff > 0 else -rate

        # Map 0-180 to 500-2500us pulse
        pulse = int(500 + (self._current_angle / 180.0) * 2000)
        return {
            "angle": self._current_angle,
            "at_target": abs(self._current_angle - target) < 0.5,
            "pulse_us": pulse,
        }


class StepperNode(BaseNode):
    """Stepper motor — step count with acceleration profile."""
    type_name   = "rob_stepper"
    label       = "Stepper"
    category    = "Actuators"
    description = "Stepper motor command. Outputs step count, direction, and step timing."

    def __init__(self):
        self._position = 0  # steps
        self._velocity = 0.0  # steps/s
        super().__init__()

    def _setup_ports(self) -> None:
        self.add_input("target_steps", PortType.INT, 0)
        self.add_input("max_speed",    PortType.FLOAT, 1000.0,
                       description="Max steps per second")
        self.add_input("acceleration", PortType.FLOAT, 500.0,
                       description="Steps/s^2")
        self.add_input("dt",           PortType.FLOAT, 0.01)
        self.add_output("position",    PortType.INT,
                        description="Current step position")
        self.add_output("velocity",    PortType.FLOAT,
                        description="Current velocity (steps/s)")
        self.add_output("moving",      PortType.BOOL)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        target = int(inputs.get("target_steps") or 0)
        max_v  = abs(float(inputs.get("max_speed") or 1000))
        accel  = abs(float(inputs.get("acceleration") or 500))
        dt     = max(1e-6, float(inputs.get("dt") or 0.01))

        error = target - self._position
        desired_v = max_v if error > 0 else (-max_v if error < 0 else 0)

        # Accelerate toward desired velocity
        dv = accel * dt
        if abs(desired_v - self._velocity) <= dv:
            self._velocity = desired_v
        elif desired_v > self._velocity:
            self._velocity += dv
        else:
            self._velocity -= dv

        # Decelerate near target
        stopping_dist = (self._velocity ** 2) / (2 * accel + 1e-9)
        if abs(error) <= stopping_dist:
            self._velocity *= 0.9  # simple deceleration

        # Update position
        steps = self._velocity * dt
        self._position += int(steps) if abs(steps) >= 1 else 0

        return {
            "position": self._position,
            "velocity": self._velocity,
            "moving": abs(self._velocity) > 0.5,
        }
