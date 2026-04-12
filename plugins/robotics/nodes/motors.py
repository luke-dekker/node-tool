"""Motor/actuator nodes — command output for robotics graphs."""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


class MotorCommandNode(BaseNode):
    """Generate a motor command with speed + direction + clamping."""
    type_name   = "rob_motor_cmd"
    label       = "Motor Command"
    category    = "Robotics"
    subcategory = "Robotics"
    subcategory = "Actuators"
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
    category    = "Robotics"
    subcategory = "Actuators"
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


class ServoBusNode(BaseNode):
    """Multi-joint servo bus — read/write a chain of servos over serial.

    Protocol-agnostic: pick Feetech STS/SCS, Dynamixel, or raw PWM. Joint
    names and IDs are configurable so this works for any robot arm, not
    just the SO-101. Drivers are imported lazily per protocol so you only
    need the SDKs for the protocol you actually use.
    """
    type_name   = "rob_servo_bus"
    label       = "Servo Bus"
    category    = "Robotics"
    subcategory = "Actuators"
    description = (
        "Read/write a bus of servos over serial. Pick a protocol "
        "(feetech_sts, feetech_scs, dynamixel, pwm_direct) and configure "
        "joint names + IDs. Works for any robot arm, not just SO-101."
    )

    def __init__(self):
        self._bus = None
        self._bus_key = ""
        super().__init__()

    def _setup_ports(self) -> None:
        self.add_input("protocol",   PortType.STRING, "feetech_sts",
                       choices=["feetech_sts", "feetech_scs", "dynamixel", "pwm_direct"],
                       description="Servo protocol / SDK to use")
        self.add_input("port",       PortType.STRING, "",
                       description="Serial port (e.g. COM3 or /dev/ttyACM0)")
        self.add_input("baud_rate",  PortType.INT, 1000000)
        self.add_input("joint_names", PortType.STRING,
                       "j1,j2,j3,j4,j5,j6",
                       description="Comma-separated joint names")
        self.add_input("joint_ids",   PortType.STRING,
                       "1,2,3,4,5,6",
                       description="Comma-separated motor IDs (integers)")
        self.add_input("model",       PortType.STRING, "sts3215",
                       description="Servo model name (e.g. sts3215, xl330)")
        self.add_input("target_positions", PortType.ANY, None,
                       description="List/tensor of target positions, length = #joints")
        self.add_input("write",       PortType.BOOL, False,
                       description="Set True to write target_positions to the bus")
        self.add_output("positions",  PortType.ANY,
                        description="Current joint positions (list of floats)")
        self.add_output("connected",  PortType.BOOL)
        self.add_output("info",       PortType.STRING)

    def _parse_list(self, text: str) -> list[str]:
        return [p.strip() for p in str(text or "").split(",") if p.strip()]

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        protocol = str(inputs.get("protocol") or "feetech_sts").lower()
        port     = str(inputs.get("port") or "")
        baud     = int(inputs.get("baud_rate") or 1000000)
        names    = self._parse_list(inputs.get("joint_names"))
        try:
            ids = [int(x) for x in self._parse_list(inputs.get("joint_ids"))]
        except ValueError:
            return {"positions": [], "connected": False,
                    "info": "joint_ids must be comma-separated integers"}
        model    = str(inputs.get("model") or "sts3215")
        targets  = inputs.get("target_positions")
        do_write = bool(inputs.get("write", False))

        n = min(len(names), len(ids))
        if n == 0:
            return {"positions": [], "connected": False,
                    "info": "Configure joint_names and joint_ids"}
        names, ids = names[:n], ids[:n]

        if not port:
            return {"positions": [0.0] * n, "connected": False,
                    "info": f"{protocol}: set serial port"}

        key = f"{protocol}|{port}|{baud}|{model}|{','.join(names)}|{','.join(map(str, ids))}"
        try:
            if protocol in ("feetech_sts", "feetech_scs"):
                if self._bus is None or self._bus_key != key:
                    from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus
                    self._bus = FeetechMotorsBus(
                        port=port,
                        motors={name: (mid, model) for name, mid in zip(names, ids)},
                    )
                    self._bus.connect()
                    self._bus_key = key
                positions = list(self._bus.read("Present_Position"))[:n]
                if do_write and targets is not None:
                    vals = self._coerce_targets(targets, n)
                    if vals is not None:
                        self._bus.write("Goal_Position", vals)
                return {"positions": [float(p) for p in positions],
                        "connected": True,
                        "info": f"feetech {port}@{baud} — {n} joints"}

            if protocol == "dynamixel":
                if self._bus is None or self._bus_key != key:
                    from dynamixel_sdk import PortHandler, PacketHandler
                    ph = PortHandler(port)
                    if not ph.openPort():
                        return {"positions": [0.0] * n, "connected": False,
                                "info": f"dynamixel: failed to open {port}"}
                    ph.setBaudRate(baud)
                    self._bus = (ph, PacketHandler(2.0))
                    self._bus_key = key
                ph, pk = self._bus
                positions = []
                for mid in ids:
                    pos, _, _ = pk.read4ByteTxRx(ph, mid, 132)
                    positions.append(float(pos))
                if do_write and targets is not None:
                    vals = self._coerce_targets(targets, n)
                    if vals is not None:
                        for mid, v in zip(ids, vals):
                            pk.write4ByteTxRx(ph, mid, 116, int(v))
                return {"positions": positions, "connected": True,
                        "info": f"dynamixel {port}@{baud} — {n} joints"}

            if protocol == "pwm_direct":
                # Passthrough: echo targets as positions, no hardware.
                vals = self._coerce_targets(targets, n) or [0.0] * n
                return {"positions": [float(v) for v in vals], "connected": True,
                        "info": f"pwm_direct (sim) — {n} channels"}

            return {"positions": [0.0] * n, "connected": False,
                    "info": f"Unknown protocol: {protocol}"}

        except ImportError as exc:
            hint = {
                "feetech_sts": "pip install lerobot[feetech] or scservo-sdk",
                "feetech_scs": "pip install lerobot[feetech] or scservo-sdk",
                "dynamixel":   "pip install dynamixel-sdk",
            }.get(protocol, "")
            return {"positions": [0.0] * n, "connected": False,
                    "info": f"{protocol}: driver not installed ({exc}). {hint}"}
        except Exception as exc:
            return {"positions": [0.0] * n, "connected": False,
                    "info": f"{protocol} error: {exc}"}

    def _coerce_targets(self, targets: Any, n: int):
        try:
            import torch
            if isinstance(targets, torch.Tensor):
                targets = targets.detach().cpu().tolist()
        except ImportError:
            pass
        if isinstance(targets, (list, tuple)) and len(targets) >= n:
            return [float(x) for x in targets[:n]]
        return None

    def export(self, iv, ov):
        return [], [
            "# Servo bus — drivers imported at runtime based on protocol.",
            "# See plugins/robotics/nodes/motors.py for the full implementation.",
        ]


class StepperNode(BaseNode):
    """Stepper motor — step count with acceleration profile."""
    type_name   = "rob_stepper"
    label       = "Stepper"
    category    = "Robotics"
    subcategory = "Actuators"
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
