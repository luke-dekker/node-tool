"""Feetech Servo Bus node — read/write STS3215 servos for the SO-100/101 arms.

The SO-101 uses 6x Feetech STS3215 servo motors on a half-duplex UART bus.
This node wraps the serial communication to read joint positions and write
target positions.

Hardware: Feetech STS3215 servos + USB-to-UART adapter
Protocol: Feetech half-duplex serial (similar to Dynamixel)
"""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


# SO-101 joint names in order (motor IDs 1-6)
_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                "wrist_flex", "wrist_roll", "gripper"]


class FeetechServoBusNode(BaseNode):
    """Read/write Feetech STS3215 servos — the SO-101's motor interface."""
    type_name   = "lr_feetech_bus"
    label       = "Feetech Servo Bus"
    category    = "LeRobot"
    description = (
        "Connect to Feetech STS3215 servos via serial. Reads current joint "
        "positions and writes target positions. 6 joints for the SO-101."
    )

    def __init__(self):
        self._bus = None
        self._connected_port = ""
        super().__init__()

    def _setup_ports(self) -> None:
        self.add_input("port",          PortType.STRING, "",
                       description="Serial port (e.g. COM3 or /dev/ttyACM0)")
        self.add_input("baud_rate",     PortType.INT, 1000000)
        self.add_input("target_joints", PortType.ANY, None,
                       description="6-element list/tensor of target positions (degrees)")
        self.add_input("write",         PortType.BOOL, False,
                       description="Set True to write target_joints to servos")
        # Output: current joint positions
        for i, name in enumerate(_JOINT_NAMES):
            self.add_output(name, PortType.FLOAT,
                            description=f"Joint {i+1} position (degrees)")
        self.add_output("connected", PortType.BOOL)
        self.add_output("info",      PortType.STRING)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        port = str(inputs.get("port") or "")
        baud = int(inputs.get("baud_rate") or 1000000)
        targets = inputs.get("target_joints")
        do_write = bool(inputs.get("write", False))

        out = {name: 0.0 for name in _JOINT_NAMES}
        out["connected"] = False
        out["info"] = ""

        if not port:
            out["info"] = "Set serial port"
            return out

        # Try to connect / read / write via the Feetech SDK
        try:
            from lerobot.common.robot_devices.motors.feetech import FeetechMotorsBus

            if self._bus is None or self._connected_port != port:
                self._bus = FeetechMotorsBus(
                    port=port,
                    motors={name: (i + 1, "sts3215")
                            for i, name in enumerate(_JOINT_NAMES)},
                )
                self._bus.connect()
                self._connected_port = port

            # Read current positions
            positions = self._bus.read("Present_Position")
            for i, name in enumerate(_JOINT_NAMES):
                if i < len(positions):
                    out[name] = float(positions[i])

            # Write targets if requested
            if do_write and targets is not None:
                import torch
                if isinstance(targets, torch.Tensor):
                    targets = targets.tolist()
                if isinstance(targets, (list, tuple)) and len(targets) >= 6:
                    goal = {name: targets[i] for i, name in enumerate(_JOINT_NAMES)}
                    self._bus.write("Goal_Position", list(goal.values()))

            out["connected"] = True
            out["info"] = f"Connected to {port} @ {baud}"

        except ImportError:
            out["info"] = "lerobot not installed. Run: pip install lerobot[feetech]"
        except Exception as exc:
            out["info"] = f"Error: {exc}"
            out["connected"] = False

        return out

    def export(self, iv, ov):
        return [], [
            f"# Feetech Servo Bus — requires hardware + lerobot SDK",
            f"# See: pip install lerobot[feetech]",
        ]
