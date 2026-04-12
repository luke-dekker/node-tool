"""Kinematics nodes — FK/IK for robot arms, coordinate transforms."""
from __future__ import annotations
import math
from typing import Any
from core.node import BaseNode, PortType


class ForwardKinematics2DNode(BaseNode):
    """2-link planar arm FK: joint angles → end effector (x, y)."""
    type_name   = "rob_fk_2d"
    label       = "FK 2-Link"
    category    = "Robotics"
    subcategory = "Robotics"
    subcategory = "Kinematics"
    description = (
        "Forward kinematics for a 2-link planar arm. "
        "Joint angles (degrees) → end effector position (x, y)."
    )

    def _setup_ports(self) -> None:
        self.add_input("theta1",  PortType.FLOAT, 0.0,
                       description="Joint 1 angle (degrees)")
        self.add_input("theta2",  PortType.FLOAT, 0.0,
                       description="Joint 2 angle (degrees)")
        self.add_input("L1",      PortType.FLOAT, 1.0, description="Link 1 length")
        self.add_input("L2",      PortType.FLOAT, 1.0, description="Link 2 length")
        self.add_output("x",      PortType.FLOAT, description="End effector X")
        self.add_output("y",      PortType.FLOAT, description="End effector Y")
        self.add_output("elbow_x", PortType.FLOAT, description="Elbow joint X")
        self.add_output("elbow_y", PortType.FLOAT, description="Elbow joint Y")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        t1 = math.radians(float(inputs.get("theta1") or 0))
        t2 = math.radians(float(inputs.get("theta2") or 0))
        L1 = float(inputs.get("L1") or 1)
        L2 = float(inputs.get("L2") or 1)
        ex = L1 * math.cos(t1)
        ey = L1 * math.sin(t1)
        x = ex + L2 * math.cos(t1 + t2)
        y = ey + L2 * math.sin(t1 + t2)
        return {"x": x, "y": y, "elbow_x": ex, "elbow_y": ey}


class InverseKinematics2DNode(BaseNode):
    """2-link planar arm IK: target (x, y) → joint angles."""
    type_name   = "rob_ik_2d"
    label       = "IK 2-Link"
    category    = "Robotics"
    subcategory = "Kinematics"
    description = (
        "Inverse kinematics for a 2-link planar arm. "
        "Target (x, y) → joint angles (degrees). Returns elbow-up solution."
    )

    def _setup_ports(self) -> None:
        self.add_input("target_x", PortType.FLOAT, 1.0)
        self.add_input("target_y", PortType.FLOAT, 1.0)
        self.add_input("L1",       PortType.FLOAT, 1.0)
        self.add_input("L2",       PortType.FLOAT, 1.0)
        self.add_output("theta1",  PortType.FLOAT, description="Joint 1 angle (degrees)")
        self.add_output("theta2",  PortType.FLOAT, description="Joint 2 angle (degrees)")
        self.add_output("reachable", PortType.BOOL)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        x  = float(inputs.get("target_x") or 1)
        y  = float(inputs.get("target_y") or 1)
        L1 = float(inputs.get("L1") or 1)
        L2 = float(inputs.get("L2") or 1)
        d  = math.sqrt(x*x + y*y)
        if d > L1 + L2 or d < abs(L1 - L2):
            return {"theta1": 0.0, "theta2": 0.0, "reachable": False}
        cos_t2 = (x*x + y*y - L1*L1 - L2*L2) / (2 * L1 * L2)
        cos_t2 = max(-1, min(1, cos_t2))
        t2 = math.acos(cos_t2)  # elbow-up
        t1 = math.atan2(y, x) - math.atan2(L2 * math.sin(t2), L1 + L2 * math.cos(t2))
        return {
            "theta1": math.degrees(t1),
            "theta2": math.degrees(t2),
            "reachable": True,
        }


class TransformNode(BaseNode):
    """Apply 2D rotation + translation to a point."""
    type_name   = "rob_transform_2d"
    label       = "Transform 2D"
    category    = "Robotics"
    subcategory = "Kinematics"
    description = "Rotate and translate a 2D point: (x', y') = R(angle) * (x, y) + (tx, ty)."

    def _setup_ports(self) -> None:
        self.add_input("x",     PortType.FLOAT, 0.0)
        self.add_input("y",     PortType.FLOAT, 0.0)
        self.add_input("angle", PortType.FLOAT, 0.0, description="Rotation (degrees)")
        self.add_input("tx",    PortType.FLOAT, 0.0, description="Translation X")
        self.add_input("ty",    PortType.FLOAT, 0.0, description="Translation Y")
        self.add_output("x_out", PortType.FLOAT)
        self.add_output("y_out", PortType.FLOAT)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        x  = float(inputs.get("x") or 0)
        y  = float(inputs.get("y") or 0)
        a  = math.radians(float(inputs.get("angle") or 0))
        tx = float(inputs.get("tx") or 0)
        ty = float(inputs.get("ty") or 0)
        xr = x * math.cos(a) - y * math.sin(a) + tx
        yr = x * math.sin(a) + y * math.cos(a) + ty
        return {"x_out": xr, "y_out": yr}
