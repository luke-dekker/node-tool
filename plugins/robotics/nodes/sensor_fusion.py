"""Sensor fusion nodes — complementary filter and extended Kalman filter.

Both maintain internal state across graph executions — each execute() is one tick.
"""
from __future__ import annotations
import numpy as np
from typing import Any
from core.node import BaseNode, PortType


class ComplementaryFilterNode(BaseNode):
    type_name   = "rob_complementary_filter"
    label       = "Complementary Filter"
    category    = "Robotics"
    subcategory = "Signal"
    description = (
        "Fuse accelerometer and gyroscope for angle estimation. "
        "angle = alpha * (angle + gyro*dt) + (1-alpha) * accel_angle."
    )

    def __init__(self):
        self._angle = 0.0
        super().__init__()

    def _setup_ports(self) -> None:
        self.add_input("accel_angle", PortType.FLOAT, 0.0,
                       description="Angle from accelerometer (degrees)")
        self.add_input("gyro_rate",   PortType.FLOAT, 0.0,
                       description="Angular velocity from gyroscope (deg/s)")
        self.add_input("alpha",       PortType.FLOAT, 0.98,
                       description="Filter coefficient (0-1). Higher = trust gyro more.")
        self.add_input("dt",          PortType.FLOAT, 0.01, description="Time step")
        self.add_input("reset",       PortType.BOOL,  False)

        self.add_output("angle", PortType.FLOAT, description="Fused angle estimate (degrees)")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        accel = float(inputs.get("accel_angle") or 0)
        gyro  = float(inputs.get("gyro_rate") or 0)
        alpha = max(0.0, min(1.0, float(inputs.get("alpha") or 0.98)))
        dt    = max(1e-6, float(inputs.get("dt") or 0.01))

        if bool(inputs.get("reset", False)):
            self._angle = 0.0

        self._angle = alpha * (self._angle + gyro * dt) + (1 - alpha) * accel
        return {"angle": self._angle}

    def export(self, iv, ov):
        return [], [
            f"# Complementary Filter — stateful, one tick per call",
            f"{ov.get('angle', '_cf_angle')} = 0.0",
        ]


class ExtendedKalmanFilterNode(BaseNode):
    type_name   = "rob_ekf"
    label       = "EKF 2D"
    category    = "Robotics"
    subcategory = "Signal"
    description = (
        "Extended Kalman Filter for 2D position+velocity tracking. "
        "State: [x, y, vx, vy]. Measurement: [x, y]."
    )

    def __init__(self):
        self._x = np.zeros(4)   # [x, y, vx, vy]
        self._P = np.eye(4)     # covariance
        super().__init__()

    def _setup_ports(self) -> None:
        self.add_input("meas_x",            PortType.FLOAT, 0.0, description="Measured x")
        self.add_input("meas_y",            PortType.FLOAT, 0.0, description="Measured y")
        self.add_input("dt",                PortType.FLOAT, 0.01)
        self.add_input("process_noise",     PortType.FLOAT, 0.1,
                       description="Q scalar")
        self.add_input("measurement_noise", PortType.FLOAT, 1.0,
                       description="R scalar")
        self.add_input("reset",             PortType.BOOL, False)

        self.add_output("est_x",  PortType.FLOAT, description="Estimated x")
        self.add_output("est_y",  PortType.FLOAT, description="Estimated y")
        self.add_output("est_vx", PortType.FLOAT, description="Estimated vx")
        self.add_output("est_vy", PortType.FLOAT, description="Estimated vy")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        mx = float(inputs.get("meas_x") or 0)
        my = float(inputs.get("meas_y") or 0)
        dt = max(1e-6, float(inputs.get("dt") or 0.01))
        q  = max(1e-9, float(inputs.get("process_noise") or 0.1))
        r  = max(1e-9, float(inputs.get("measurement_noise") or 1.0))

        if bool(inputs.get("reset", False)):
            self._x = np.zeros(4)
            self._P = np.eye(4)

        # Predict — constant velocity model
        F = np.array([[1, 0, dt, 0],
                       [0, 1, 0, dt],
                       [0, 0, 1,  0],
                       [0, 0, 0,  1]], dtype=np.float64)
        Q = q * np.eye(4)

        self._x = F @ self._x
        self._P = F @ self._P @ F.T + Q

        # Update — measure [x, y]
        H = np.array([[1, 0, 0, 0],
                       [0, 1, 0, 0]], dtype=np.float64)
        R = r * np.eye(2)

        z = np.array([mx, my])
        y = z - H @ self._x
        S = H @ self._P @ H.T + R
        K = self._P @ H.T @ np.linalg.inv(S)

        self._x = self._x + K @ y
        self._P = (np.eye(4) - K @ H) @ self._P

        return {
            "est_x":  float(self._x[0]),
            "est_y":  float(self._x[1]),
            "est_vx": float(self._x[2]),
            "est_vy": float(self._x[3]),
        }

    def export(self, iv, ov):
        return [], [
            f"# EKF 2D — stateful, one tick per call",
            f"{ov.get('est_x', '_ekf_x')} = 0.0",
            f"{ov.get('est_y', '_ekf_y')} = 0.0",
            f"{ov.get('est_vx', '_ekf_vx')} = 0.0",
            f"{ov.get('est_vy', '_ekf_vy')} = 0.0",
        ]
