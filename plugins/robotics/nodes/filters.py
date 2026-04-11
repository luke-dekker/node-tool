"""Signal processing / filter nodes for cleaning sensor data."""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


class LowPassFilterNode(BaseNode):
    """First-order exponential low-pass filter."""
    type_name   = "rob_lowpass"
    label       = "Low Pass Filter"
    category    = "Signal"
    description = "Smooth noisy data. alpha=0.1 is heavy filtering, alpha=0.9 is light."

    def __init__(self):
        self._filtered = 0.0
        self._initialized = False
        super().__init__()

    def _setup_ports(self) -> None:
        self.add_input("value",  PortType.FLOAT, 0.0)
        self.add_input("alpha",  PortType.FLOAT, 0.2,
                       description="Filter coefficient (0-1). Lower = smoother.")
        self.add_input("reset",  PortType.BOOL, False)
        self.add_output("filtered", PortType.FLOAT)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        val   = float(inputs.get("value") or 0)
        alpha = max(0.0, min(1.0, float(inputs.get("alpha") or 0.2)))
        if bool(inputs.get("reset", False)) or not self._initialized:
            self._filtered = val
            self._initialized = True
        else:
            self._filtered = alpha * val + (1 - alpha) * self._filtered
        return {"filtered": self._filtered}


class MovingAverageNode(BaseNode):
    """Windowed moving average — smooth signal with configurable window size."""
    type_name   = "rob_moving_avg"
    label       = "Moving Average"
    category    = "Signal"
    description = "Sliding window mean. Window=10 averages the last 10 values."

    def __init__(self):
        self._buffer: list[float] = []
        super().__init__()

    def _setup_ports(self) -> None:
        self.add_input("value",  PortType.FLOAT, 0.0)
        self.add_input("window", PortType.INT, 10,
                       description="Number of samples to average")
        self.add_input("reset",  PortType.BOOL, False)
        self.add_output("average", PortType.FLOAT)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        val = float(inputs.get("value") or 0)
        win = max(1, int(inputs.get("window") or 10))
        if bool(inputs.get("reset", False)):
            self._buffer.clear()
        self._buffer.append(val)
        if len(self._buffer) > win:
            self._buffer = self._buffer[-win:]
        avg = sum(self._buffer) / len(self._buffer) if self._buffer else 0.0
        return {"average": avg}


class DerivativeNode(BaseNode):
    """Numerical derivative — velocity from position (or any rate of change)."""
    type_name   = "rob_derivative"
    label       = "Derivative"
    category    = "Signal"
    description = "dx/dt — numerical differentiation. First call returns 0."

    def __init__(self):
        self._prev = None
        super().__init__()

    def _setup_ports(self) -> None:
        self.add_input("value", PortType.FLOAT, 0.0)
        self.add_input("dt",    PortType.FLOAT, 0.01)
        self.add_output("derivative", PortType.FLOAT)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        val = float(inputs.get("value") or 0)
        dt  = max(1e-6, float(inputs.get("dt") or 0.01))
        if self._prev is None:
            self._prev = val
            return {"derivative": 0.0}
        deriv = (val - self._prev) / dt
        self._prev = val
        return {"derivative": deriv}


class IntegratorNode(BaseNode):
    """Numerical integrator — position from velocity (trapezoidal rule)."""
    type_name   = "rob_integrator"
    label       = "Integrator"
    category    = "Signal"
    description = "Accumulate value * dt over time. Trapezoidal rule."

    def __init__(self):
        self._sum = 0.0
        self._prev = 0.0
        super().__init__()

    def _setup_ports(self) -> None:
        self.add_input("value", PortType.FLOAT, 0.0)
        self.add_input("dt",    PortType.FLOAT, 0.01)
        self.add_input("reset", PortType.BOOL, False)
        self.add_output("integral", PortType.FLOAT)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        val = float(inputs.get("value") or 0)
        dt  = float(inputs.get("dt") or 0.01)
        if bool(inputs.get("reset", False)):
            self._sum = 0.0
            self._prev = val
        self._sum += 0.5 * (val + self._prev) * dt  # trapezoidal
        self._prev = val
        return {"integral": self._sum}


class KalmanFilter1DNode(BaseNode):
    """1D Kalman filter — estimate true state from noisy measurements."""
    type_name   = "rob_kalman1d"
    label       = "Kalman Filter 1D"
    category    = "Signal"
    description = (
        "1D Kalman filter. Estimates position from noisy position measurements. "
        "process_noise: how much the state changes per step. "
        "measurement_noise: how noisy the sensor is."
    )

    def __init__(self):
        self._x = 0.0  # state estimate
        self._p = 1.0  # estimate uncertainty
        super().__init__()

    def _setup_ports(self) -> None:
        self.add_input("measurement",       PortType.FLOAT, 0.0)
        self.add_input("process_noise",     PortType.FLOAT, 0.01,
                       description="Q: expected state change variance per step")
        self.add_input("measurement_noise", PortType.FLOAT, 0.1,
                       description="R: sensor noise variance")
        self.add_input("reset",             PortType.BOOL, False)
        self.add_output("estimate",         PortType.FLOAT)
        self.add_output("uncertainty",      PortType.FLOAT)
        self.add_output("kalman_gain",      PortType.FLOAT)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        z = float(inputs.get("measurement") or 0)
        q = max(1e-9, float(inputs.get("process_noise") or 0.01))
        r = max(1e-9, float(inputs.get("measurement_noise") or 0.1))

        if bool(inputs.get("reset", False)):
            self._x = z
            self._p = 1.0

        # Predict
        self._p += q
        # Update
        k = self._p / (self._p + r)
        self._x += k * (z - self._x)
        self._p *= (1 - k)

        return {"estimate": self._x, "uncertainty": self._p, "kalman_gain": k}
