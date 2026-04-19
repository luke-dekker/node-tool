"""RoboticsController — GUI-free state and actions for the robotics panel.

Mirrors the pytorch TrainingOrchestrator pattern: one class owns domain
logic, every frontend drives it through the same RPC methods. Today most
actions are stubs (no hardware attached). Replace the stubs with actual
pyserial / ROS / SO-101 calls when the time comes — no GUI code changes.
"""
from __future__ import annotations
from typing import Any


class RoboticsController:
    def __init__(self):
        self._log_lines: list[str] = ["(serial output will appear here)"]
        self._connected: bool = False
        self._port: str = ""
        self._baud: str = "115200"

    # ── Actions ──────────────────────────────────────────────────────────

    def list_ports(self, _params: dict | None = None) -> dict:
        """Enumerate available serial ports."""
        try:
            import serial.tools.list_ports
            ports = [p.device for p in serial.tools.list_ports.comports()]
            return {"ports": ports or []}
        except ImportError:
            return {"ports": [], "error": "pyserial not installed"}

    def connect(self, params: dict) -> dict:
        port = str(params.get("port", "")).strip()
        baud = str(params.get("baud", "115200")).strip()
        if not port:
            return {"ok": False, "error": "No port selected"}
        self._port = port
        self._baud = baud
        self._connected = True
        self._log_lines.append(f"[Serial] connected to {port} @ {baud}")
        return {"ok": True, "port": port, "baud": baud}

    def disconnect(self, _params: dict | None = None) -> dict:
        if self._connected:
            self._log_lines.append(f"[Serial] disconnected from {self._port}")
        self._connected = False
        return {"ok": True}

    def send(self, params: dict) -> dict:
        cmd = str(params.get("cmd", "")).strip()
        if not cmd:
            return {"ok": False, "error": "Empty command"}
        if not self._connected:
            return {"ok": False, "error": "Not connected"}
        self._log_lines.append(f"> {cmd}")
        return {"ok": True}

    # ── Accessors ────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "connected": "Yes" if self._connected else "No",
            "port":      self._port or "—",
            "baud":      self._baud,
            "log_tail":  self._log_lines[-1] if self._log_lines else "",
        }

    def log(self) -> dict:
        """Return the last N log lines for the serial monitor display."""
        return {"lines": self._log_lines[-30:]}

    # ── RPC entry point ──────────────────────────────────────────────────

    def handle_rpc(self, method: str, params: dict | None = None) -> Any:
        params = params or {}
        handlers = {
            "robotics_list_ports":  self.list_ports,
            "robotics_connect":     self.connect,
            "robotics_disconnect":  self.disconnect,
            "robotics_send":        self.send,
            "get_robotics_state":   lambda _p: self.status(),
            "get_robotics_log":     lambda _p: self.log(),
        }
        h = handlers.get(method)
        if h is None:
            raise ValueError(f"Unknown robotics RPC method: {method}")
        return h(params)
