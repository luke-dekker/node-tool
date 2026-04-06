"""Serial port IO nodes (pyserial).

Install:  pip install pyserial

Each node degrades gracefully when pyserial is not installed.
"""
from __future__ import annotations
import json
from typing import Any
from core.node import BaseNode, PortType


def _serial_available() -> bool:
    try:
        import serial  # noqa: F401
        return True
    except ImportError:
        return False


def _flatten(lst):
    for item in lst:
        if isinstance(item, (list, tuple)):
            yield from _flatten(item)
        else:
            yield item


def _encode(data: Any, encoding: str) -> bytes:
    """Convert data to bytes for serial transmission."""
    try:
        import torch
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().tolist()
    except ImportError:
        pass
    try:
        import numpy as np
        if isinstance(data, np.ndarray):
            data = data.tolist()
    except ImportError:
        pass

    if encoding == "json":
        return json.dumps(data).encode("utf-8")
    if encoding == "csv":
        if isinstance(data, (list, tuple)):
            flat = _flatten(data)
            return ",".join(str(v) for v in flat).encode("utf-8")
        return str(data).encode("utf-8")
    # raw
    return str(data).encode("utf-8")


class SerialOutputNode(BaseNode):
    """Send data to a serial port (Arduino, servo controller, MCU)."""
    type_name   = "io_serial_out"
    label       = "Serial Output"
    category    = "IO"
    subcategory = "Serial"
    description = (
        "Write data to a serial port. Data can be a tensor, list, number, or string. "
        "Encoding: 'json' (default), 'csv', or 'raw' (str conversion)."
    )

    def _setup_ports(self) -> None:
        self.add_input("data",     PortType.ANY,    default=None,
                       description="Tensor, list, number, or string to send")
        self.add_input("port",     PortType.STRING, default="COM3",
                       description="Serial port name, e.g. COM3 or /dev/ttyUSB0")
        self.add_input("baud",     PortType.INT,    default=115200)
        self.add_input("encoding", PortType.STRING, default="json",
                       description="json | csv | raw")
        self.add_input("newline",  PortType.BOOL,   default=True,
                       description="Append newline after each send")
        self.add_output("status",  PortType.STRING, description="'ok' or error message")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if not _serial_available():
            return {"status": "error: pyserial not installed (pip install pyserial)"}
        import serial
        data     = inputs.get("data")
        port     = inputs.get("port")     or "COM3"
        baud     = int(inputs.get("baud") or 115200)
        encoding = (inputs.get("encoding") or "json").lower()
        newline  = bool(inputs.get("newline", True))

        if data is None:
            return {"status": "no data"}

        try:
            payload = _encode(data, encoding)
            if newline:
                payload += b"\n"
        except Exception as e:
            return {"status": f"encode error: {e}"}

        try:
            with serial.Serial(port, baud, timeout=1) as ser:
                ser.write(payload)
            return {"status": "ok"}
        except Exception as e:
            return {"status": f"error: {e}"}

    def export(self, iv, ov):
        return [], [f"# [{self.label}]: visualization nodes render inline - skipped in export"]


class SerialInputNode(BaseNode):
    """Read one line from a serial port (sensors, encoders, MCU feedback)."""
    type_name   = "io_serial_in"
    label       = "Serial Input"
    category    = "IO"
    subcategory = "Serial"
    description = (
        "Read one line from a serial port. Returns the raw string and, "
        "if parseable, a list of floats."
    )

    def _setup_ports(self) -> None:
        self.add_input("port",    PortType.STRING, default="COM3")
        self.add_input("baud",    PortType.INT,    default=115200)
        self.add_input("timeout", PortType.FLOAT,  default=1.0,
                       description="Read timeout in seconds")
        self.add_output("raw",    PortType.STRING, description="Raw line received")
        self.add_output("values", PortType.ANY,    description="Parsed list of floats (if possible)")
        self.add_output("status", PortType.STRING)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if not _serial_available():
            return {"raw": "", "values": None,
                    "status": "error: pyserial not installed (pip install pyserial)"}
        import serial
        port    = inputs.get("port")    or "COM3"
        baud    = int(inputs.get("baud") or 115200)
        timeout = float(inputs.get("timeout") or 1.0)
        try:
            with serial.Serial(port, baud, timeout=timeout) as ser:
                raw = ser.readline().decode("utf-8", errors="replace").strip()
            try:
                values = [float(v.strip()) for v in raw.split(",") if v.strip()]
            except ValueError:
                values = None
            return {"raw": raw, "values": values, "status": "ok"}
        except Exception as e:
            return {"raw": "", "values": None, "status": f"error: {e}"}

    def export(self, iv, ov):
        return [], [f"# [{self.label}]: visualization nodes render inline - skipped in export"]


class ListSerialPortsNode(BaseNode):
    """List available serial ports on the system."""
    type_name   = "io_serial_list"
    label       = "List Serial Ports"
    category    = "IO"
    subcategory = "Serial"
    description = "Return a comma-separated list of serial ports found on the system."

    def _setup_ports(self) -> None:
        self.add_output("ports",  PortType.STRING, description="Comma-separated port names")
        self.add_output("status", PortType.STRING)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if not _serial_available():
            return {"ports": "", "status": "error: pyserial not installed"}
        from serial.tools import list_ports
        ports = [p.device for p in list_ports.comports()]
        return {"ports": ", ".join(ports), "status": "ok"}

    def export(self, iv, ov):
        return [], [f"# [{self.label}]: visualization nodes render inline - skipped in export"]
