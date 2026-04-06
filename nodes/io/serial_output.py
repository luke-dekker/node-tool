"""Serial Output node."""
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

        # Convert data to bytes
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
