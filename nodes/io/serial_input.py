"""Serial Input node."""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


def _serial_available() -> bool:
    try:
        import serial  # noqa: F401
        return True
    except ImportError:
        return False


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
            # Try parsing CSV floats
            try:
                values = [float(v.strip()) for v in raw.split(",") if v.strip()]
            except ValueError:
                values = None
            return {"raw": raw, "values": values, "status": "ok"}
        except Exception as e:
            return {"raw": "", "values": None, "status": f"error: {e}"}

    def export(self, iv, ov):
        return [], [f"# [{self.label}]: visualization nodes render inline - skipped in export"]
