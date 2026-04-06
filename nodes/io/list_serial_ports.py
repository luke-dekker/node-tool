"""List Serial Ports node."""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


def _serial_available() -> bool:
    try:
        import serial  # noqa: F401
        return True
    except ImportError:
        return False


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
