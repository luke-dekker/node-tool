"""Serial protocol nodes — build/parse framed packets for hardware communication.

Common robotics protocols use framed packets:
    [HEADER] [LENGTH] [COMMAND] [PAYLOAD...] [CRC/CHECKSUM]

These nodes let you build and parse such frames without writing protocol
code — just wire the fields together. Works with the Serial Output node
from the IO plugin for sending, and Serial Input for receiving.
"""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


def _crc8(data: bytes) -> int:
    """Simple CRC-8 (polynomial 0x07)."""
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) if (crc & 0x80) else (crc << 1)
            crc &= 0xFF
    return crc


class FrameBuilderNode(BaseNode):
    """Build a framed serial packet from fields."""
    type_name   = "rob_frame_build"
    label       = "Frame Builder"
    category    = "Control"
    subcategory = "Serial"
    description = (
        "Build a framed packet: [header] [length] [command] [payload] [crc8]. "
        "Wire the output bytes to a Serial Output node."
    )

    def _setup_ports(self) -> None:
        self.add_input("header",  PortType.INT, 0xAA,
                       description="Start-of-frame marker byte")
        self.add_input("command", PortType.INT, 0x01)
        self.add_input("payload", PortType.STRING, "",
                       description="Payload as comma-separated byte values (e.g. '10,20,30')")
        self.add_output("frame",    PortType.STRING,
                        description="Hex string of the built frame")
        self.add_output("raw_bytes", PortType.ANY,
                        description="bytes object ready for serial.write()")
        self.add_output("length",   PortType.INT)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        header = int(inputs.get("header") or 0xAA) & 0xFF
        cmd    = int(inputs.get("command") or 0x01) & 0xFF
        payload_str = str(inputs.get("payload") or "")
        payload = bytes()
        if payload_str.strip():
            try:
                payload = bytes(int(x.strip()) & 0xFF
                                for x in payload_str.split(",") if x.strip())
            except ValueError:
                payload = payload_str.encode("utf-8")

        body = bytes([cmd]) + payload
        length = len(body)
        frame_data = bytes([header, length]) + body
        crc = _crc8(frame_data)
        frame = frame_data + bytes([crc])

        return {
            "frame": frame.hex(" "),
            "raw_bytes": frame,
            "length": len(frame),
        }


class FrameParserNode(BaseNode):
    """Parse a framed serial packet into fields."""
    type_name   = "rob_frame_parse"
    label       = "Frame Parser"
    category    = "Control"
    subcategory = "Serial"
    description = (
        "Parse a framed packet: [header] [length] [command] [payload] [crc8]. "
        "Wire Serial Input bytes into this node."
    )

    def _setup_ports(self) -> None:
        self.add_input("data",     PortType.STRING, "",
                       description="Hex string or raw bytes to parse")
        self.add_input("header",   PortType.INT, 0xAA,
                       description="Expected header byte")
        self.add_output("command",  PortType.INT)
        self.add_output("payload",  PortType.STRING,
                        description="Payload as comma-separated byte values")
        self.add_output("valid",    PortType.BOOL,
                        description="True if header matched and CRC is correct")
        self.add_output("error",    PortType.STRING)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        raw = inputs.get("data") or ""
        expected_hdr = int(inputs.get("header") or 0xAA) & 0xFF
        empty = {"command": 0, "payload": "", "valid": False, "error": ""}

        # Parse hex string to bytes
        try:
            if isinstance(raw, bytes):
                data = raw
            elif isinstance(raw, str):
                data = bytes.fromhex(raw.replace(" ", ""))
            else:
                return {**empty, "error": "Unknown data format"}
        except ValueError as e:
            return {**empty, "error": f"Hex parse error: {e}"}

        if len(data) < 4:
            return {**empty, "error": f"Frame too short ({len(data)} bytes, need >= 4)"}

        hdr = data[0]
        if hdr != expected_hdr:
            return {**empty, "error": f"Header mismatch: got 0x{hdr:02X}, expected 0x{expected_hdr:02X}"}

        length = data[1]
        if len(data) < 2 + length + 1:
            return {**empty, "error": "Frame truncated"}

        body = data[2:2 + length]
        crc_received = data[2 + length]
        crc_calc = _crc8(data[:2 + length])

        if crc_received != crc_calc:
            return {**empty, "error": f"CRC mismatch: got 0x{crc_received:02X}, expected 0x{crc_calc:02X}"}

        cmd = body[0] if body else 0
        payload = ",".join(str(b) for b in body[1:]) if len(body) > 1 else ""
        return {"command": cmd, "payload": payload, "valid": True, "error": ""}


class MapValueNode(BaseNode):
    """Map a value from one range to another (like Arduino's map())."""
    type_name   = "rob_map_value"
    label       = "Map Value"
    category    = "Signal"
    description = "Map a value from [in_min, in_max] to [out_min, out_max]. Like Arduino's map()."

    def _setup_ports(self) -> None:
        self.add_input("value",   PortType.FLOAT, 0.0)
        self.add_input("in_min",  PortType.FLOAT, 0.0)
        self.add_input("in_max",  PortType.FLOAT, 1023.0)
        self.add_input("out_min", PortType.FLOAT, 0.0)
        self.add_input("out_max", PortType.FLOAT, 255.0)
        self.add_input("clamp",   PortType.BOOL, True)
        self.add_output("result", PortType.FLOAT)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        val = float(inputs.get("value") or 0)
        i0  = float(inputs.get("in_min") or 0)
        i1  = float(inputs.get("in_max") or 1023)
        o0  = float(inputs.get("out_min") or 0)
        o1  = float(inputs.get("out_max") or 255)
        if abs(i1 - i0) < 1e-9:
            result = o0
        else:
            result = (val - i0) / (i1 - i0) * (o1 - o0) + o0
        if bool(inputs.get("clamp", True)):
            lo, hi = min(o0, o1), max(o0, o1)
            result = max(lo, min(hi, result))
        return {"result": result}
