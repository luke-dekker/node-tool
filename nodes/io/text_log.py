"""Text Log node."""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


def _to_plain(data: Any) -> Any:
    try:
        import torch
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().tolist()
    except ImportError:
        pass
    try:
        import numpy as np
        if isinstance(data, np.ndarray):
            return data.tolist()
    except ImportError:
        pass
    return data


class TextLogNode(BaseNode):
    """Append a formatted text line to a log file."""
    type_name   = "io_text_log"
    label       = "Text Log"
    category    = "IO"
    subcategory = "File"
    description = "Append a timestamped line to a plain-text log file."

    def _setup_ports(self) -> None:
        self.add_input("message",   PortType.STRING, default="",
                       description="Text message to log")
        self.add_input("data",      PortType.ANY,    default=None,
                       description="Optional extra data appended after message")
        self.add_input("path",      PortType.STRING, default="run.log")
        self.add_input("timestamp", PortType.BOOL,   default=True)
        self.add_output("status",   PortType.STRING)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import datetime
        msg  = inputs.get("message") or ""
        data = _to_plain(inputs.get("data"))
        path = inputs.get("path") or "run.log"
        ts   = bool(inputs.get("timestamp", True))

        parts = []
        if ts:
            parts.append(datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]"))
        if msg:
            parts.append(msg)
        if data is not None:
            parts.append(str(data))
        line = " ".join(parts)

        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            return {"status": "ok"}
        except Exception as e:
            return {"status": f"error: {e}"}

    def export(self, iv, ov):
        return [], [f"# [{self.label}]: visualization nodes render inline - skipped in export"]
