"""File-based IO nodes: CSV writer, JSON writer, text log."""

from __future__ import annotations
import csv
import json
import os
from typing import Any
from core.node import BaseNode
from core.node import PortType

CATEGORY    = "IO"
SUBCATEGORY = "File"


def _to_plain(data: Any) -> Any:
    """Convert tensors / numpy arrays to plain Python."""
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


class CSVWriterNode(BaseNode):
    """Append a row of data to a CSV file on every execution."""
    type_name   = "io_csv_writer"
    label       = "CSV Writer"
    category    = CATEGORY
    subcategory = SUBCATEGORY
    description = (
        "Append a row to a CSV file each time this node executes. "
        "Useful for logging inference outputs, sensor readings, or metrics."
    )

    def _setup_ports(self) -> None:
        self.add_input("data",      PortType.ANY,    default=None,
                       description="List, tensor, or scalar to write as one row")
        self.add_input("path",      PortType.STRING, default="output.csv")
        self.add_input("headers",   PortType.STRING, default="",
                       description="Comma-separated column headers (written on first row only)")
        self.add_input("overwrite", PortType.BOOL,   default=False,
                       description="If True, overwrite file; if False, append")
        self.add_output("rows_written", PortType.INT,    description="Total rows in file after write")
        self.add_output("status",       PortType.STRING)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        data      = _to_plain(inputs.get("data"))
        path      = inputs.get("path")    or "output.csv"
        headers   = inputs.get("headers") or ""
        overwrite = bool(inputs.get("overwrite", False))

        if data is None:
            return {"rows_written": 0, "status": "no data"}

        # Flatten to a single row
        if not isinstance(data, (list, tuple)):
            row = [data]
        else:
            row = list(_flatten(data))

        mode = "w" if overwrite or not os.path.exists(path) else "a"
        write_header = (headers.strip() and
                        (overwrite or not os.path.exists(path) or os.path.getsize(path) == 0))
        try:
            with open(path, mode, newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow([h.strip() for h in headers.split(",") if h.strip()])
                writer.writerow(row)
            # count rows
            with open(path, "r", encoding="utf-8") as f:
                total = sum(1 for _ in f)
            return {"rows_written": total, "status": "ok"}
        except Exception as e:
            return {"rows_written": 0, "status": f"error: {e}"}


class JSONWriterNode(BaseNode):
    """Write data as JSON to a file (overwrite or append to a JSON-lines file)."""
    type_name   = "io_json_writer"
    label       = "JSON Writer"
    category    = CATEGORY
    subcategory = SUBCATEGORY
    description = (
        "Write data as JSON. Mode 'overwrite' replaces the file; "
        "'append' adds one JSON object per line (JSON-Lines format)."
    )

    def _setup_ports(self) -> None:
        self.add_input("data",   PortType.ANY,    default=None)
        self.add_input("path",   PortType.STRING, default="output.json")
        self.add_input("mode",   PortType.STRING, default="overwrite",
                       description="overwrite | append")
        self.add_input("indent", PortType.INT,    default=2,
                       description="JSON indent (ignored in append/jsonl mode)")
        self.add_output("status", PortType.STRING)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        data   = _to_plain(inputs.get("data"))
        path   = inputs.get("path")  or "output.json"
        mode   = (inputs.get("mode") or "overwrite").lower()
        indent = int(inputs.get("indent") or 2)

        if data is None:
            return {"status": "no data"}
        try:
            if mode == "append":
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(data) + "\n")
            else:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=indent)
            return {"status": "ok"}
        except Exception as e:
            return {"status": f"error: {e}"}


class TextLogNode(BaseNode):
    """Append a formatted text line to a log file."""
    type_name   = "io_text_log"
    label       = "Text Log"
    category    = CATEGORY
    subcategory = SUBCATEGORY
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


# ── helper ────────────────────────────────────────────────────────────────────

def _flatten(lst):
    for item in lst:
        if isinstance(item, (list, tuple)):
            yield from _flatten(item)
        else:
            yield item
