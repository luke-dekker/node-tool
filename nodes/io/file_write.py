"""Consolidated file-writer node — replaces CSVWriterNode, JSONWriterNode, TextLogNode.

Pick `kind`:
  csv   — append a row to a CSV file (uses `headers`, `overwrite`)
  json  — write JSON (mode=overwrite full dump | mode=append JSONL)
  text  — append a timestamped line (uses `message`, `timestamp`)

Output:
  status        — 'ok' or error string
  rows_written  — INT (csv only — total rows in file after write)
"""
from __future__ import annotations
import csv
import datetime
import json
import os
from typing import Any
from core.node import BaseNode, PortType


_KINDS = ["csv", "json", "text"]


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


def _flatten(lst):
    for item in lst:
        if isinstance(item, (list, tuple)):
            yield from _flatten(item)
        else:
            yield item


class FileWriteNode(BaseNode):
    type_name   = "io_file_write"
    label       = "File Write"
    category    = "IO"
    subcategory = "File"
    description = (
        "Append / overwrite a file. Pick `kind`:\n"
        "  csv   — append a row, optionally write `headers` once\n"
        "  json  — overwrite or JSONL-append `data` as JSON\n"
        "  text  — append `message` (+ optional `data`) with a timestamp"
    )

    def relevant_inputs(self, values):
        kind = (values.get("kind") or "csv").strip()
        if kind == "csv":  return ["kind", "path", "headers", "overwrite"]
        if kind == "json": return ["kind", "path", "json_mode", "indent"]
        if kind == "text": return ["kind", "path", "message", "timestamp"]
        return ["kind", "path"]

    def _setup_ports(self):
        self.add_input("data",       PortType.ANY,    default=None,
                       description="Data to write (list/tensor/scalar/string/dict)")
        self.add_input("kind",       PortType.STRING, "csv", choices=_KINDS)
        self.add_input("path",       PortType.STRING, "output.txt")
        # csv-only
        self.add_input("headers",    PortType.STRING, "", optional=True)
        self.add_input("overwrite",  PortType.BOOL,   False, optional=True)
        # json-only
        self.add_input("json_mode",  PortType.STRING, "overwrite",
                       choices=["overwrite", "append"], optional=True)
        self.add_input("indent",     PortType.INT,    2, optional=True)
        # text-only
        self.add_input("message",    PortType.STRING, "", optional=True)
        self.add_input("timestamp",  PortType.BOOL,   True, optional=True)
        self.add_output("status",       PortType.STRING)
        self.add_output("rows_written", PortType.INT)

    def execute(self, inputs):
        out  = {"status": "", "rows_written": 0}
        kind = (inputs.get("kind") or "csv").strip()
        path = inputs.get("path") or "output.txt"
        try:
            if kind == "csv":
                data = _to_plain(inputs.get("data"))
                if data is None:
                    return out | {"status": "no data"}
                row = list(_flatten(data)) if isinstance(data, (list, tuple)) else [data]
                headers = inputs.get("headers") or ""
                overwrite = bool(inputs.get("overwrite", False))
                mode = "w" if overwrite or not os.path.exists(path) else "a"
                write_header = (headers.strip() and
                                (overwrite or not os.path.exists(path)
                                 or os.path.getsize(path) == 0))
                with open(path, mode, newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if write_header:
                        writer.writerow([h.strip() for h in headers.split(",") if h.strip()])
                    writer.writerow(row)
                with open(path, "r", encoding="utf-8") as f:
                    out["rows_written"] = sum(1 for _ in f)
                out["status"] = "ok"
                return out
            if kind == "json":
                data = _to_plain(inputs.get("data"))
                if data is None:
                    return out | {"status": "no data"}
                mode   = (inputs.get("json_mode") or "overwrite").lower()
                indent = int(inputs.get("indent") or 2)
                if mode == "append":
                    with open(path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(data) + "\n")
                else:
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=indent)
                out["status"] = "ok"
                return out
            if kind == "text":
                msg  = inputs.get("message") or ""
                data = _to_plain(inputs.get("data"))
                ts   = bool(inputs.get("timestamp", True))
                parts = []
                if ts:
                    parts.append(datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]"))
                if msg:
                    parts.append(msg)
                if data is not None:
                    parts.append(str(data))
                line = " ".join(parts) + "\n"
                with open(path, "a", encoding="utf-8") as f:
                    f.write(line)
                out["status"] = "ok"
                return out
            return out | {"status": f"unknown kind {kind!r}"}
        except Exception as exc:
            return out | {"status": f"error: {exc}"}

    def export(self, iv, ov):
        return [], [f"# [{self.label}]: side-effect node — skipped in export"]
