"""CSV Writer node."""
from __future__ import annotations
import csv
import os
from typing import Any
from core.node import BaseNode, PortType


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


def _flatten(lst):
    for item in lst:
        if isinstance(item, (list, tuple)):
            yield from _flatten(item)
        else:
            yield item


class CSVWriterNode(BaseNode):
    """Append a row of data to a CSV file on every execution."""
    type_name   = "io_csv_writer"
    label       = "CSV Writer"
    category    = "IO"
    subcategory = "File"
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

    def export(self, iv, ov):
        return [], [f"# [{self.label}]: visualization nodes render inline - skipped in export"]
