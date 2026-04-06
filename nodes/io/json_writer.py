"""JSON Writer node."""
from __future__ import annotations
import json
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


class JSONWriterNode(BaseNode):
    """Write data as JSON to a file (overwrite or append to a JSON-lines file)."""
    type_name   = "io_json_writer"
    label       = "JSON Writer"
    category    = "IO"
    subcategory = "File"
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

    def export(self, iv, ov):
        return [], [f"# [{self.label}]: visualization nodes render inline - skipped in export"]
