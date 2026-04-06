"""HTTP POST node."""
from __future__ import annotations
import json
from typing import Any
from core.node import BaseNode, PortType


def _to_serialisable(data: Any) -> Any:
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


class HTTPPostNode(BaseNode):
    """POST data as JSON to an HTTP endpoint."""
    type_name   = "io_http_post"
    label       = "HTTP POST"
    category    = "IO"
    subcategory = "Network"
    description = (
        "POST data (tensor, list, dict, or string) as JSON to a URL. "
        "Requires: pip install requests"
    )

    def _setup_ports(self) -> None:
        self.add_input("data",    PortType.ANY,    default=None)
        self.add_input("url",     PortType.STRING, default="http://localhost:8080/data")
        self.add_input("headers", PortType.STRING, default="",
                       description="Extra headers as JSON string, e.g. {\"X-Key\":\"val\"}")
        self.add_input("timeout", PortType.FLOAT,  default=5.0)
        self.add_output("status_code", PortType.INT,    description="HTTP status code")
        self.add_output("response",    PortType.STRING, description="Response body text")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        try:
            import requests
        except ImportError:
            return {"status_code": 0, "response": "error: pip install requests"}

        data    = _to_serialisable(inputs.get("data"))
        url     = inputs.get("url") or "http://localhost:8080/data"
        timeout = float(inputs.get("timeout") or 5.0)
        headers = {"Content-Type": "application/json"}
        hdr_str = inputs.get("headers") or ""
        if hdr_str.strip():
            try:
                headers.update(json.loads(hdr_str))
            except Exception:
                pass
        try:
            resp = requests.post(url, json=data, headers=headers, timeout=timeout)
            return {"status_code": resp.status_code, "response": resp.text}
        except Exception as e:
            return {"status_code": 0, "response": f"error: {e}"}

    def export(self, iv, ov):
        return [], [f"# [{self.label}]: visualization nodes render inline - skipped in export"]
