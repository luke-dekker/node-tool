"""WebSocket Send node."""
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


def _flatten_iter(lst):
    for item in lst:
        if isinstance(item, (list, tuple)):
            yield from _flatten_iter(item)
        else:
            yield item


def _encode_net(data: Any, encoding: str) -> str:
    data = _to_serialisable(data)
    if encoding == "json":
        return json.dumps(data)
    if encoding == "csv":
        if isinstance(data, (list, tuple)):
            return ",".join(str(v) for v in _flatten_iter(data))
        return str(data)
    return str(data)


class WebSocketSendNode(BaseNode):
    """Send data over a WebSocket connection (one-shot)."""
    type_name   = "io_websocket_send"
    label       = "WebSocket Send"
    category    = "IO"
    subcategory = "Network"
    description = (
        "Send data as JSON text over a WebSocket. "
        "Requires: pip install websockets"
    )

    def _setup_ports(self) -> None:
        self.add_input("data",     PortType.ANY,    default=None)
        self.add_input("uri",      PortType.STRING, default="ws://localhost:8765")
        self.add_input("encoding", PortType.STRING, default="json")
        self.add_output("status",  PortType.STRING)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        try:
            import websockets
            import asyncio
        except ImportError:
            return {"status": "error: pip install websockets"}

        data     = inputs.get("data")
        uri      = inputs.get("uri") or "ws://localhost:8765"
        encoding = (inputs.get("encoding") or "json").lower()
        payload  = _encode_net(data, encoding)

        async def _send():
            async with websockets.connect(uri) as ws:
                await ws.send(payload)

        try:
            asyncio.get_event_loop().run_until_complete(_send())
            return {"status": "ok"}
        except Exception as e:
            return {"status": f"error: {e}"}

    def export(self, iv, ov):
        return [], [f"# [{self.label}]: visualization nodes render inline - skipped in export"]
