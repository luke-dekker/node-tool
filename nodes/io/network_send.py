"""Consolidated network-send node — replaces HTTPPostNode, MQTTPublishNode,
WebSocketSendNode, ROSPublishNode.

Pick `kind`:
  http   — POST data as JSON to `url` (requests)
  mqtt   — Publish to `topic` on `broker`:`port` (paho-mqtt)
  ws     — Send to WebSocket `uri` (websockets, one-shot)
  ros2   — Publish Float32MultiArray to `topic` (rclpy)

Output:
  status    — 'ok' or error string
  response  — string (http: response body)
"""
from __future__ import annotations
import json
from typing import Any
from core.node import BaseNode, PortType


_KINDS = ["http", "mqtt", "ws", "ros2"]


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


def _flatten(lst):
    for item in lst:
        if isinstance(item, (list, tuple)):
            yield from _flatten(item)
        else:
            yield item


def _encode(data, encoding: str) -> str:
    data = _to_serialisable(data)
    if encoding == "json":
        return json.dumps(data)
    if encoding == "csv":
        if isinstance(data, (list, tuple)):
            return ",".join(str(v) for v in _flatten(data))
        return str(data)
    return str(data)


class NetworkSendNode(BaseNode):
    type_name   = "io_network_send"
    label       = "Network Send"
    category    = "IO"
    subcategory = "Network"
    description = (
        "One-shot send to a network endpoint. Pick `kind`:\n"
        "  http  — POST to `url` (requests)\n"
        "  mqtt  — publish to `topic` on `broker`:`port` (paho-mqtt)\n"
        "  ws    — send to WebSocket `uri` (websockets)\n"
        "  ros2  — publish to ROS2 `topic` as Float32MultiArray"
    )

    def relevant_inputs(self, values):
        kind = (values.get("kind") or "http").strip()
        if kind == "http": return ["kind", "url", "headers", "timeout"]
        if kind == "mqtt": return ["kind", "topic", "broker", "port", "encoding"]
        if kind == "ws":   return ["kind", "uri", "encoding"]
        if kind == "ros2": return ["kind", "topic", "node_name"]
        return ["kind"]

    def _setup_ports(self):
        self.add_input("data",     PortType.ANY,    default=None)
        self.add_input("kind",     PortType.STRING, "http", choices=_KINDS)
        # http
        self.add_input("url",      PortType.STRING, "http://localhost:8080/data", optional=True)
        self.add_input("headers",  PortType.STRING, "", optional=True)
        self.add_input("timeout",  PortType.FLOAT,  5.0, optional=True)
        # mqtt
        self.add_input("broker",   PortType.STRING, "localhost", optional=True)
        self.add_input("port",     PortType.INT,    1883, optional=True)
        # mqtt + ws
        self.add_input("topic",    PortType.STRING, "node_tool/data", optional=True)
        self.add_input("uri",      PortType.STRING, "ws://localhost:8765", optional=True)
        self.add_input("encoding", PortType.STRING, "json",
                       choices=["json", "csv", "raw"], optional=True)
        # ros2
        self.add_input("node_name", PortType.STRING, "node_tool_publisher", optional=True)
        self.add_output("status",   PortType.STRING)
        self.add_output("response", PortType.STRING)

    def execute(self, inputs):
        out = {"status": "", "response": ""}
        kind = (inputs.get("kind") or "http").strip()
        data = _to_serialisable(inputs.get("data"))
        if data is None:
            return out | {"status": "no data"}
        try:
            if kind == "http":
                try:
                    import requests
                except ImportError:
                    return out | {"status": "error: pip install requests"}
                hdr = {}
                if (inputs.get("headers") or "").strip():
                    try: hdr = json.loads(inputs["headers"])
                    except Exception: pass
                r = requests.post(inputs.get("url") or "http://localhost:8080/data",
                                  json=data, headers=hdr,
                                  timeout=float(inputs.get("timeout", 5.0)))
                return {"status": "ok", "response": r.text[:1000]}

            if kind == "mqtt":
                try:
                    import paho.mqtt.publish as mqtt_pub
                except ImportError:
                    return out | {"status": "error: pip install paho-mqtt"}
                payload = _encode(data, (inputs.get("encoding") or "json").lower())
                mqtt_pub.single(
                    inputs.get("topic") or "node_tool/data",
                    payload=payload,
                    hostname=inputs.get("broker") or "localhost",
                    port=int(inputs.get("port") or 1883),
                )
                return out | {"status": "ok"}

            if kind == "ws":
                try:
                    import asyncio
                    import websockets
                except ImportError:
                    return out | {"status": "error: pip install websockets"}
                uri     = inputs.get("uri") or "ws://localhost:8765"
                payload = _encode(data, (inputs.get("encoding") or "json").lower())
                async def _send():
                    async with websockets.connect(uri) as ws:
                        await ws.send(payload)
                asyncio.run(_send())
                return out | {"status": "ok"}

            if kind == "ros2":
                try:
                    import rclpy
                    from std_msgs.msg import Float32MultiArray
                except ImportError:
                    return out | {"status": "error: ROS2 / rclpy not available"}
                if isinstance(data, (list, tuple)):
                    flat = [float(v) for v in _flatten(data)]
                else:
                    flat = [float(data)]
                if not rclpy.ok():
                    rclpy.init(args=None)
                node = rclpy.create_node(inputs.get("node_name") or "node_tool_publisher")
                pub  = node.create_publisher(Float32MultiArray,
                                             inputs.get("topic") or "/inference/output", 10)
                msg = Float32MultiArray(); msg.data = flat
                pub.publish(msg)
                node.destroy_node()
                return out | {"status": "ok"}

            return out | {"status": f"unknown kind {kind!r}"}
        except Exception as exc:
            return out | {"status": f"error: {exc}"}

    def export(self, iv, ov):
        return [], [f"# [{self.label}]: side-effect node — skipped in export"]
