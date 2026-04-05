"""Network IO nodes: HTTP POST, MQTT publish, WebSocket send, ROS2 publish.

Dependencies (all optional — nodes degrade gracefully if missing):
  pip install requests paho-mqtt websockets
  rclpy requires a ROS2 installation.
"""

from __future__ import annotations
import json
from typing import Any
from core.node import BaseNode
from core.node import PortType

CATEGORY    = "IO"


def _to_serialisable(data: Any) -> Any:
    """Recursively convert tensors / numpy arrays to plain Python for JSON."""
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


# ── HTTP POST ──────────────────────────────────────────────────────────────────

class HTTPPostNode(BaseNode):
    """POST data as JSON to an HTTP endpoint."""
    type_name   = "io_http_post"
    label       = "HTTP POST"
    category    = CATEGORY
    subcategory = "Network"
    description = (
        "POST data (tensor, list, dict, or string) as JSON to a URL. "
        "Requires: pip install requests"
    )

    def _setup_ports(self) -> None:
        self.add_input("data",    PortType.ANY,    default=None)
        self.add_input("url",     PortType.STRING, default="http://localhost:8000/data")
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
        url     = inputs.get("url") or "http://localhost:8000/data"
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


# ── MQTT Publish ───────────────────────────────────────────────────────────────

class MQTTPublishNode(BaseNode):
    """Publish data to an MQTT topic (paho-mqtt)."""
    type_name   = "io_mqtt_publish"
    label       = "MQTT Publish"
    category    = CATEGORY
    subcategory = "Network"
    description = (
        "Publish data to an MQTT broker topic. "
        "Requires: pip install paho-mqtt"
    )

    def _setup_ports(self) -> None:
        self.add_input("data",     PortType.ANY,    default=None)
        self.add_input("broker",   PortType.STRING, default="localhost")
        self.add_input("port",     PortType.INT,    default=1883)
        self.add_input("topic",    PortType.STRING, default="robot/inference")
        self.add_input("encoding", PortType.STRING, default="json",
                       description="json | csv | raw")
        self.add_input("qos",      PortType.INT,    default=0)
        self.add_output("status",  PortType.STRING)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        try:
            import paho.mqtt.publish as publish
        except ImportError:
            return {"status": "error: pip install paho-mqtt"}

        data     = inputs.get("data")
        broker   = inputs.get("broker") or "localhost"
        port     = int(inputs.get("port") or 1883)
        topic    = inputs.get("topic")   or "robot/inference"
        encoding = (inputs.get("encoding") or "json").lower()
        qos      = int(inputs.get("qos") or 0)

        payload = _encode_net(data, encoding)
        try:
            publish.single(topic, payload=payload, qos=qos,
                           hostname=broker, port=port)
            return {"status": "ok"}
        except Exception as e:
            return {"status": f"error: {e}"}


# ── WebSocket Send ─────────────────────────────────────────────────────────────

class WebSocketSendNode(BaseNode):
    """Send data over a WebSocket connection (one-shot)."""
    type_name   = "io_websocket_send"
    label       = "WebSocket Send"
    category    = CATEGORY
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


# ── ROS2 Publish ───────────────────────────────────────────────────────────────

class ROSPublishNode(BaseNode):
    """Publish a Float32MultiArray message to a ROS2 topic."""
    type_name   = "io_ros_publish"
    label       = "ROS2 Publish"
    category    = CATEGORY
    subcategory = "Network"
    description = (
        "Publish tensor/list data as std_msgs/Float32MultiArray to a ROS2 topic. "
        "Requires a ROS2 installation and sourced environment."
    )

    def _setup_ports(self) -> None:
        self.add_input("data",     PortType.ANY,    default=None)
        self.add_input("topic",    PortType.STRING, default="/inference/output")
        self.add_input("node_name",PortType.STRING, default="node_tool_publisher")
        self.add_output("status",  PortType.STRING)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        try:
            import rclpy
            from std_msgs.msg import Float32MultiArray
        except ImportError:
            return {"status": "error: ROS2 / rclpy not available"}

        data  = _to_serialisable(inputs.get("data"))
        topic = inputs.get("topic")     or "/inference/output"
        name  = inputs.get("node_name") or "node_tool_publisher"

        # Flatten to list of floats
        if isinstance(data, (list, tuple)):
            flat = list(_flatten_iter(data))
        elif isinstance(data, (int, float)):
            flat = [float(data)]
        else:
            return {"status": f"error: unsupported data type {type(data)}"}

        try:
            if not rclpy.ok():
                rclpy.init()
            node = rclpy.create_node(name)
            pub  = node.create_publisher(Float32MultiArray, topic, 10)
            msg  = Float32MultiArray()
            msg.data = [float(v) for v in flat]
            pub.publish(msg)
            node.destroy_node()
            return {"status": f"published {len(flat)} values to {topic}"}
        except Exception as e:
            return {"status": f"error: {e}"}


# ── helpers ────────────────────────────────────────────────────────────────────

def _encode_net(data: Any, encoding: str) -> str:
    data = _to_serialisable(data)
    if encoding == "json":
        return json.dumps(data)
    if encoding == "csv":
        if isinstance(data, (list, tuple)):
            return ",".join(str(v) for v in _flatten_iter(data))
        return str(data)
    return str(data)


def _flatten_iter(lst):
    for item in lst:
        if isinstance(item, (list, tuple)):
            yield from _flatten_iter(item)
        else:
            yield item
