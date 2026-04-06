"""MQTT Publish node."""
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


class MQTTPublishNode(BaseNode):
    """Publish data to an MQTT topic (paho-mqtt)."""
    type_name   = "io_mqtt_publish"
    label       = "MQTT Publish"
    category    = "IO"
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

    def export(self, iv, ov):
        return [], [f"# [{self.label}]: visualization nodes render inline - skipped in export"]
