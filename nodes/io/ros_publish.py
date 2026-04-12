"""ROS2 Publish node."""
from __future__ import annotations
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


class ROSPublishNode(BaseNode):
    """Publish a Float32MultiArray message to a ROS2 topic."""
    type_name   = "io_ros_publish"
    label       = "ROS2 Publish"
    category    = "Robotics"
    subcategory = "Comm"
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

    def export(self, iv, ov):
        return [], [f"# [{self.label}]: visualization nodes render inline - skipped in export"]
