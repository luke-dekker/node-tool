"""Re-export shim — NetworkSendNode (kind dropdown) replaces HTTP/MQTT/WS/ROS2."""
from nodes.io.network_send import NetworkSendNode

HTTPPostNode      = NetworkSendNode
MQTTPublishNode   = NetworkSendNode
WebSocketSendNode = NetworkSendNode
ROSPublishNode    = NetworkSendNode

__all__ = ["NetworkSendNode", "HTTPPostNode", "MQTTPublishNode",
           "WebSocketSendNode", "ROSPublishNode"]
