"""IO nodes — consolidated 12 → 4 (active).

- FileWriteNode      — kind: csv | json | text  (replaces CSV/JSON/TextLog)
- NetworkSendNode    — kind: http | mqtt | ws | ros2  (replaces 4 publishers)
- WebcamCaptureNode  — webcam frame capture (kept; distinct semantics)
- Serial nodes       — kept as 3 (in/out/list) in serial_nodes.py (test
                       monkeypatching pattern depends on the per-class layout)

Old class names alias to the consolidated ones. Caller sets `kind` on the
instance to recover specific behavior.
"""
from nodes.io.file_write    import FileWriteNode
from nodes.io.network_send  import NetworkSendNode
from nodes.io.webcam        import WebcamCaptureNode
from nodes.io.serial_nodes  import (
    SerialOutputNode, SerialInputNode, ListSerialPortsNode,
)

# Back-compat — FileWriteNode kinds
CSVWriterNode  = FileWriteNode    # set kind="csv" (default)
JSONWriterNode = FileWriteNode    # set kind="json"
TextLogNode    = FileWriteNode    # set kind="text"

# Back-compat — NetworkSendNode kinds
HTTPPostNode      = NetworkSendNode    # set kind="http" (default)
MQTTPublishNode   = NetworkSendNode    # set kind="mqtt"
WebSocketSendNode = NetworkSendNode    # set kind="ws"
ROSPublishNode    = NetworkSendNode    # set kind="ros2"

__all__ = [
    "FileWriteNode", "NetworkSendNode", "WebcamCaptureNode",
    "SerialOutputNode", "SerialInputNode", "ListSerialPortsNode",
    "CSVWriterNode", "JSONWriterNode", "TextLogNode",
    "HTTPPostNode", "MQTTPublishNode", "WebSocketSendNode", "ROSPublishNode",
]
