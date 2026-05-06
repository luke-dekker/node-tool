"""Re-export shim — FileWriteNode (kind dropdown) replaces CSV/JSON/TextLog."""
from nodes.io.file_write import FileWriteNode

CSVWriterNode  = FileWriteNode
JSONWriterNode = FileWriteNode
TextLogNode    = FileWriteNode

__all__ = ["FileWriteNode", "CSVWriterNode", "JSONWriterNode", "TextLogNode"]
