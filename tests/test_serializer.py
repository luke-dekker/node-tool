import os
import json
import tempfile
import pytest


def test_save_load_roundtrip():
    from core.graph import Graph
    from core.io import Serializer
    from nodes.math import MathNode
    from nodes.data import FloatConstNode

    graph = Graph()
    a = FloatConstNode(); a.inputs["Value"].default_value = 7.0
    b = FloatConstNode(); b.inputs["Value"].default_value = 3.0
    add = MathNode()  # default Op="add"
    graph.add_node(a); graph.add_node(b); graph.add_node(add)
    graph.add_connection(a.id, "Value", add.id, "A")
    graph.add_connection(b.id, "Value", add.id, "B")

    positions = {a.id: [10, 20], b.id: [10, 80], add.id: [200, 50]}

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        Serializer.save(graph, positions, path)
        graph2, pos2 = Serializer.load(path)
        assert len(graph2.nodes) == 3
        assert len(graph2.connections) == 2
        outputs, _ = graph2.execute()
        math_node = next(n for n in graph2.nodes.values() if n.type_name == "math")
        assert outputs[math_node.id]["Result"] == 10.0
    finally:
        os.unlink(path)


def test_unknown_type_raises():
    from core.io import Serializer
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump({
            "version": 1,
            "nodes": [{"id": "x", "type_name": "nonexistent_xyz", "pos": [0, 0], "inputs": {}}],
            "connections": []
        }, f)
        path = f.name
    try:
        with pytest.raises(KeyError):
            Serializer.load(path)
    finally:
        os.unlink(path)
