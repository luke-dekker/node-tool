import os
import json
import tempfile
import pytest


def test_save_load_roundtrip():
    from core.graph import Graph
    from core.serializer import Serializer
    from nodes.code.python_node import PythonNode

    graph = Graph()
    a = PythonNode(); a.inputs["code"].default_value = "result = 7.0"
    b = PythonNode(); b.inputs["code"].default_value = "result = 3.0"
    add = PythonNode(); add.inputs["code"].default_value = "result = a + b"
    graph.add_node(a); graph.add_node(b); graph.add_node(add)
    graph.add_connection(a.id, "result", add.id, "a")
    graph.add_connection(b.id, "result", add.id, "b")

    positions = {a.id: [10, 20], b.id: [10, 80], add.id: [200, 50]}

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        Serializer.save(graph, positions, path)
        graph2, pos2 = Serializer.load(path)
        assert len(graph2.nodes) == 3
        assert len(graph2.connections) == 2
        outputs, _, _ = graph2.execute()
        add_node = next(n for n in graph2.nodes.values()
                        if n.inputs["code"].default_value == "result = a + b")
        assert outputs[add_node.id]["result"] == 10.0
    finally:
        os.unlink(path)


def test_unknown_type_raises():
    from core.serializer import Serializer
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump({
            "version": 1,
            "nodes": [{"id": "x", "type_name": "nonexistent_xyz", "pos": [0, 0], "inputs": {}}],
            "connections": []
        }, f)
        path = f.name
    try:
        graph, _ = Serializer.load(path)
        assert "nonexistent_xyz" not in [n.type_name for n in graph.nodes.values()]
    finally:
        os.unlink(path)
