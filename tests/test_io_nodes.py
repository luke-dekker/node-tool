"""Tests for IO nodes — serial, network, file."""
import sys, os, tempfile, pathlib, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


# ── Registry ──────────────────────────────────────────────────────────────────

def test_io_nodes_registered():
    from nodes import NODE_REGISTRY
    expected = [
        "io_serial_out", "io_serial_in", "io_serial_list",
        "io_http_post", "io_mqtt_publish", "io_websocket_send", "io_ros_publish",
        "io_csv_writer", "io_json_writer", "io_text_log",
    ]
    for tn in expected:
        assert tn in NODE_REGISTRY, f"{tn} not in registry"


# ── Serial — graceful degradation ─────────────────────────────────────────────

def test_serial_out_no_pyserial(monkeypatch):
    """If pyserial is not installed, node returns an error status without crashing."""
    import nodes.io.serial_nodes as sn
    monkeypatch.setattr(sn, "_serial_available", lambda: False)
    from nodes.io.serial_nodes import SerialOutputNode
    result = SerialOutputNode().execute({"data": [1.0, 2.0], "port": "COM3", "baud": 9600,
                                         "encoding": "json", "newline": True})
    assert "error" in result["status"]


def test_serial_in_no_pyserial(monkeypatch):
    import nodes.io.serial_nodes as sn
    monkeypatch.setattr(sn, "_serial_available", lambda: False)
    from nodes.io.serial_nodes import SerialInputNode
    result = SerialInputNode().execute({"port": "COM3", "baud": 9600, "timeout": 0.1})
    assert "error" in result["status"]


def test_serial_list_no_pyserial(monkeypatch):
    import nodes.io.serial_nodes as sn
    monkeypatch.setattr(sn, "_serial_available", lambda: False)
    from nodes.io.serial_nodes import ListSerialPortsNode
    result = ListSerialPortsNode().execute({})
    assert "error" in result["status"]


def test_serial_out_bad_port():
    """Connecting to a nonexistent port should return an error status, not raise."""
    from nodes.io.serial_nodes import SerialOutputNode
    try:
        import serial  # noqa
    except ImportError:
        pytest.skip("pyserial not installed")
    result = SerialOutputNode().execute({
        "data": [1.0, 2.0], "port": "BADPORT999", "baud": 9600,
        "encoding": "json", "newline": True
    })
    assert "error" in result["status"]


# ── Serial encoding helpers ────────────────────────────────────────────────────

def test_serial_encode_json():
    from nodes.io.serial_nodes import _encode
    import torch
    t = torch.tensor([1.0, 2.0, 3.0])
    result = _encode(t, "json")
    assert result == b"[1.0, 2.0, 3.0]"


def test_serial_encode_csv():
    from nodes.io.serial_nodes import _encode
    result = _encode([1.5, 2.5, 3.5], "csv")
    assert result == b"1.5,2.5,3.5"


def test_serial_encode_raw():
    from nodes.io.serial_nodes import _encode
    result = _encode(42, "raw")
    assert result == b"42"


# ── HTTP POST — no requests ────────────────────────────────────────────────────

def test_http_post_no_requests(monkeypatch):
    import builtins
    real_import = builtins.__import__
    def mock_import(name, *args, **kwargs):
        if name == "requests":
            raise ImportError("mocked")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", mock_import)
    from nodes.io.network_nodes import HTTPPostNode
    result = HTTPPostNode().execute({"data": [1, 2], "url": "http://x", "headers": "", "timeout": 1})
    assert "error" in result["response"]


def test_http_post_bad_url():
    """Connection refused → error status, no crash."""
    from nodes.io.network_nodes import HTTPPostNode
    try:
        import requests  # noqa
    except ImportError:
        pytest.skip("requests not installed")
    result = HTTPPostNode().execute({
        "data": {"x": 1}, "url": "http://127.0.0.1:19999/nope",
        "headers": "", "timeout": 0.5
    })
    assert result["status_code"] == 0
    assert "error" in result["response"]


# ── MQTT — no paho ────────────────────────────────────────────────────────────

def test_mqtt_no_paho(monkeypatch):
    import builtins
    real_import = builtins.__import__
    def mock_import(name, *args, **kwargs):
        if "paho" in name:
            raise ImportError("mocked")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", mock_import)
    from nodes.io.network_nodes import MQTTPublishNode
    result = MQTTPublishNode().execute({
        "data": [1.0], "broker": "localhost", "port": 1883,
        "topic": "test", "encoding": "json", "qos": 0
    })
    assert "error" in result["status"]


# ── CSV Writer ────────────────────────────────────────────────────────────────

def test_csv_writer_creates_file():
    from nodes.io.file_nodes import CSVWriterNode
    with tempfile.TemporaryDirectory() as tmp:
        path = str(pathlib.Path(tmp) / "out.csv")
        result = CSVWriterNode().execute({
            "data": [1.0, 2.0, 3.0], "path": path,
            "headers": "a,b,c", "overwrite": False
        })
        assert result["status"] == "ok"
        assert pathlib.Path(path).exists()
        content = pathlib.Path(path).read_text()
        assert "a,b,c" in content
        assert "1.0" in content


def test_csv_writer_append():
    from nodes.io.file_nodes import CSVWriterNode
    with tempfile.TemporaryDirectory() as tmp:
        path = str(pathlib.Path(tmp) / "out.csv")
        node = CSVWriterNode()
        node.execute({"data": [1, 2], "path": path, "headers": "", "overwrite": False})
        node.execute({"data": [3, 4], "path": path, "headers": "", "overwrite": False})
        lines = pathlib.Path(path).read_text().strip().split("\n")
        assert len(lines) == 2


def test_csv_writer_overwrite():
    from nodes.io.file_nodes import CSVWriterNode
    with tempfile.TemporaryDirectory() as tmp:
        path = str(pathlib.Path(tmp) / "out.csv")
        node = CSVWriterNode()
        node.execute({"data": [1, 2], "path": path, "headers": "", "overwrite": False})
        node.execute({"data": [3, 4], "path": path, "headers": "", "overwrite": True})
        lines = pathlib.Path(path).read_text().strip().split("\n")
        assert len(lines) == 1
        assert "3" in lines[0]


def test_csv_writer_tensor():
    import torch
    from nodes.io.file_nodes import CSVWriterNode
    with tempfile.TemporaryDirectory() as tmp:
        path = str(pathlib.Path(tmp) / "out.csv")
        t = torch.tensor([0.1, 0.9])
        result = CSVWriterNode().execute({
            "data": t, "path": path, "headers": "", "overwrite": False
        })
        assert result["status"] == "ok"
        assert "0.1" in pathlib.Path(path).read_text()


def test_csv_writer_none_data():
    from nodes.io.file_nodes import CSVWriterNode
    with tempfile.TemporaryDirectory() as tmp:
        path = str(pathlib.Path(tmp) / "out.csv")
        result = CSVWriterNode().execute({"data": None, "path": path, "headers": "", "overwrite": False})
        assert result["status"] == "no data"


# ── JSON Writer ────────────────────────────────────────────────────────────────

def test_json_writer_overwrite():
    from nodes.io.file_nodes import JSONWriterNode
    with tempfile.TemporaryDirectory() as tmp:
        path = str(pathlib.Path(tmp) / "out.json")
        result = JSONWriterNode().execute({
            "data": {"epoch": 1, "loss": 0.5}, "path": path, "mode": "overwrite", "indent": 2
        })
        assert result["status"] == "ok"
        loaded = json.loads(pathlib.Path(path).read_text())
        assert loaded["epoch"] == 1


def test_json_writer_append_jsonl():
    from nodes.io.file_nodes import JSONWriterNode
    with tempfile.TemporaryDirectory() as tmp:
        path = str(pathlib.Path(tmp) / "out.jsonl")
        node = JSONWriterNode()
        node.execute({"data": {"step": 1}, "path": path, "mode": "append", "indent": 0})
        node.execute({"data": {"step": 2}, "path": path, "mode": "append", "indent": 0})
        lines = pathlib.Path(path).read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["step"] == 1
        assert json.loads(lines[1])["step"] == 2


# ── Text Log ──────────────────────────────────────────────────────────────────

def test_text_log_creates_file():
    from nodes.io.file_nodes import TextLogNode
    with tempfile.TemporaryDirectory() as tmp:
        path = str(pathlib.Path(tmp) / "run.log")
        result = TextLogNode().execute({
            "message": "hello", "data": None, "path": path, "timestamp": False
        })
        assert result["status"] == "ok"
        assert "hello" in pathlib.Path(path).read_text()


def test_text_log_with_data():
    from nodes.io.file_nodes import TextLogNode
    with tempfile.TemporaryDirectory() as tmp:
        path = str(pathlib.Path(tmp) / "run.log")
        TextLogNode().execute({
            "message": "loss:", "data": 0.123, "path": path, "timestamp": False
        })
        content = pathlib.Path(path).read_text()
        assert "loss:" in content
        assert "0.123" in content


def test_text_log_appends():
    from nodes.io.file_nodes import TextLogNode
    with tempfile.TemporaryDirectory() as tmp:
        path = str(pathlib.Path(tmp) / "run.log")
        node = TextLogNode()
        node.execute({"message": "line1", "data": None, "path": path, "timestamp": False})
        node.execute({"message": "line2", "data": None, "path": path, "timestamp": False})
        lines = pathlib.Path(path).read_text().strip().split("\n")
        assert len(lines) == 2
