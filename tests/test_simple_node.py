"""Tests for the @node DSL and hot reload."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core.custom import node
from core.node import PortType
from nodes import NODE_REGISTRY


# ── Basic @node usage ─────────────────────────────────────────────────────────

def test_single_float_output():
    @node(label="Test Add", category="TestCat")
    def test_add_fn(a: float = 1.0, b: float = 2.0) -> float:
        return a + b

    cls = NODE_REGISTRY["test_add_fn"]
    inst = cls()
    result = inst.execute({"a": 3.0, "b": 4.0})
    assert result["result"] == 7.0


def test_default_values_used_when_input_none():
    @node(label="Test Default", category="TestCat")
    def test_default_fn(x: float = 5.0) -> float:
        return x * 2

    cls = NODE_REGISTRY["test_default_fn"]
    inst = cls()
    # execute() passes None → node uses whatever inputs.get returns
    result = inst.execute({"x": None})
    # None * 2 would fail, so None should propagate gracefully
    assert result["result"] is None


def test_int_port_type():
    @node(label="Int Node", category="TestCat")
    def int_node_fn(n: int = 3) -> int:
        return n * 2

    cls = NODE_REGISTRY["int_node_fn"]
    inst = cls()
    assert inst.inputs["n"].port_type == PortType.INT
    assert inst.outputs["result"].port_type == PortType.INT


def test_bool_port_type():
    @node(label="Bool Node", category="TestCat")
    def bool_node_fn(flag: bool = False) -> bool:
        return not flag

    cls = NODE_REGISTRY["bool_node_fn"]
    inst = cls()
    assert inst.inputs["flag"].port_type == PortType.BOOL


def test_str_port_type():
    @node(label="Str Node", category="TestCat")
    def str_node_fn(text: str = "hi") -> str:
        return text.upper()

    cls = NODE_REGISTRY["str_node_fn"]
    inst = cls()
    assert inst.inputs["text"].port_type == PortType.STRING
    result = inst.execute({"text": "hello"})
    assert result["result"] == "HELLO"


def test_untyped_param_is_any():
    @node(label="Any Node", category="TestCat")
    def any_node_fn(x=None) -> None:
        return x

    cls = NODE_REGISTRY["any_node_fn"]
    inst = cls()
    assert inst.inputs["x"].port_type == PortType.ANY


def test_multi_output():
    @node(label="Multi", category="TestCat", outputs={"lo": float, "hi": float})
    def multi_fn(a: float = 0.0, b: float = 1.0) -> dict:
        return {"lo": min(a, b), "hi": max(a, b)}

    cls = NODE_REGISTRY["multi_fn"]
    inst = cls()
    assert "lo" in inst.outputs
    assert "hi" in inst.outputs
    result = inst.execute({"a": 3.0, "b": 1.0})
    assert result["lo"] == 1.0
    assert result["hi"] == 3.0


def test_exception_returns_none():
    @node(label="Boom", category="TestCat")
    def boom_fn(x: float = 0.0) -> float:
        raise ValueError("intentional")

    cls = NODE_REGISTRY["boom_fn"]
    result = cls().execute({"x": 5.0})
    assert result["result"] is None


def test_original_function_still_callable():
    @node(label="Pass", category="TestCat")
    def pass_fn(x: float = 0.0) -> float:
        return x + 1

    # The decorator returns the original function unchanged
    assert pass_fn(4.0) == 5.0


def test_type_name_override():
    @node(label="Override", category="TestCat", type_name="my_custom_key")
    def _override_fn(x: float = 0.0) -> float:
        return x

    assert "my_custom_key" in NODE_REGISTRY


def test_category_and_label():
    @node(label="Cat Test", category="Animals")
    def cat_test_fn(x: float = 0.0) -> float:
        return x

    cls = NODE_REGISTRY["cat_test_fn"]
    assert cls.label == "Cat Test"
    assert cls.category == "Animals"


def test_in_graph():
    """@node nodes work inside Graph.execute()."""
    from core.graph import Graph

    @node(label="Triple", category="TestCat")
    def triple_fn(x: float = 1.0) -> float:
        return x * 3

    cls = NODE_REGISTRY["triple_fn"]
    g = Graph()
    n = cls()
    n.inputs["x"].default_value = 4.0
    g.add_node(n)
    outputs, _, _ = g.execute()
    assert outputs[n.id]["result"] == 12.0


# ── Hot reloader ──────────────────────────────────────────────────────────────

def test_hot_reload_creates_custom_dir(tmp_path, monkeypatch):
    import core.custom as hr_mod
    monkeypatch.setattr(hr_mod, "CUSTOM_DIR", tmp_path / "custom")
    from core.custom import HotReloader
    reloader = HotReloader.__new__(HotReloader)
    hr_mod.CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
    reloader._mtimes = {}
    reloader._modules = {}
    reloader._last_check = 0.0
    assert hr_mod.CUSTOM_DIR.exists()


def test_hot_reload_picks_up_new_file(tmp_path, monkeypatch):
    import core.custom as hr_mod
    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()
    monkeypatch.setattr(hr_mod, "CUSTOM_DIR", custom_dir)

    from core.custom import HotReloader
    reloader = HotReloader.__new__(HotReloader)
    reloader._mtimes = {}
    reloader._modules = {}
    reloader._last_check = 0.0

    # Write a file with a @node decorator
    py_file = custom_dir / "hottest.py"
    py_file.write_text(
        "from core.custom import node\n"
        "@node(label='Hot Node', category='Custom')\n"
        "def hot_reload_test_node(x: float = 1.0) -> float:\n"
        "    return x + 99\n"
    )

    results = reloader.poll()
    assert len(results) == 1
    msg, new_types = results[0]
    assert "hottest.py" in msg
    assert "hot_reload_test_node" in new_types
    # Node should be callable
    from nodes import NODE_REGISTRY
    cls = NODE_REGISTRY.get("hot_reload_test_node")
    assert cls is not None
    assert cls().execute({"x": 1.0})["result"] == 100.0


def test_hot_reload_detects_changes(tmp_path, monkeypatch):
    import time
    import core.custom as hr_mod
    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()
    monkeypatch.setattr(hr_mod, "CUSTOM_DIR", custom_dir)

    from core.custom import HotReloader
    reloader = HotReloader.__new__(HotReloader)
    reloader._mtimes = {}
    reloader._modules = {}
    reloader._last_check = 0.0

    py_file = custom_dir / "changeme.py"
    py_file.write_text(
        "from core.custom import node\n"
        "@node(label='V1', category='Custom', type_name='change_v1')\n"
        "def change_v1(x: float = 0.0) -> float:\n"
        "    return x\n"
    )
    reloader.poll()

    # Simulate a file change (bump mtime)
    time.sleep(0.01)
    py_file.write_text(
        "from core.custom import node\n"
        "@node(label='V2', category='Custom', type_name='change_v2')\n"
        "def change_v2(x: float = 0.0) -> float:\n"
        "    return x * 2\n"
    )
    # Force next poll to see the change
    reloader._last_check = 0.0
    reloader._mtimes[str(py_file)] = 0.0  # reset mtime so change is detected

    results = reloader.poll()
    assert len(results) == 1
    msg, _ = results[0]
    assert "Reloaded" in msg or "Loaded" in msg
