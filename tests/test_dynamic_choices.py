"""Port.dynamic_choices — lets the Inspector populate a dropdown from
an RPC on demand (e.g. ollama installed-model list). Covers:

  - Port carries the field
  - add_input accepts the kwarg
  - Server's node-dict emits it (both for graph nodes and for the palette
    registry)
  - OllamaClientNode.model ships with `agent_list_local_models` wired up
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _register_port_types():
    from plugins.agents.port_types import register_all
    register_all()


def test_port_dataclass_has_dynamic_choices_field():
    from core.node import Port
    p = Port(name="x", port_type="STRING", is_input=True,
             dynamic_choices="my_rpc")
    assert p.dynamic_choices == "my_rpc"


def test_add_input_accepts_dynamic_choices_kwarg():
    from core.node import BaseNode, PortType

    class _N(BaseNode):
        type_name = "_t_dc"
        def _setup_ports(self):
            self.add_input("pick", PortType.STRING,
                           dynamic_choices="some_list_rpc")
            self.add_output("out", PortType.ANY)
        def execute(self, inputs):
            return {"out": None}

    n = _N()
    assert n.inputs["pick"].dynamic_choices == "some_list_rpc"


def test_dynamic_choices_surfaces_in_node_dict():
    """Server's `_node_to_dict` must serialize `dynamic_choices` so the
    Inspector can render a combobox for it."""
    from server import NodeToolServer
    from nodes.agents.ollama_client import OllamaClientNode

    s = NodeToolServer()
    node = OllamaClientNode()
    d = s._node_to_dict(node)
    assert d["inputs"]["model"]["dynamic_choices"] == "agent_list_local_models"
    # Empty string (never set) round-trips as empty string, not missing.
    assert d["inputs"]["host"]["dynamic_choices"] == ""


def test_ollama_client_model_ships_with_dynamic_choices():
    from nodes.agents.ollama_client import OllamaClientNode
    n = OllamaClientNode()
    assert n.inputs["model"].dynamic_choices == "agent_list_local_models"


def test_registry_payload_includes_dynamic_choices():
    """`get_registry` must surface the field so palette-origin nodes
    (dragged onto the canvas fresh) carry it too."""
    from server import NodeToolServer
    s = NodeToolServer()
    resp = s.get_registry({})
    # Find the OllamaClient entry in the registry.
    found_model_field = None
    for cat, nodes in resp["categories"].items():
        for n in nodes:
            if n["type_name"] == "ag_ollama_client":
                found_model_field = n["inputs"]["model"]
                break
    assert found_model_field is not None
    assert found_model_field["dynamic_choices"] == "agent_list_local_models"
