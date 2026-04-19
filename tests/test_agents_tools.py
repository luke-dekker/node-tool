"""Phase B Tools — TOOL port type, ToolNode, PythonFunctionToolNode,
AgentNode function-calling iteration loop."""
import json
import pytest


@pytest.fixture(autouse=True)
def _register_agent_port_types():
    from plugins.agents.port_types import register_all
    register_all()


# ── ToolDef + port type ───────────────────────────────────────────────────

def test_tool_port_type_registered():
    from core.port_types import PortTypeRegistry
    assert PortTypeRegistry.get("TOOL") is not None


def test_tooldef_to_openai_wire_format():
    from plugins.agents._llm.protocol import ToolDef
    td = ToolDef(name="ping", description="ping pong",
                 input_schema={"type": "object", "properties": {"x": {"type": "integer"}}})
    wire = td.to_openai()
    assert wire["type"] == "function"
    assert wire["function"]["name"] == "ping"
    assert wire["function"]["description"] == "ping pong"
    assert wire["function"]["parameters"]["properties"]["x"]["type"] == "integer"


# ── ToolNode (dotted path) ────────────────────────────────────────────────

def test_tool_node_resolves_dotted_path():
    from nodes.agents.tool import ToolNode
    n = ToolNode()
    out = n.execute({
        "name": "isoformat",
        "description": "format a datetime",
        "input_schema": "",
        "python_callable": "datetime.datetime.now",
        "side_effect": False,
    })
    td = out["tool"]
    assert td.name == "isoformat"
    assert td.callable is not None
    # The callable should actually work
    result = td.callable()
    assert hasattr(result, "isoformat")


def test_tool_node_rejects_invalid_schema():
    from nodes.agents.tool import ToolNode
    n = ToolNode()
    with pytest.raises(RuntimeError, match="invalid input_schema"):
        n.execute({
            "name": "x", "description": "", "input_schema": "{not valid",
            "python_callable": "datetime.datetime.now", "side_effect": False,
        })


def test_tool_node_rejects_empty_callable():
    from nodes.agents.tool import ToolNode
    n = ToolNode()
    with pytest.raises(RuntimeError, match="python_callable is empty"):
        n.execute({
            "name": "x", "description": "", "input_schema": "",
            "python_callable": "", "side_effect": False,
        })


# ── PythonFunctionToolNode (inline exec) ──────────────────────────────────

def test_python_function_tool_default_body():
    """Default body returns the current time — sanity check end-to-end compile."""
    from nodes.agents.python_function_tool import PythonFunctionToolNode
    n = PythonFunctionToolNode()
    out = n.execute({
        "name": "get_time",
        "description": "Returns current time",
        "input_schema": "",
        "code": n._DEFAULT_BODY,
        "side_effect": True,
    })
    td = out["tool"]
    assert td.name == "get_time"
    assert td.side_effect is True
    iso = td.callable()
    assert isinstance(iso, str) and len(iso) >= 10  # YYYY-MM-DD at minimum


def test_python_function_tool_custom_body_with_kwargs():
    from nodes.agents.python_function_tool import PythonFunctionToolNode
    n = PythonFunctionToolNode()
    schema = json.dumps({"type": "object",
                         "properties": {"a": {"type": "integer"},
                                        "b": {"type": "integer"}},
                         "required": ["a", "b"]})
    out = n.execute({
        "name": "add",
        "description": "add two ints",
        "input_schema": schema,
        "code": "return kwargs['a'] + kwargs['b']",
        "side_effect": False,
    })
    td = out["tool"]
    assert td.callable(a=2, b=3) == 5
    assert td.input_schema["required"] == ["a", "b"]


def test_python_function_tool_syntax_error():
    from nodes.agents.python_function_tool import PythonFunctionToolNode
    n = PythonFunctionToolNode()
    with pytest.raises(RuntimeError, match="syntax error"):
        n.execute({
            "name": "broken", "description": "", "input_schema": "",
            "code": "def x(:\n  return 1", "side_effect": True,
        })


# ── AgentNode function-calling loop ───────────────────────────────────────

class _ScriptedLLM:
    """Returns canned ChatResults in order. For testing the tool loop."""
    capabilities = {"tools", "stream"}

    def __init__(self, scripted_responses):
        from plugins.agents._llm.protocol import Message, ChatResult
        self._Message = Message
        self._ChatResult = ChatResult
        self._responses = list(scripted_responses)
        self.calls = []

    def chat(self, messages, *, model=None, tools=None, **kw):
        self.calls.append({"messages": list(messages), "tools": tools})
        if not self._responses:
            return self._ChatResult(
                message=self._Message(role="assistant", content="<no more scripted responses>"),
                model=model or "scripted",
            )
        spec = self._responses.pop(0)
        msg = self._Message(role="assistant",
                            content=spec.get("content", ""),
                            tool_calls=spec.get("tool_calls"))
        return self._ChatResult(message=msg, model=model or "scripted",
                                tokens_in=1, tokens_out=1)


def _make_tool(name, fn, *, side_effect=False, schema=None):
    from plugins.agents._llm.protocol import ToolDef
    return ToolDef(name=name, description=name,
                   input_schema=schema or {"type": "object", "properties": {},
                                            "additionalProperties": True},
                   callable=fn, side_effect=side_effect)


def test_agent_no_tools_falls_through_immediately():
    from nodes.agents.agent import AgentNode
    from plugins.agents._llm.protocol import Message
    llm = _ScriptedLLM([{"content": "hello"}])
    n = AgentNode()
    out = n.execute({
        "llm": llm, "messages": [Message(role="user", content="hi")],
        "system_prompt": "", "model": "", "temperature": 0.0,
        "tool_1": None, "tool_2": None, "tool_3": None, "tool_4": None,
        "max_iterations": 3, "allow_side_effect_tools": False,
    })
    assert out["text"] == "hello"
    assert out["tool_calls"] == []
    assert len(llm.calls) == 1


def test_agent_dispatches_one_tool_call_then_finalizes():
    from nodes.agents.agent import AgentNode
    from plugins.agents._llm.protocol import Message

    add = _make_tool("add", lambda **kw: kw["a"] + kw["b"])
    llm = _ScriptedLLM([
        # turn 1: model asks for the tool
        {"tool_calls": [{"function": {"name": "add", "arguments": {"a": 2, "b": 3}}}]},
        # turn 2: model answers using the tool's result
        {"content": "The answer is 5."},
    ])
    n = AgentNode()
    out = n.execute({
        "llm": llm, "messages": [Message(role="user", content="2+3?")],
        "system_prompt": "", "model": "", "temperature": 0.0,
        "tool_1": add, "tool_2": None, "tool_3": None, "tool_4": None,
        "max_iterations": 5, "allow_side_effect_tools": False,
    })
    assert out["text"] == "The answer is 5."
    assert len(out["tool_calls"]) == 1
    assert out["tool_calls"][0]["name"] == "add"
    assert out["tool_calls"][0]["result"] == 5
    # Second LLM call should have the tool-role message in its history.
    assert any(getattr(m, "role", "") == "tool" for m in llm.calls[1]["messages"])


def test_agent_parses_openai_style_json_string_arguments():
    from nodes.agents.agent import AgentNode
    from plugins.agents._llm.protocol import Message

    echo = _make_tool("echo", lambda **kw: kw.get("text", ""))
    llm = _ScriptedLLM([
        {"tool_calls": [{"id": "call_x", "function": {
            "name": "echo", "arguments": '{"text":"hi"}'}}]},
        {"content": "echoed: hi"},
    ])
    n = AgentNode()
    out = n.execute({
        "llm": llm, "messages": [Message(role="user", content="echo hi")],
        "system_prompt": "", "model": "", "temperature": 0.0,
        "tool_1": echo, "tool_2": None, "tool_3": None, "tool_4": None,
        "max_iterations": 3, "allow_side_effect_tools": False,
    })
    assert out["tool_calls"][0]["result"] == "hi"
    assert out["text"] == "echoed: hi"


def test_agent_blocks_side_effect_tool_without_flag():
    from nodes.agents.agent import AgentNode
    from plugins.agents._llm.protocol import Message

    sentinel = []
    danger = _make_tool("danger", lambda **kw: sentinel.append("called!"),
                        side_effect=True)
    llm = _ScriptedLLM([
        {"tool_calls": [{"function": {"name": "danger", "arguments": {}}}]},
        {"content": "i tried"},
    ])
    n = AgentNode()
    out = n.execute({
        "llm": llm, "messages": [Message(role="user", content="do it")],
        "system_prompt": "", "model": "", "temperature": 0.0,
        "tool_1": danger, "tool_2": None, "tool_3": None, "tool_4": None,
        "max_iterations": 3, "allow_side_effect_tools": False,
    })
    assert sentinel == []  # tool was NOT called
    assert "side_effect=True" in out["tool_calls"][0]["error"]


def test_agent_allows_side_effect_tool_when_flag_set():
    from nodes.agents.agent import AgentNode
    from plugins.agents._llm.protocol import Message

    sentinel = []
    fn = _make_tool("touch", lambda **kw: sentinel.append("called!") or "ok",
                    side_effect=True)
    llm = _ScriptedLLM([
        {"tool_calls": [{"function": {"name": "touch", "arguments": {}}}]},
        {"content": "done"},
    ])
    n = AgentNode()
    n.execute({
        "llm": llm, "messages": [Message(role="user", content="touch")],
        "system_prompt": "", "model": "", "temperature": 0.0,
        "tool_1": fn, "tool_2": None, "tool_3": None, "tool_4": None,
        "max_iterations": 3, "allow_side_effect_tools": True,
    })
    assert sentinel == ["called!"]


def test_agent_handles_unknown_tool_gracefully():
    from nodes.agents.agent import AgentNode
    from plugins.agents._llm.protocol import Message

    llm = _ScriptedLLM([
        {"tool_calls": [{"function": {"name": "ghost", "arguments": {}}}]},
        {"content": "n/a"},
    ])
    n = AgentNode()
    out = n.execute({
        "llm": llm, "messages": [Message(role="user", content="hi")],
        "system_prompt": "", "model": "", "temperature": 0.0,
        "tool_1": None, "tool_2": None, "tool_3": None, "tool_4": None,
        "max_iterations": 3, "allow_side_effect_tools": False,
    })
    # No tools were bound, so the model's hallucinated tool_call ends the loop.
    assert out["text"] == ""  # the assistant message had only tool_calls, no content


def test_agent_tool_callable_exception_surfaces_as_tool_error():
    from nodes.agents.agent import AgentNode
    from plugins.agents._llm.protocol import Message

    boom = _make_tool("boom", lambda **kw: (_ for _ in ()).throw(ValueError("bad")))
    llm = _ScriptedLLM([
        {"tool_calls": [{"function": {"name": "boom", "arguments": {}}}]},
        {"content": "saw error"},
    ])
    n = AgentNode()
    out = n.execute({
        "llm": llm, "messages": [Message(role="user", content="explode")],
        "system_prompt": "", "model": "", "temperature": 0.0,
        "tool_1": boom, "tool_2": None, "tool_3": None, "tool_4": None,
        "max_iterations": 3, "allow_side_effect_tools": False,
    })
    assert "ValueError" in out["tool_calls"][0]["error"]


def test_agent_max_iterations_caps_runaway_loop():
    from nodes.agents.agent import AgentNode
    from plugins.agents._llm.protocol import Message

    forever = _make_tool("ping", lambda **kw: "pong")
    # Model never stops asking for ping
    llm = _ScriptedLLM([
        {"tool_calls": [{"function": {"name": "ping", "arguments": {}}}]},
    ] * 20)
    n = AgentNode()
    out = n.execute({
        "llm": llm, "messages": [Message(role="user", content="loop")],
        "system_prompt": "", "model": "", "temperature": 0.0,
        "tool_1": forever, "tool_2": None, "tool_3": None, "tool_4": None,
        "max_iterations": 3, "allow_side_effect_tools": False,
    })
    # Exactly max_iterations LLM calls, then we append the cap-reached message.
    assert len(llm.calls) == 3
    assert "max_iterations" in out["text"]


def test_agent_collects_multiple_tool_slots():
    from nodes.agents.agent import AgentNode
    from plugins.agents._llm.protocol import Message

    a = _make_tool("a", lambda **kw: "a-result")
    b = _make_tool("b", lambda **kw: "b-result")
    llm = _ScriptedLLM([{"content": "noop"}])
    n = AgentNode()
    n.execute({
        "llm": llm, "messages": [Message(role="user", content="x")],
        "system_prompt": "", "model": "", "temperature": 0.0,
        "tool_1": a, "tool_2": None, "tool_3": b, "tool_4": None,
        "max_iterations": 1, "allow_side_effect_tools": False,
    })
    sent_tools = llm.calls[0]["tools"]
    assert sent_tools is not None
    names = {t["function"]["name"] for t in sent_tools}
    assert names == {"a", "b"}
