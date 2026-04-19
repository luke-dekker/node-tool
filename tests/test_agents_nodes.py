"""Per-node smoke tests using MockLLMClient — no network calls."""
import pytest


@pytest.fixture(autouse=True)
def _register_agent_port_types():
    """Every test in this module needs LLM/MESSAGE/CONVERSATION/PROMPT_TEMPLATE
    registered so add_input() lookups succeed."""
    from plugins.agents.port_types import register_all
    register_all()


def test_chat_message_node():
    from nodes.agents.chat_message import ChatMessageNode
    from plugins.agents._llm.protocol import Message
    n = ChatMessageNode()
    out = n.execute({"role": "user", "content": "hi"})
    assert isinstance(out["message"], Message)
    assert out["message"].role == "user"
    assert out["message"].content == "hi"


def test_chat_message_node_invalid_role_falls_back_to_user():
    from nodes.agents.chat_message import ChatMessageNode
    n = ChatMessageNode()
    out = n.execute({"role": "wizard", "content": "✨"})
    assert out["message"].role == "user"


def test_conversation_node():
    from nodes.agents.conversation import ConversationNode
    from plugins.agents._llm.protocol import Message
    n = ConversationNode()
    out = n.execute({
        "system":    Message(role="system", content="be brief"),
        "user":      Message(role="user", content="hello"),
        "assistant": None,
    })
    assert len(out["conversation"]) == 2
    assert out["conversation"][0].role == "system"
    assert out["conversation"][1].role == "user"


def test_prompt_template_node():
    from nodes.agents.prompt_template import PromptTemplateNode
    n = PromptTemplateNode()
    out = n.execute({"template": "Hello {name}!", "vars": {"name": "Luke"}})
    assert out["text"] == "Hello Luke!"


def test_prompt_template_missing_var_keeps_template_intact():
    from nodes.agents.prompt_template import PromptTemplateNode
    n = PromptTemplateNode()
    out = n.execute({"template": "Hello {name}!", "vars": {}})
    assert out["text"] == "Hello {name}!"


def test_prompt_template_non_dict_vars():
    from nodes.agents.prompt_template import PromptTemplateNode
    n = PromptTemplateNode()
    out = n.execute({"template": "static", "vars": "not a dict"})
    assert out["text"] == "static"


def test_ollama_client_node_yields_client():
    from nodes.agents.ollama_client import OllamaClientNode
    from plugins.agents._llm.ollama_client import OllamaClient
    n = OllamaClientNode()
    out = n.execute({"host": "http://localhost:9999", "model": "qwen2.5:7b"})
    assert isinstance(out["llm"], OllamaClient)
    assert out["llm"].host == "http://localhost:9999"
    assert out["llm"].default_model == "qwen2.5:7b"


def test_openai_compat_client_node_yields_client():
    from nodes.agents.openai_compat_client import OpenAICompatClientNode
    from plugins.agents._llm.openai_compat_client import OpenAICompatClient
    n = OpenAICompatClientNode()
    out = n.execute({"base_url": "http://x/v1", "api_key": "k", "model": "m"})
    assert isinstance(out["llm"], OpenAICompatClient)
    assert out["llm"].base_url == "http://x/v1"
    assert out["llm"].default_model == "m"


def test_agent_node_with_mock_llm():
    from nodes.agents.agent import AgentNode
    from plugins.agents._llm.mock_client import MockLLMClient
    from plugins.agents._llm.protocol import Message

    llm = MockLLMClient(response="hello back")
    n = AgentNode()
    out = n.execute({
        "llm":           llm,
        "messages":      [Message(role="user", content="hi")],
        "system_prompt": "be brief",
        "model":         "",
        "temperature":   0.5,
    })
    assert out["text"] == "hello back"
    assert out["final_message"].content == "hello back"
    assert len(out["response"]) == 3        # system + user + assistant
    assert out["response"][0].role == "system"
    assert out["response"][0].content == "be brief"
    assert llm.calls[0]["kw"]["temperature"] == 0.5


def test_agent_node_no_llm_raises():
    from nodes.agents.agent import AgentNode
    from plugins.agents._llm.protocol import Message
    n = AgentNode()
    with pytest.raises(RuntimeError, match="no LLM client"):
        n.execute({"llm": None, "messages": [Message(role="user", content="hi")],
                   "system_prompt": "", "model": "", "temperature": 0.7})


def test_agent_node_no_messages_raises():
    from nodes.agents.agent import AgentNode
    from plugins.agents._llm.mock_client import MockLLMClient
    n = AgentNode()
    with pytest.raises(RuntimeError, match="no messages"):
        n.execute({"llm": MockLLMClient(), "messages": [], "system_prompt": "",
                   "model": "", "temperature": 0.7})


def test_agent_node_replaces_existing_system_message():
    """When system_prompt is set, any incoming system message gets dropped
    so we don't send two system messages."""
    from nodes.agents.agent import AgentNode
    from plugins.agents._llm.mock_client import MockLLMClient
    from plugins.agents._llm.protocol import Message
    llm = MockLLMClient()
    n = AgentNode()
    n.execute({
        "llm":           llm,
        "messages":      [Message(role="system", content="OLD"),
                          Message(role="user", content="hi")],
        "system_prompt": "NEW",
        "model":         "",
        "temperature":   0.7,
    })
    sent = llm.calls[0]["messages"]
    sys_msgs = [m for m in sent if m.role == "system"]
    assert len(sys_msgs) == 1
    assert sys_msgs[0].content == "NEW"
