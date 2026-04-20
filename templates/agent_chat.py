"""Agent Chat — the hello-world agent graph.

An Ollama-backed agent with one Python-function tool (`get_time`). Open the
Agents panel, type a message in the Chat section, press Send — the model
identifies the tool, dispatches it, and streams a response back.

    Ollama Client  ──llm─────►
    ChatMessage(user) ──msg─► Conversation ──conv─► Agent ──response
    PythonFunctionTool (get_time) ──tool_1─────────►

Required: Ollama running on localhost:11434 with at least one model pulled
(e.g., `ollama pull qwen2.5:0.5b`).
"""
from __future__ import annotations
from core.graph import Graph
from templates._helpers import grid

LABEL = "Agent Chat (Ollama + 1 tool)"
DESCRIPTION = ("Ollama-backed agent with a get_time tool. Demonstrates the "
               "function-calling loop end-to-end.")


def build(graph: Graph) -> dict[str, tuple[int, int]]:
    from nodes.agents.ollama_client         import OllamaClientNode
    from nodes.agents.chat_message          import ChatMessageNode
    from nodes.agents.conversation          import ConversationNode
    from nodes.agents.agent                 import AgentNode
    from nodes.agents.python_function_tool  import PythonFunctionToolNode

    pos = grid(step_x=240)
    positions: dict[str, tuple[int, int]] = {}

    cli = OllamaClientNode()
    cli.inputs["host"].default_value  = "http://localhost:11434"
    cli.inputs["model"].default_value = "qwen2.5:0.5b"
    graph.add_node(cli); positions[cli.id] = pos()

    user = ChatMessageNode()
    user.inputs["role"].default_value    = "user"
    user.inputs["content"].default_value = "What time is it?"
    graph.add_node(user); positions[user.id] = pos()

    conv = ConversationNode()
    graph.add_node(conv); positions[conv.id] = pos()

    tool = PythonFunctionToolNode()
    tool.inputs["name"].default_value = "get_time"
    tool.inputs["description"].default_value = "Return the current local time."
    tool.inputs["code"].default_value = (
        "from datetime import datetime\n"
        "return datetime.now().isoformat(timespec='seconds')\n"
    )
    tool.inputs["side_effect"].default_value = False
    graph.add_node(tool); positions[tool.id] = pos(col=1, row=1)

    agent = AgentNode()
    agent.inputs["system_prompt"].default_value = (
        "You are a concise assistant. Use the get_time tool when asked about time."
    )
    agent.inputs["max_iterations"].default_value = 3
    agent.inputs["temperature"].default_value    = 0.0
    graph.add_node(agent); positions[agent.id] = pos(col=3, row=0)

    graph.add_connection(cli.id,  "llm",          agent.id, "llm")
    graph.add_connection(user.id, "message",      conv.id,  "user")
    graph.add_connection(conv.id, "conversation", agent.id, "messages")
    graph.add_connection(tool.id, "tool",         agent.id, "tool_1")

    return positions
