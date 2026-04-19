"""AgentNode — single chat turn against an LLM client.

Phase A: no tools, no streaming, no function-calling loop. Just sends the
conversation and returns the assistant's reply. Tools and the iteration
loop come in Phase B.
"""
from __future__ import annotations
from typing import Any

from core.node import BaseNode, PortType


class AgentNode(BaseNode):
    type_name   = "ag_agent"
    label       = "Agent"
    category    = "Agents"
    subcategory = "Agent"
    description = "Send a conversation to an LLM, return the assistant's reply."

    def _setup_ports(self) -> None:
        self.add_input("llm", "LLM", default=None,
                       description="LLM client (from OllamaClientNode etc.)")
        self.add_input("messages", "CONVERSATION", default=None,
                       description="Input conversation")
        self.add_input("system_prompt", PortType.STRING, default="",
                       description="Optional system prompt prepended to messages")
        self.add_input("model", PortType.STRING, default="",
                       description="Override model (blank = use client's default)")
        self.add_input("temperature", PortType.FLOAT, default=0.7,
                       description="Sampling temperature")
        self.add_output("response", "CONVERSATION",
                        description="Conversation including the assistant's reply")
        self.add_output("final_message", "MESSAGE",
                        description="The assistant's final message")
        self.add_output("text", PortType.STRING,
                        description="Plain text content of the final message")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        from plugins.agents._llm.protocol import Message  # deferred

        llm = inputs.get("llm")
        if llm is None:
            raise RuntimeError("AgentNode: no LLM client connected")

        msgs_in = list(inputs.get("messages") or [])
        sys_prompt = (inputs.get("system_prompt") or "").strip()
        if sys_prompt:
            # Prepend system; drop any incoming system messages so we don't
            # send two.
            msgs_in = [Message(role="system", content=sys_prompt)] + [
                m for m in msgs_in
                if not (isinstance(m, Message) and m.role == "system")
            ]
        if not msgs_in:
            raise RuntimeError("AgentNode: no messages to send")

        model = (inputs.get("model") or "").strip() or None
        temp = float(inputs.get("temperature") if inputs.get("temperature") is not None else 0.7)

        result = llm.chat(msgs_in, model=model, temperature=temp)
        full = msgs_in + [result.message]
        return {
            "response":      full,
            "final_message": result.message,
            "text":          result.message.content,
        }

    def export(self, iv, ov):
        return [], [f"# AgentNode export pending Phase D"]
