"""ChatMessageNode — build a single MESSAGE."""
from __future__ import annotations
from typing import Any

from core.node import BaseNode, PortType


class ChatMessageNode(BaseNode):
    type_name   = "ag_chat_message"
    label       = "Chat Message"
    category    = "Agents"
    subcategory = "Prompts"
    description = "Build a single chat message {role, content}."

    _ROLES = ["system", "user", "assistant"]

    def _setup_ports(self) -> None:
        self.add_input("role", PortType.STRING, default="user",
                       description="system | user | assistant",
                       choices=self._ROLES)
        self.add_input("content", PortType.STRING, default="",
                       description="Message body")
        self.add_output("message", "MESSAGE")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        from plugins.agents._llm.protocol import Message  # deferred
        role = inputs.get("role") or "user"
        if role not in self._ROLES:
            role = "user"
        content = inputs.get("content") or ""
        return {"message": Message(role=role, content=content)}

    def export(self, iv, ov):
        role = self._val(iv, "role")
        content = self._val(iv, "content")
        out = ov.get("message", "_msg")
        return [], [f"{out} = {{'role': {role}, 'content': {content}}}"]
