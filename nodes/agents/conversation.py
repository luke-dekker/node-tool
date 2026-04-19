"""ConversationNode — bundle MESSAGEs into a CONVERSATION list."""
from __future__ import annotations
from typing import Any

from core.node import BaseNode


class ConversationNode(BaseNode):
    type_name   = "ag_conversation"
    label       = "Conversation"
    category    = "Agents"
    subcategory = "Prompts"
    description = "Combine system / user / assistant messages into a conversation."

    def _setup_ports(self) -> None:
        # Phase A: 3 fixed slots. Variadic ports come in Phase B.
        self.add_input("system",    "MESSAGE", default=None,
                       description="Optional system message")
        self.add_input("user",      "MESSAGE", default=None,
                       description="User message")
        self.add_input("assistant", "MESSAGE", default=None,
                       description="Optional assistant prefix")
        self.add_output("conversation", "CONVERSATION")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        msgs = []
        for slot in ("system", "user", "assistant"):
            m = inputs.get(slot)
            if m is not None:
                msgs.append(m)
        return {"conversation": msgs}

    def export(self, iv, ov):
        sys_ = iv.get("system") or "None"
        usr_ = iv.get("user") or "None"
        ast_ = iv.get("assistant") or "None"
        out = ov.get("conversation", "_conv")
        return [], [f"{out} = [m for m in ({sys_}, {usr_}, {ast_}) if m is not None]"]
