"""Port types registered by the Agents plugin.

Heavy imports stay out of this file — these are pure type registrations.
Most agent types are opaque value handles (LLM client, conversation list)
that flow between nodes without coercion.

Phase A registers four types: LLM, MESSAGE, CONVERSATION, PROMPT_TEMPLATE.
TOOL / DOCUMENT / EMBEDDING / MEMORY_REF land in Phase B.
"""
from __future__ import annotations

from core.port_types import PortTypeRegistry


def register_all() -> None:
    """Register every Agents port type. Called from plugins/agents/__init__.py."""
    PortTypeRegistry.register(
        "LLM", default=None,
        color=(120, 160, 255, 255), pin_shape="triangle",
        description="LLMClient handle (Ollama / OpenAI-compat / llama.cpp)",
    )
    PortTypeRegistry.register(
        "MESSAGE", default=None,
        color=(180, 200, 100, 255), pin_shape="circle_filled",
        description="Single chat message {role, content, ...}",
    )
    PortTypeRegistry.register(
        "CONVERSATION", default=None,
        color=(220, 200, 80, 255), pin_shape="quad_filled",
        description="list[MESSAGE] — full chat transcript",
    )
    PortTypeRegistry.register(
        "PROMPT_TEMPLATE", default="", coerce=str, editable=True,
        color=(255, 180, 120, 255), pin_shape="circle_filled",
        description="String template with {var} slots",
    )
