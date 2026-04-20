"""Port types registered by the Agents plugin.

Heavy imports stay out of this file — these are pure type registrations.
Most agent types are opaque value handles (LLM client, conversation list,
tool def) that flow between nodes without coercion.

Phase A: LLM, MESSAGE, CONVERSATION, PROMPT_TEMPLATE.
Phase B: + TOOL (single tool def) + DOCUMENT / EMBEDDING / MEMORY_REF
(Memory subsystem).
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
    PortTypeRegistry.register(
        "TOOL", default=None,
        color=(255, 140, 200, 255), pin_shape="triangle",
        description="Single tool definition (name, description, schema, callable)",
    )
    PortTypeRegistry.register(
        "DOCUMENT", default=None,
        color=(120, 220, 200, 255), pin_shape="quad",
        description="{id, text, metadata} chunk emitted by loaders; input to embedder/store",
    )
    PortTypeRegistry.register(
        "EMBEDDING", default=None,
        color=(200, 120, 220, 255), pin_shape="triangle_filled",
        description="Dense vector with `dim` attribute. Distinct from TENSOR; "
                    "RetrieverNode only accepts EMBEDDING.",
    )
    PortTypeRegistry.register(
        "MEMORY_REF", default=None,
        color=(140, 200, 240, 255), pin_shape="quad_filled",
        description="String handle {backend, path, collection} for a vector "
                    "store. Serializable — not a live client.",
    )
