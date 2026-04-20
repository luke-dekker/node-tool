"""Agents panel spec — Phase A + B subset.

Sections (top → bottom): Backend, Local models (DynamicForm), Selected
agent (DynamicForm), Chat (custom — streaming chat transcript + input),
Status, Controls. Autoresearch sub-section comes in Phase C.

Single source of truth — DPG, React, Godot all render this exact spec.
"""
from __future__ import annotations

from core.panel import (
    PanelSpec, FormSection, DynamicFormSection, StatusSection,
    ButtonsSection, CustomSection, Field, Action,
)


def build_agents_panel_spec() -> PanelSpec:
    return PanelSpec(
        label="Agents",
        sections=[
            FormSection(
                id="backend",
                label="Backend",
                fields=[
                    Field("backend", "choice", label="backend",
                          choices=["ollama", "openai_compat"], default="ollama"),
                    Field("host", "str", label="host",
                          default="http://localhost:11434"),
                ],
            ),
            DynamicFormSection(
                id="models",
                label="Local models",
                source_rpc="agent_list_local_models",
                item_label_template="{name} ({size_h})",
                empty_hint=("No local Ollama models found. Pull one with "
                            "`ollama pull <model>` — see "
                            "https://ollama.com/library for options."),
                fields=[
                    Field("default", "bool", label="default", default=False),
                ],
            ),
            DynamicFormSection(
                id="agents",
                label="Selected agent",
                source_rpc="agent_list_agent_nodes",
                item_label_template="{label}",
                empty_hint="Add an Agent node to the graph to configure it.",
                fields=[
                    Field("system_prompt", "str", label="system prompt",
                          default=""),
                    Field("model", "str", label="model", default=""),
                    Field("temperature", "float", label="temp", default=0.7,
                          min=0.0, max=2.0, step=0.1),
                ],
            ),
            CustomSection(
                id="chat",
                label="Chat",
                custom_kind="chat_stream",
                params={
                    "input_field_id":  "message",
                    "start_rpc":       "agent_start_stream",
                    "stop_rpc":        "agent_stop",
                    "drain_rpc":       "agent_drain_tokens",
                    "state_rpc":       "get_agent_state",
                    "collect":         ["agents"],
                    "poll_ms":         100,
                    "placeholder":     "Type a message, press Enter to send",
                },
                fields=[
                    Field("message", "str", label="message", default=""),
                ],
            ),
            StatusSection(
                id="status",
                label="Status",
                source_rpc="get_agent_state",
                poll_ms=500,
                fields=[
                    Field("status",     "str", label="State"),
                    Field("model",      "str", label="Model"),
                    Field("tokens_in",  "int", label="tokens in"),
                    Field("tokens_out", "int", label="tokens out"),
                    Field("latency_ms", "int", label="ms"),
                    Field("error",      "str", label="Error"),
                ],
            ),
            ButtonsSection(
                id="controls",
                actions=[
                    Action(id="send_blocking", label="Send (blocking)",
                           rpc="agent_start",
                           collect=["agents", "chat"]),
                    Action(id="send_stream", label="Send (stream)",
                           rpc="agent_start_stream",
                           collect=["agents", "chat"]),
                    Action(id="stop", label="Stop", rpc="agent_stop"),
                ],
            ),
            # ── Autoresearch sub-section ────────────────────────────────
            StatusSection(
                id="autoresearch_status",
                label="Autoresearch",
                source_rpc="agent_autoresearch_state",
                poll_ms=1000,
                fields=[
                    Field("current_status",  "str", label="State"),
                    Field("trials_done",     "int", label="trials"),
                    Field("best_score",      "float", label="best loss"),
                    Field("current_op_kind", "str", label="last op"),
                    Field("error",           "str", label="error"),
                ],
            ),
            ButtonsSection(
                id="autoresearch_controls",
                actions=[
                    Action(id="ar_start", label="Start Autoresearch",
                           rpc="agent_autoresearch_start"),
                    Action(id="ar_stop",  label="Stop Autoresearch",
                           rpc="agent_autoresearch_stop"),
                ],
            ),
        ],
    )
