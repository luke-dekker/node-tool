"""Agents panel spec.

Observability + session actions. Configuration belongs on canvas nodes,
edited through the Inspector — not in panel forms. Sections here only
show live state (Status / Autoresearch Status / Trial history) or
actions on a running session (Send / Stop / Start Autoresearch).

Single source of truth — DPG, React, Godot all render this exact spec.
"""
from __future__ import annotations

from core.panel import (
    PanelSpec, StatusSection, ButtonsSection, CustomSection, Field, Action,
)


def build_agents_panel_spec() -> PanelSpec:
    """Agents panel — observability + session actions only.

    Canvas owns configuration (LLM client, agents, tools, autoresearch
    agent). The panel surfaces live state of whatever's running and
    buttons to start/stop sessions. No config forms here — edit the
    node in the Inspector instead.
    """
    return PanelSpec(
        label="Agents",
        sections=[
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
                    "collect":         [],
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
                           collect=["chat"]),
                    Action(id="send_stream", label="Send (stream)",
                           rpc="agent_start_stream",
                           collect=["chat"]),
                    Action(id="stop", label="Stop", rpc="agent_stop"),
                ],
            ),
            # ── Autoresearch ────────────────────────────────────────────
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
            # Trial-by-trial history — each row shows the mutation, the
            # resulting score, and whether it was kept or reverted. Polls
            # the same RPC as the status section.
            CustomSection(
                id="autoresearch_history",
                label="Trial history",
                custom_kind="autoresearch_history",
                params={
                    "source_rpc": "agent_autoresearch_state",
                    "poll_ms":    1000,
                },
                fields=[],
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
