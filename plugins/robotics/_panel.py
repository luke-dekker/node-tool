"""Robotics panel spec. Rendered by every GUI from a single declaration."""
from __future__ import annotations

from core.panel import (
    PanelSpec, FormSection, StatusSection, ButtonsSection, CustomSection,
    Field, Action,
)


def build_robotics_panel_spec() -> PanelSpec:
    return PanelSpec(
        label="Robotics",
        sections=[
            StatusSection(
                id="status",
                label="Serial",
                source_rpc="get_robotics_state",
                poll_ms=1000,
                fields=[
                    Field("connected", "str", label="Connected"),
                    Field("port",      "str", label="Port"),
                    Field("baud",      "str", label="Baud"),
                ],
            ),
            FormSection(
                id="conn",
                label="Port",
                fields=[
                    Field("port", "str", label="port",
                          hint="COM3, /dev/ttyUSB0, ..."),
                    Field("baud", "choice", label="baud", default="115200",
                          choices=["9600", "19200", "38400", "57600", "115200"]),
                ],
            ),
            ButtonsSection(
                id="controls",
                actions=[
                    Action(id="connect",    label="Connect",
                           rpc="robotics_connect", collect=["conn"]),
                    Action(id="disconnect", label="Disconnect",
                           rpc="robotics_disconnect"),
                ],
            ),
            CustomSection(
                id="serial_log",
                label="Serial Monitor",
                custom_kind="log_tail",
                params={"source_rpc": "get_robotics_log", "poll_ms": 500},
            ),
            FormSection(
                id="send",
                label="Send",
                fields=[
                    Field("cmd", "str", label="cmd",
                          hint="command to send"),
                ],
            ),
            ButtonsSection(
                id="send_ctrl",
                actions=[
                    Action(id="send", label="Send",
                           rpc="robotics_send", collect=["send"]),
                ],
            ),
        ],
    )
