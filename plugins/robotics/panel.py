"""Robotics panel — serial monitor + live sensor plot.

Registers as a tab in the bottom panel bar. Shows:
  - Serial port selector + connect/disconnect
  - Incoming serial data log
  - Send command field
  - Live sensor value plot (reads from the graph's last execution)
"""
from __future__ import annotations


def build_robotics_panel(parent_tag, app) -> None:
    """Build the Robotics tab content. Called by the plugin system."""
    import dearpygui.dearpygui as dpg

    with dpg.group(horizontal=True, parent=parent_tag):
        # Left half: serial monitor
        with dpg.child_window(width=400, height=-1, border=True):
            dpg.add_text("Serial Monitor", color=[88, 196, 245])
            with dpg.group(horizontal=True):
                dpg.add_combo(label="Port", tag="rob_serial_port",
                              items=_list_ports(), width=150)
                dpg.add_combo(label="Baud", tag="rob_serial_baud",
                              items=["9600", "19200", "38400", "57600", "115200"],
                              default_value="115200", width=80)
                dpg.add_button(label="Refresh",
                               callback=lambda: _refresh_ports())
            with dpg.group(horizontal=True):
                dpg.add_button(label="Connect", tag="rob_connect_btn",
                               callback=lambda: app._log("[Robotics] Serial connect not yet wired to hardware"))
                dpg.add_button(label="Disconnect",
                               callback=lambda: app._log("[Robotics] Disconnected"))
            dpg.add_separator()
            dpg.add_input_text(tag="rob_serial_log", multiline=True,
                               readonly=True, height=80, width=-1,
                               default_value="(serial output will appear here)")
            with dpg.group(horizontal=True):
                dpg.add_input_text(tag="rob_serial_send", hint="Send command...",
                                   width=-80, on_enter=True,
                                   callback=lambda s, a: _send_cmd(a, app))
                dpg.add_button(label="Send",
                               callback=lambda: _send_cmd(
                                   dpg.get_value("rob_serial_send"), app))

        # Right half: live sensor plot
        dpg.add_spacer(width=8, parent=parent_tag)
        with dpg.child_window(width=-1, height=-1, border=True, parent=parent_tag):
            dpg.add_text("Live Sensor Plot", color=[88, 196, 245])
            dpg.add_text("Wire sensor nodes to see live data here.",
                         color=[140, 148, 162])
            with dpg.plot(label="Sensors", height=-1, width=-1,
                          tag="rob_sensor_plot"):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="tick",
                                  tag="rob_plot_x")
                with dpg.plot_axis(dpg.mvYAxis, label="value",
                                    tag="rob_plot_y"):
                    dpg.add_line_series([], [], label="sensor_0",
                                         tag="rob_plot_s0")
                    dpg.add_line_series([], [], label="sensor_1",
                                         tag="rob_plot_s1")


def _list_ports() -> list[str]:
    """List available serial ports."""
    try:
        import serial.tools.list_ports
        return [p.device for p in serial.tools.list_ports.comports()] or ["(none)"]
    except ImportError:
        return ["(pyserial not installed)"]


def _refresh_ports() -> None:
    import dearpygui.dearpygui as dpg
    try:
        ports = _list_ports()
        dpg.configure_item("rob_serial_port", items=ports)
    except Exception:
        pass


def _send_cmd(text: str, app) -> None:
    import dearpygui.dearpygui as dpg
    if not text:
        return
    app._log(f"[Serial TX] {text}")
    try:
        current = dpg.get_value("rob_serial_log") or ""
        dpg.set_value("rob_serial_log", current + f"> {text}\n")
        dpg.set_value("rob_serial_send", "")
    except Exception:
        pass
