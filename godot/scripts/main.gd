extends Control
## Main controller — builds the entire UI from code, manages WebSocket
## connection to Python backend, coordinates palette + graph + inspector.

const SERVER_URL := "ws://127.0.0.1:9800"

var ws := WebSocketPeer.new()
var _connected := false
var _req_id := 0
var _pending: Dictionary = {}  # id -> Callable

# UI references (built in _ready)
var status_label: Label
var node_graph: GraphEdit
var palette_list: VBoxContainer
var terminal: RichTextLabel
var search_box: LineEdit
var inspector_box: VBoxContainer

# Node registry cache from Python
var registry: Dictionary = {}
var category_order: Array = []

# Graph node tracking
var _nodes: Dictionary = {}       # python node_id -> GraphNode
var _name_to_id: Dictionary = {}  # GraphNode.name -> python node_id
var _spawn_count := 0
var _inspector_node_id: String = ""


func _ready() -> void:
	_build_ui()
	# Connect to Python server — increase buffer for large registry payload (~150KB)
	ws.inbound_buffer_size = 1 * 1024 * 1024  # 1 MB
	ws.max_queued_packets = 256
	var err := ws.connect_to_url(SERVER_URL)
	if err != OK:
		_log("[ERROR] Failed to initiate WebSocket connection")
		status_label.text = "Connection failed"
	else:
		status_label.text = "Connecting..."


func _process(_delta: float) -> void:
	ws.poll()
	var state := ws.get_ready_state()

	if state == WebSocketPeer.STATE_OPEN:
		if not _connected:
			_connected = true
			status_label.text = "Connected"
			_log("Connected to Python backend")
			_rpc("get_registry", {}, _on_registry_received)

		while ws.get_available_packet_count() > 0:
			var raw := ws.get_packet().get_string_from_utf8()
			_handle_message(raw)

	elif state == WebSocketPeer.STATE_CLOSED:
		if _connected:
			_connected = false
			status_label.text = "Disconnected — restart server"
			_log("Disconnected from server")


# ── UI Construction ──────────────────────────────────────────────────

func _build_ui() -> void:
	# Root vertical split: menu bar on top, main content below
	var root := VBoxContainer.new()
	root.set_anchors_and_offsets_preset(PRESET_FULL_RECT)
	add_child(root)

	# ── Menu bar ─────────────────────────────────────────────────────
	var menu := HBoxContainer.new()
	menu.custom_minimum_size.y = 36
	root.add_child(menu)

	var run_btn := Button.new()
	run_btn.text = "  Run Graph  "
	run_btn.pressed.connect(_on_run_pressed)
	menu.add_child(run_btn)

	var clear_btn := Button.new()
	clear_btn.text = "  Clear All  "
	clear_btn.pressed.connect(_on_clear_pressed)
	menu.add_child(clear_btn)

	var spacer := Control.new()
	spacer.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	menu.add_child(spacer)

	status_label = Label.new()
	status_label.text = "Starting..."
	menu.add_child(status_label)

	# ── Main content: palette | (graph + terminal) | inspector ───────
	var hsplit := HSplitContainer.new()
	hsplit.size_flags_vertical = Control.SIZE_EXPAND_FILL
	root.add_child(hsplit)

	# Left: Palette
	var palette_panel := VBoxContainer.new()
	palette_panel.custom_minimum_size.x = 220
	hsplit.add_child(palette_panel)

	search_box = LineEdit.new()
	search_box.placeholder_text = "Search nodes..."
	search_box.text_changed.connect(_on_search_changed)
	palette_panel.add_child(search_box)

	var palette_scroll := ScrollContainer.new()
	palette_scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	palette_panel.add_child(palette_scroll)

	palette_list = VBoxContainer.new()
	palette_list.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	palette_scroll.add_child(palette_list)

	# Center + right split
	var center_right := HSplitContainer.new()
	center_right.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	hsplit.add_child(center_right)

	# Center: graph + terminal vertical split
	var center_vsplit := VSplitContainer.new()
	center_vsplit.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	center_right.add_child(center_vsplit)

	# Graph editor
	node_graph = GraphEdit.new()
	node_graph.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	node_graph.size_flags_vertical = Control.SIZE_EXPAND_FILL
	node_graph.minimap_enabled = true
	node_graph.right_disconnects = true
	node_graph.connection_request.connect(_on_connection_request)
	node_graph.disconnection_request.connect(_on_disconnection_request)
	center_vsplit.add_child(node_graph)

	# Terminal
	var term_tabs := TabContainer.new()
	term_tabs.custom_minimum_size.y = 180
	center_vsplit.add_child(term_tabs)

	terminal = RichTextLabel.new()
	terminal.name = "Output"
	terminal.scroll_following = true
	term_tabs.add_child(terminal)

	# Right: Inspector
	var inspector_scroll := ScrollContainer.new()
	inspector_scroll.custom_minimum_size.x = 300
	center_right.add_child(inspector_scroll)

	inspector_box = VBoxContainer.new()
	inspector_box.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	inspector_scroll.add_child(inspector_box)

	var hint := Label.new()
	hint.text = "Select a node to inspect."
	hint.add_theme_color_override("font_color", Color(0.5, 0.55, 0.63))
	inspector_box.add_child(hint)

	_log("NodeTool Godot frontend ready.")


# ── WebSocket RPC ────────────────────────────────────────────────────

func _rpc(method: String, params: Dictionary, callback: Callable = Callable()) -> void:
	_req_id += 1
	var msg := {"jsonrpc": "2.0", "method": method, "params": params, "id": _req_id}
	if callback.is_valid():
		_pending[_req_id] = callback
	ws.send_text(JSON.stringify(msg))


func server_rpc(method: String, params: Dictionary, callback: Callable = Callable()) -> void:
	_rpc(method, params, callback)


func _handle_message(raw: String) -> void:
	_log("[DEBUG] Received %d bytes" % raw.length())
	var parsed = JSON.parse_string(raw)
	if parsed == null:
		_log("[ERROR] Invalid JSON from server (first 200 chars): %s" % raw.substr(0, 200))
		return
	var msg: Dictionary = parsed
	var id = msg.get("id")
	_log("[DEBUG] Message id=%s has_error=%s has_result=%s" % [str(id), str(msg.has("error")), str(msg.has("result"))])
	if msg.has("error"):
		var err = msg["error"]
		_log("[Error] %s" % str(err.get("message", err)))
		if id != null:
			_pending.erase(id)
		return
	if id != null and _pending.has(id):
		var cb: Callable = _pending[id]
		_pending.erase(id)
		_log("[DEBUG] Calling callback for id=%s" % str(id))
		cb.call(msg.get("result", {}))
	else:
		_log("[DEBUG] No pending callback for id=%s (pending keys: %s)" % [str(id), str(_pending.keys())])


func _log(text: String) -> void:
	if terminal:
		terminal.append_text(text + "\n")
	print(text)


# ── Registry / Palette ───────────────────────────────────────────────

func _on_registry_received(result: Dictionary) -> void:
	_log("[DEBUG] _on_registry_received called, result keys: %s" % str(result.keys()))
	registry = result.get("categories", {})
	category_order = result.get("category_order", [])
	var total := 0
	for cat in registry:
		total += registry[cat].size()
	_log("Registry loaded: %d categories, %d nodes" % [registry.size(), total])
	_build_palette()


func _build_palette() -> void:
	for child in palette_list.get_children():
		child.queue_free()

	for cat in category_order:
		if not registry.has(cat):
			continue
		var nodes_arr: Array = registry[cat]
		if nodes_arr.is_empty():
			continue

		var cat_label := Label.new()
		cat_label.text = "  %s" % cat
		cat_label.add_theme_color_override("font_color", Color(0.55, 0.7, 0.9))
		palette_list.add_child(cat_label)

		for node_def in nodes_arr:
			var btn := Button.new()
			btn.text = "    %s" % node_def["label"]
			btn.alignment = HORIZONTAL_ALIGNMENT_LEFT
			btn.set_meta("type_name", node_def["type_name"])
			btn.set_meta("node_def", node_def)
			btn.pressed.connect(_on_palette_btn.bind(node_def["type_name"]))
			palette_list.add_child(btn)

	_log("Palette built")


func _on_palette_btn(type_name: String) -> void:
	_rpc("add_node", {"type_name": type_name}, func(result: Dictionary):
		_add_graph_node(result)
		_log("Added: %s" % result.get("label", type_name))
	)


func _on_search_changed(text: String) -> void:
	var query := text.strip_edges().to_lower()
	for child in palette_list.get_children():
		if child is Button:
			child.visible = query.is_empty() or query in child.text.to_lower()


# ── Graph Node Management ────────────────────────────────────────────

func _add_graph_node(node_data: Dictionary) -> void:
	var node_id: String = node_data["id"]
	var gn := GraphNode.new()
	gn.title = node_data.get("label", "Node")
	gn.name = "gn_%s" % node_id.substr(0, 8)
	gn.set_meta("node_id", node_id)
	gn.set_meta("node_data", node_data)

	# Stagger position
	var px := 300 + (_spawn_count % 4) * 240
	var py := 100 + (_spawn_count / 4) * 160 + (_spawn_count % 4) * 30
	gn.position_offset = Vector2(px, py)
	_spawn_count += 1

	# Config summary (no port, just text)
	var config_parts: PackedStringArray = []
	var inputs: Dictionary = node_data.get("inputs", {})
	for port_name in inputs:
		var port: Dictionary = inputs[port_name]
		if port.get("editable", false):
			var val = port.get("default_value")
			if val != null and str(val) != "" and str(val) != "false" and str(val) != "False" and str(val) != "0":
				config_parts.append(str(val))
	if config_parts.size() > 0:
		var summary := Label.new()
		summary.text = ", ".join(config_parts.slice(0, 5))
		summary.add_theme_color_override("font_color", Color(0.55, 0.55, 0.63))
		summary.add_theme_font_size_override("font_size", 11)
		gn.add_child(summary)

	# Data input ports (left side pins)
	var input_slot_count := 0
	for port_name in inputs:
		var port: Dictionary = inputs[port_name]
		if port.get("editable", false):
			continue
		var lbl := Label.new()
		lbl.text = port_name
		var col := _color_from_array(port.get("color", [160, 160, 180, 255]))
		lbl.add_theme_color_override("font_color", col)
		gn.add_child(lbl)
		var idx := gn.get_child_count() - 1
		gn.set_slot(idx, true, 0, col, false, 0, Color.WHITE)
		input_slot_count += 1

	# Output ports (right side pins)
	var outputs: Dictionary = node_data.get("outputs", {})
	for port_name in outputs:
		var port: Dictionary = outputs[port_name]
		var lbl := Label.new()
		lbl.text = port_name
		lbl.name = "out_%s" % port_name
		lbl.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
		var col := _color_from_array(port.get("color", [160, 160, 180, 255]))
		lbl.add_theme_color_override("font_color", col)
		gn.add_child(lbl)
		var idx := gn.get_child_count() - 1
		gn.set_slot(idx, false, 0, Color.WHITE, true, 0, col)

	node_graph.add_child(gn)
	_nodes[node_id] = gn
	_name_to_id[gn.name] = node_id

	gn.node_selected.connect(_on_node_selected.bind(node_id))


func _on_node_selected(node_id: String) -> void:
	var gn = _nodes.get(node_id)
	if gn:
		_show_inspector(gn.get_meta("node_data"), node_id)


# ── Inspector ────────────────────────────────────────────────────────

func _show_inspector(node_data: Dictionary, node_id: String) -> void:
	_inspector_node_id = node_id
	for child in inspector_box.get_children():
		child.queue_free()

	# Title
	var title := Label.new()
	title.text = node_data.get("label", "Node")
	title.add_theme_font_size_override("font_size", 18)
	inspector_box.add_child(title)

	var cat_lbl := Label.new()
	cat_lbl.text = node_data.get("category", "")
	cat_lbl.add_theme_color_override("font_color", Color(0.5, 0.55, 0.63))
	inspector_box.add_child(cat_lbl)

	var desc := Label.new()
	desc.text = node_data.get("description", "")
	desc.autowrap_mode = TextServer.AUTOWRAP_WORD
	desc.add_theme_color_override("font_color", Color(0.5, 0.55, 0.63))
	inspector_box.add_child(desc)

	inspector_box.add_child(HSeparator.new())

	# Config inputs (editable widgets)
	var inputs: Dictionary = node_data.get("inputs", {})
	var has_config := false
	for port_name in inputs:
		var port: Dictionary = inputs[port_name]
		if not port.get("editable", false):
			continue
		if not has_config:
			var hdr := Label.new()
			hdr.text = "Config"
			hdr.add_theme_color_override("font_color", Color(0.35, 0.77, 0.96))
			inspector_box.add_child(hdr)
			has_config = true

		var val = port.get("default_value")
		var port_type: String = port.get("port_type", "STRING")
		var choices = port.get("choices")

		var hbox := HBoxContainer.new()
		var lbl := Label.new()
		lbl.text = port_name
		lbl.custom_minimum_size.x = 90
		hbox.add_child(lbl)

		if port_type == "BOOL":
			var cb := CheckBox.new()
			cb.button_pressed = bool(val) if val != null else false
			cb.toggled.connect(func(v): _set_input(node_id, port_name, v))
			hbox.add_child(cb)
		elif port_type == "INT":
			var spin := SpinBox.new()
			spin.min_value = -999999
			spin.max_value = 999999
			spin.step = 1
			spin.value = int(val) if val != null else 0
			spin.size_flags_horizontal = Control.SIZE_EXPAND_FILL
			spin.value_changed.connect(func(v): _set_input(node_id, port_name, int(v)))
			hbox.add_child(spin)
		elif port_type == "FLOAT":
			var spin := SpinBox.new()
			spin.min_value = -999999.0
			spin.max_value = 999999.0
			spin.step = 0.001
			spin.value = float(val) if val != null else 0.0
			spin.size_flags_horizontal = Control.SIZE_EXPAND_FILL
			spin.value_changed.connect(func(v): _set_input(node_id, port_name, v))
			hbox.add_child(spin)
		elif port_type == "STRING" and choices != null and choices is Array and choices.size() > 0:
			var opt := OptionButton.new()
			for choice in choices:
				opt.add_item(str(choice))
			var current := str(val) if val != null else ""
			for i in range(opt.item_count):
				if opt.get_item_text(i) == current:
					opt.select(i)
					break
			opt.size_flags_horizontal = Control.SIZE_EXPAND_FILL
			opt.item_selected.connect(func(idx): _set_input(node_id, port_name, opt.get_item_text(idx)))
			hbox.add_child(opt)
		else:
			var line := LineEdit.new()
			line.text = str(val) if val != null else ""
			line.size_flags_horizontal = Control.SIZE_EXPAND_FILL
			line.text_submitted.connect(func(t): _set_input(node_id, port_name, t))
			hbox.add_child(line)

		inspector_box.add_child(hbox)

	# Outputs
	var out_ports: Dictionary = node_data.get("outputs", {})
	if not out_ports.is_empty():
		inspector_box.add_child(HSeparator.new())
		var out_hdr := Label.new()
		out_hdr.text = "Outputs"
		out_hdr.add_theme_color_override("font_color", Color(0.5, 0.55, 0.63))
		inspector_box.add_child(out_hdr)
		for pname in out_ports:
			var col_arr: Array = out_ports[pname].get("color", [160, 160, 180, 255])
			var ol := Label.new()
			ol.text = "  %s" % pname
			ol.add_theme_color_override("font_color", Color(
				col_arr[0] / 255.0, col_arr[1] / 255.0, col_arr[2] / 255.0))
			inspector_box.add_child(ol)


func _set_input(node_id: String, port_name: String, value) -> void:
	_rpc("set_input", {"node_id": node_id, "port_name": port_name, "value": value})


# ── Connections ──────────────────────────────────────────────────────

func _on_connection_request(from_node: StringName, from_port: int, to_node: StringName, to_port: int) -> void:
	node_graph.connect_node(from_node, from_port, to_node, to_port)
	var from_id: String = str(_name_to_id.get(str(from_node), ""))
	var to_id: String = str(_name_to_id.get(str(to_node), ""))
	if from_id.is_empty() or to_id.is_empty():
		return
	var from_port_name := _get_output_port_name(from_node, from_port)
	var to_port_name := _get_input_port_name(to_node, to_port)
	if from_port_name.is_empty() or to_port_name.is_empty():
		return
	_rpc("connect", {
		"from_node": from_id, "from_port": from_port_name,
		"to_node": to_id, "to_port": to_port_name,
	})


func _on_disconnection_request(from_node: StringName, from_port: int, to_node: StringName, to_port: int) -> void:
	node_graph.disconnect_node(from_node, from_port, to_node, to_port)
	var from_id: String = str(_name_to_id.get(str(from_node), ""))
	var to_id: String = str(_name_to_id.get(str(to_node), ""))
	if from_id.is_empty() or to_id.is_empty():
		return
	var from_port_name := _get_output_port_name(from_node, from_port)
	var to_port_name := _get_input_port_name(to_node, to_port)
	_rpc("disconnect", {
		"from_node": from_id, "from_port": from_port_name,
		"to_node": to_id, "to_port": to_port_name,
	})


func _get_output_port_name(node_name: StringName, port_idx: int) -> String:
	var gn = node_graph.get_node_or_null(NodePath(str(node_name)))
	if gn == null:
		return ""
	var data: Dictionary = gn.get_meta("node_data", {})
	var keys: Array = data.get("outputs", {}).keys()
	if port_idx >= 0 and port_idx < keys.size():
		return keys[port_idx]
	return ""


func _get_input_port_name(node_name: StringName, port_idx: int) -> String:
	var gn = node_graph.get_node_or_null(NodePath(str(node_name)))
	if gn == null:
		return ""
	var data: Dictionary = gn.get_meta("node_data", {})
	var inputs: Dictionary = data.get("inputs", {})
	var data_ports: Array = []
	for pname in inputs:
		if not inputs[pname].get("editable", false):
			data_ports.append(pname)
	if port_idx >= 0 and port_idx < data_ports.size():
		return data_ports[port_idx]
	return ""


# ── Actions ──────────────────────────────────────────────────────────

func _on_run_pressed() -> void:
	_log("--- Running graph ---")
	_rpc("execute", {}, func(result: Dictionary):
		if result.has("error"):
			_log("[Error] %s" % str(result["error"]))
		for line in result.get("terminal", []):
			_log(str(line))
		var outputs: Dictionary = result.get("outputs", {})
		_log("Done: %d nodes produced output" % outputs.size())
		# Update output labels on canvas nodes
		for nid in outputs:
			var gn = _nodes.get(nid)
			if gn == null:
				continue
			for pname in outputs[nid]:
				var lbl_node = gn.find_child("out_%s" % pname, false)
				if lbl_node and lbl_node is Label:
					lbl_node.text = "%s: %s" % [pname, str(outputs[nid][pname])]
	)


func _on_clear_pressed() -> void:
	_rpc("clear", {}, func(_result: Dictionary):
		node_graph.clear_connections()
		for gn in _nodes.values():
			if is_instance_valid(gn):
				gn.queue_free()
		_nodes.clear()
		_name_to_id.clear()
		_spawn_count = 0
		_log("Graph cleared")
	)


# ── Helpers ──────────────────────────────────────────────────────────

func _color_from_array(arr: Array) -> Color:
	return Color(
		arr[0] / 255.0 if arr.size() > 0 else 0.6,
		arr[1] / 255.0 if arr.size() > 1 else 0.6,
		arr[2] / 255.0 if arr.size() > 2 else 0.7,
		arr[3] / 255.0 if arr.size() > 3 else 1.0)
