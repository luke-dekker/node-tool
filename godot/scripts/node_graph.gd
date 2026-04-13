extends GraphEdit
## Node graph editor — spawns GraphNodes from Python backend data,
## handles connections, and reports changes back via RPC.

# Maps Python node_id -> GraphNode instance
var _nodes: Dictionary = {}
# Maps GraphNode name -> Python node_id
var _name_to_id: Dictionary = {}

var _spawn_count := 0


func _ready() -> void:
	connection_request.connect(_on_connection_request)
	disconnection_request.connect(_on_disconnection_request)


## Create a GraphNode from a Python node dict (returned by add_node RPC).
func add_graph_node(node_data: Dictionary) -> void:
	var node_id: String = node_data["id"]
	var gn := GraphNode.new()
	gn.title = node_data.get("label", "Node")
	gn.name = "node_%s" % node_id.substr(0, 8)  # short unique name
	gn.set_meta("node_id", node_id)
	gn.set_meta("node_data", node_data)

	# Position with stagger
	var px := 300 + (_spawn_count % 4) * 240
	var py := 100 + (_spawn_count / 4) * 160 + (_spawn_count % 4) * 30
	gn.position_offset = Vector2(px, py)
	_spawn_count += 1

	# ── Config summary (static text, no port) ────────────────────────
	var config_parts: PackedStringArray = []
	var inputs: Dictionary = node_data.get("inputs", {})
	for port_name in inputs:
		var port: Dictionary = inputs[port_name]
		if port.get("editable", false):
			var val = port.get("default_value")
			if val != null and str(val) != "" and str(val) != "false" and str(val) != "False":
				config_parts.append(str(val))
	if config_parts.size() > 0:
		var summary_label := Label.new()
		summary_label.text = ", ".join(config_parts.slice(0, 5))
		summary_label.add_theme_color_override("font_color", Color(0.55, 0.55, 0.63))
		summary_label.add_theme_font_size_override("font_size", 12)
		gn.add_child(summary_label)

	# ── Data input ports (left side) ─────────────────────────────────
	var slot_idx := gn.get_child_count()
	for port_name in inputs:
		var port: Dictionary = inputs[port_name]
		if port.get("editable", false):
			continue  # config ports go to inspector, not canvas
		var label := Label.new()
		label.text = port_name
		var col := _color_from_array(port.get("color", [160, 160, 180, 255]))
		label.add_theme_color_override("font_color", col)
		gn.add_child(label)
		var idx := gn.get_child_count() - 1
		gn.set_slot(idx,
			true,   # left enabled (input)
			0,      # left type
			col,    # left color
			false,  # right disabled
			0, Color.WHITE)

	# ── Output ports (right side) ────────────────────────────────────
	var outputs: Dictionary = node_data.get("outputs", {})
	for port_name in outputs:
		var port: Dictionary = outputs[port_name]
		var label := Label.new()
		label.text = port_name
		label.name = "out_%s" % port_name
		label.horizontal_alignment = HORIZONTAL_ALIGNMENT_RIGHT
		var col := _color_from_array(port.get("color", [160, 160, 180, 255]))
		label.add_theme_color_override("font_color", col)
		gn.add_child(label)
		var idx := gn.get_child_count() - 1
		gn.set_slot(idx,
			false,  # left disabled
			0, Color.WHITE,
			true,   # right enabled (output)
			0,      # right type
			col)    # right color

	add_child(gn)
	_nodes[node_id] = gn
	_name_to_id[gn.name] = node_id

	# When selected, tell inspector to rebuild
	gn.node_selected.connect(func(): _on_node_selected(node_id))


## Update output labels on nodes after execution.
func update_outputs(outputs: Dictionary) -> void:
	for node_id in outputs:
		var gn = _nodes.get(node_id)
		if gn == null:
			continue
		var node_outputs: Dictionary = outputs[node_id]
		for port_name in node_outputs:
			var label_node = gn.find_child("out_%s" % port_name, false)
			if label_node and label_node is Label:
				label_node.text = "%s: %s" % [port_name, str(node_outputs[port_name])]


## Clear all nodes from the canvas.
func clear_all() -> void:
	clear_connections()
	for gn in _nodes.values():
		if is_instance_valid(gn):
			gn.queue_free()
	_nodes.clear()
	_name_to_id.clear()
	_spawn_count = 0


func _on_node_selected(node_id: String) -> void:
	# Find the inspector and rebuild it for this node
	var inspector = get_tree().get_first_node_in_group("inspector")
	if inspector and inspector.has_method("show_node"):
		var gn = _nodes.get(node_id)
		if gn:
			inspector.show_node(gn.get_meta("node_data"), node_id)


func _on_connection_request(from_node: StringName, from_port: int, to_node: StringName, to_port: int) -> void:
	# Create visual connection
	connect_node(from_node, from_port, to_node, to_port)

	# Tell Python backend
	var from_id := _name_to_id.get(str(from_node), "")
	var to_id := _name_to_id.get(str(to_node), "")
	if from_id.is_empty() or to_id.is_empty():
		return

	# Map port indices to port names
	var from_port_name := _get_output_port_name(from_node, from_port)
	var to_port_name := _get_input_port_name(to_node, to_port)
	if from_port_name.is_empty() or to_port_name.is_empty():
		return

	var main = get_parent().get_parent().get_parent()  # Main node
	if main.has_method("rpc"):
		main.rpc("connect", {
			"from_node": from_id,
			"from_port": from_port_name,
			"to_node": to_id,
			"to_port": to_port_name,
		})


func _on_disconnection_request(from_node: StringName, from_port: int, to_node: StringName, to_port: int) -> void:
	disconnect_node(from_node, from_port, to_node, to_port)

	var from_id := _name_to_id.get(str(from_node), "")
	var to_id := _name_to_id.get(str(to_node), "")
	if from_id.is_empty() or to_id.is_empty():
		return

	var from_port_name := _get_output_port_name(from_node, from_port)
	var to_port_name := _get_input_port_name(to_node, to_port)

	var main = get_parent().get_parent().get_parent()
	if main.has_method("rpc"):
		main.rpc("disconnect", {
			"from_node": from_id,
			"from_port": from_port_name,
			"to_node": to_id,
			"to_port": to_port_name,
		})


## Map a visual output slot index to the port name string.
func _get_output_port_name(node_name: StringName, port_idx: int) -> String:
	var gn = get_node_or_null(NodePath(str(node_name)))
	if gn == null:
		return ""
	var data: Dictionary = gn.get_meta("node_data", {})
	var outputs: Dictionary = data.get("outputs", {})
	var keys := outputs.keys()
	if port_idx >= 0 and port_idx < keys.size():
		return keys[port_idx]
	return ""


## Map a visual input slot index to the data port name string.
func _get_input_port_name(node_name: StringName, port_idx: int) -> String:
	var gn = get_node_or_null(NodePath(str(node_name)))
	if gn == null:
		return ""
	var data: Dictionary = gn.get_meta("node_data", {})
	var inputs: Dictionary = data.get("inputs", {})
	# Only data (non-editable) ports have slots on the canvas
	var data_port_names: Array = []
	for pname in inputs:
		if not inputs[pname].get("editable", false):
			data_port_names.append(pname)
	if port_idx >= 0 and port_idx < data_port_names.size():
		return data_port_names[port_idx]
	return ""


func _color_from_array(arr: Array) -> Color:
	return Color(
		arr[0] / 255.0 if arr.size() > 0 else 0.6,
		arr[1] / 255.0 if arr.size() > 1 else 0.6,
		arr[2] / 255.0 if arr.size() > 2 else 0.7,
		arr[3] / 255.0 if arr.size() > 3 else 1.0,
	)
