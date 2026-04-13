extends VBoxContainer
## Inspector panel — shows editable config widgets for the selected node.
## Data ports and outputs are displayed as read-only labels.

var _current_node_id: String = ""

func _ready() -> void:
	add_to_group("inspector")
	_show_empty()


func _show_empty() -> void:
	_clear()
	var label := Label.new()
	label.text = "Select a node to inspect."
	label.add_theme_color_override("font_color", Color(0.5, 0.55, 0.63))
	add_child(label)


func show_node(node_data: Dictionary, node_id: String) -> void:
	_clear()
	_current_node_id = node_id

	# Title
	var title := Label.new()
	title.text = node_data.get("label", "Node")
	title.add_theme_font_size_override("font_size", 18)
	add_child(title)

	# Category
	var cat_label := Label.new()
	cat_label.text = node_data.get("category", "")
	cat_label.add_theme_color_override("font_color", Color(0.5, 0.55, 0.63))
	add_child(cat_label)

	# Description
	var desc := Label.new()
	desc.text = node_data.get("description", "")
	desc.autowrap_mode = TextServer.AUTOWRAP_WORD
	desc.add_theme_color_override("font_color", Color(0.5, 0.55, 0.63))
	add_child(desc)

	add_child(HSeparator.new())

	# ── Config inputs (editable widgets) ─────────────────────────────
	var inputs: Dictionary = node_data.get("inputs", {})
	var has_config := false
	for port_name in inputs:
		var port: Dictionary = inputs[port_name]
		if not port.get("editable", false):
			continue
		if not has_config:
			var header := Label.new()
			header.text = "Config"
			header.add_theme_color_override("font_color", Color(0.35, 0.77, 0.96))
			add_child(header)
			has_config = true

		var val = port.get("default_value")
		var port_type: String = port.get("port_type", "STRING")
		var choices = port.get("choices")

		if port_type == "BOOL":
			var cb := CheckBox.new()
			cb.text = port_name
			cb.button_pressed = bool(val) if val != null else false
			cb.toggled.connect(func(v): _set_input(node_id, port_name, v))
			add_child(cb)

		elif port_type == "INT":
			var hbox := HBoxContainer.new()
			var lbl := Label.new()
			lbl.text = port_name
			lbl.custom_minimum_size.x = 100
			hbox.add_child(lbl)
			var spin := SpinBox.new()
			spin.min_value = -999999
			spin.max_value = 999999
			spin.step = 1
			spin.value = int(val) if val != null else 0
			spin.size_flags_horizontal = Control.SIZE_EXPAND_FILL
			spin.value_changed.connect(func(v): _set_input(node_id, port_name, int(v)))
			hbox.add_child(spin)
			add_child(hbox)

		elif port_type == "FLOAT":
			var hbox := HBoxContainer.new()
			var lbl := Label.new()
			lbl.text = port_name
			lbl.custom_minimum_size.x = 100
			hbox.add_child(lbl)
			var spin := SpinBox.new()
			spin.min_value = -999999.0
			spin.max_value = 999999.0
			spin.step = 0.001
			spin.value = float(val) if val != null else 0.0
			spin.size_flags_horizontal = Control.SIZE_EXPAND_FILL
			spin.value_changed.connect(func(v): _set_input(node_id, port_name, v))
			hbox.add_child(spin)
			add_child(hbox)

		elif port_type == "STRING" and choices != null and choices is Array:
			var hbox := HBoxContainer.new()
			var lbl := Label.new()
			lbl.text = port_name
			lbl.custom_minimum_size.x = 100
			hbox.add_child(lbl)
			var opt := OptionButton.new()
			for choice in choices:
				opt.add_item(str(choice))
			# Select current value
			var current := str(val) if val != null else ""
			for i in range(opt.item_count):
				if opt.get_item_text(i) == current:
					opt.select(i)
					break
			opt.size_flags_horizontal = Control.SIZE_EXPAND_FILL
			opt.item_selected.connect(func(idx): _set_input(node_id, port_name, opt.get_item_text(idx)))
			hbox.add_child(opt)
			add_child(hbox)

		else:  # STRING without choices
			var hbox := HBoxContainer.new()
			var lbl := Label.new()
			lbl.text = port_name
			lbl.custom_minimum_size.x = 100
			hbox.add_child(lbl)
			var line := LineEdit.new()
			line.text = str(val) if val != null else ""
			line.size_flags_horizontal = Control.SIZE_EXPAND_FILL
			line.text_submitted.connect(func(t): _set_input(node_id, port_name, t))
			hbox.add_child(line)
			add_child(hbox)

	# ── Data inputs (read-only) ──────────────────────────────────────
	var has_data := false
	for port_name in inputs:
		var port: Dictionary = inputs[port_name]
		if port.get("editable", false):
			continue
		if not has_data:
			add_child(HSeparator.new())
			var header := Label.new()
			header.text = "Data Inputs"
			header.add_theme_color_override("font_color", Color(0.5, 0.55, 0.63))
			add_child(header)
			has_data = true
		var lbl := Label.new()
		var col_arr: Array = port.get("color", [160, 160, 180, 255])
		lbl.text = "  %s" % port_name
		lbl.add_theme_color_override("font_color", Color(
			col_arr[0] / 255.0, col_arr[1] / 255.0, col_arr[2] / 255.0))
		add_child(lbl)

	# ── Outputs (read-only) ──────────────────────────────────────────
	var outputs: Dictionary = node_data.get("outputs", {})
	if not outputs.is_empty():
		add_child(HSeparator.new())
		var header := Label.new()
		header.text = "Outputs"
		header.add_theme_color_override("font_color", Color(0.5, 0.55, 0.63))
		add_child(header)
		for port_name in outputs:
			var port: Dictionary = outputs[port_name]
			var col_arr: Array = port.get("color", [160, 160, 180, 255])
			var lbl := Label.new()
			lbl.text = "  %s" % port_name
			lbl.add_theme_color_override("font_color", Color(
				col_arr[0] / 255.0, col_arr[1] / 255.0, col_arr[2] / 255.0))
			add_child(lbl)


func _set_input(node_id: String, port_name: String, value) -> void:
	var main = get_tree().root.get_child(0)
	if main.has_method("rpc"):
		main.rpc("set_input", {
			"node_id": node_id,
			"port_name": port_name,
			"value": value,
		})


func _clear() -> void:
	for child in get_children():
		child.queue_free()
