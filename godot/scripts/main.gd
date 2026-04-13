extends VBoxContainer
## Main controller — manages WebSocket connection to Python backend
## and coordinates the palette, graph editor, and inspector.

const SERVER_URL := "ws://127.0.0.1:9800"

var ws := WebSocketPeer.new()
var _connected := false
var _req_id := 0
var _pending: Dictionary = {}  # id -> Callable

@onready var status_label: Label = %"StatusLabel" if has_node("%StatusLabel") else $MenuBar/StatusLabel
@onready var node_graph: GraphEdit = $HSplit/CenterRight/CenterVSplit/NodeGraph
@onready var palette_list: VBoxContainer = $HSplit/Palette/PaletteScroll/PaletteList
@onready var terminal: RichTextLabel = $HSplit/CenterRight/CenterVSplit/Terminal/Output
@onready var search_box: LineEdit = $HSplit/Palette/SearchBox
@onready var run_btn: Button = $MenuBar/RunBtn
@onready var clear_btn: Button = $MenuBar/ClearBtn

# Node registry cache from Python
var registry: Dictionary = {}
var category_order: Array = []


func _ready() -> void:
	run_btn.pressed.connect(_on_run_pressed)
	clear_btn.pressed.connect(_on_clear_pressed)
	search_box.text_changed.connect(_on_search_changed)

	# Connect to Python server
	var err := ws.connect_to_url(SERVER_URL)
	if err != OK:
		_log("Failed to initiate WebSocket connection")
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
			# Fetch node registry
			rpc("get_registry", {}, _on_registry_received)

		# Process incoming messages
		while ws.get_available_packet_count() > 0:
			var raw := ws.get_packet().get_string_from_utf8()
			_handle_message(raw)

	elif state == WebSocketPeer.STATE_CLOSED:
		if _connected:
			_connected = false
			status_label.text = "Disconnected"
			_log("Disconnected from server")


## Send a JSON-RPC request and register a callback for the response.
func rpc(method: String, params: Dictionary, callback: Callable = Callable()) -> void:
	_req_id += 1
	var msg := {
		"jsonrpc": "2.0",
		"method": method,
		"params": params,
		"id": _req_id,
	}
	if callback.is_valid():
		_pending[_req_id] = callback
	ws.send_text(JSON.stringify(msg))


func _handle_message(raw: String) -> void:
	var parsed = JSON.parse_string(raw)
	if parsed == null:
		_log("Invalid JSON from server")
		return

	var msg: Dictionary = parsed
	var id = msg.get("id")

	if msg.has("error"):
		var err = msg["error"]
		_log("[Error] %s" % str(err.get("message", err)))
		_pending.erase(id)
		return

	if id != null and _pending.has(id):
		var cb: Callable = _pending[id]
		_pending.erase(id)
		cb.call(msg.get("result", {}))


## Log a message to the terminal panel.
func _log(text: String) -> void:
	if terminal:
		terminal.append_text(text + "\n")


# ── Registry / Palette ───────────────────────────────────────────────

func _on_registry_received(result: Dictionary) -> void:
	registry = result.get("categories", {})
	category_order = result.get("category_order", [])
	_log("Registry: %d categories" % registry.size())
	_build_palette()


func _build_palette() -> void:
	# Clear existing
	for child in palette_list.get_children():
		child.queue_free()

	for cat in category_order:
		if not registry.has(cat):
			continue
		var nodes: Array = registry[cat]
		if nodes.is_empty():
			continue

		# Category label
		var label := Label.new()
		label.text = cat
		label.add_theme_color_override("font_color", Color(0.6, 0.75, 0.9))
		palette_list.add_child(label)

		# Node buttons
		for node_def in nodes:
			var btn := Button.new()
			btn.text = node_def["label"]
			btn.alignment = HORIZONTAL_ALIGNMENT_LEFT
			btn.set_meta("type_name", node_def["type_name"])
			btn.set_meta("node_def", node_def)
			btn.pressed.connect(_on_palette_btn.bind(node_def["type_name"]))
			palette_list.add_child(btn)

		# Spacer between categories
		var spacer := Control.new()
		spacer.custom_minimum_size.y = 8
		palette_list.add_child(spacer)

	_log("Palette built: %d categories" % category_order.size())


func _on_palette_btn(type_name: String) -> void:
	rpc("add_node", {"type_name": type_name}, func(result: Dictionary):
		node_graph.add_graph_node(result)
		_log("Added: %s" % result.get("label", type_name))
	)


func _on_search_changed(text: String) -> void:
	var query := text.strip_edges().to_lower()
	for child in palette_list.get_children():
		if child is Button:
			child.visible = query.is_empty() or query in child.text.to_lower()
		elif child is Label:
			# Show category label if any child button under it is visible
			child.visible = query.is_empty()


# ── Actions ──────────────────────────────────────────────────────────

func _on_run_pressed() -> void:
	_log("--- Running graph ---")
	rpc("execute", {}, func(result: Dictionary):
		if result.has("error"):
			_log("[Error] %s" % result["error"])
		var lines: Array = result.get("terminal", [])
		for line in lines:
			_log(str(line))
		var outputs: Dictionary = result.get("outputs", {})
		_log("Execution complete: %d nodes produced output" % outputs.size())
		# Update output displays on graph nodes
		node_graph.update_outputs(outputs)
	)


func _on_clear_pressed() -> void:
	rpc("clear", {}, func(_result: Dictionary):
		node_graph.clear_all()
		_log("Graph cleared")
	)
