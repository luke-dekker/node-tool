extends Control
## Main controller — builds the entire UI from code, manages WebSocket
## connection to Python backend, coordinates palette + graph + inspector.

const SERVER_URL := "ws://127.0.0.1:9800"

# ── Dark theme palette (matches DPG version) ─────────────────────────
const BG_DARK    := Color(0.047, 0.055, 0.078)   # canvas / deepest
const BG_MID     := Color(0.078, 0.090, 0.122)   # window bg
const BG_LIGHT   := Color(0.110, 0.125, 0.165)   # frame bg (inputs)
const BG_RAISED  := Color(0.133, 0.153, 0.204)   # hovered / raised
const BG_HEADER  := Color(0.094, 0.110, 0.149)   # section header bar
const ACCENT     := Color(0.345, 0.769, 0.961)   # primary cyan
const ACCENT_DIM := Color(0.216, 0.471, 0.627)
const TXT        := Color(0.878, 0.894, 0.933)
const TXT_DIM    := Color(0.518, 0.557, 0.635)
const TXT_BRIGHT := Color(0.980, 0.988, 1.000)
const BORDER_COL := Color(0.173, 0.204, 0.282, 0.78)
const OK_GREEN   := Color(0.314, 0.784, 0.510)
const ERR_RED    := Color(0.922, 0.353, 0.353)

const CATEGORY_COLORS := {
	"Python":  Color(0.392, 0.706, 0.314),
	"PyTorch": Color(0.863, 0.353, 0.118),
	"NumPy":   Color(0.275, 0.510, 0.784),
	"Pandas":  Color(0.549, 0.392, 0.784),
	"Sklearn": Color(0.902, 0.549, 0.157),
	"SciPy":   Color(0.235, 0.706, 0.510),
	"Viz":     Color(0.863, 0.235, 0.471),
}

var ws := WebSocketPeer.new()
var _connected := false
var _req_id := 0
var _pending: Dictionary = {}  # id -> Callable

# UI references (built in _ready)
var status_label: Label
var node_graph: GraphEdit
var palette_list: VBoxContainer
var terminal: RichTextLabel
var code_text: RichTextLabel
var search_box: LineEdit
var inspector_box: VBoxContainer
# Training tab widgets
var train_status_label: Label
var train_epoch_label: Label
var train_loss_label: Label
var train_epochs_spin: SpinBox
var train_lr_spin: SpinBox
var train_optim_btn: OptionButton
var train_loss_btn: OptionButton
var train_device_btn: OptionButton
var train_dataset_box: VBoxContainer

# Node registry cache from Python
var registry: Dictionary = {}
var category_order: Array = []

# Graph node tracking
var _nodes: Dictionary = {}       # python node_id -> GraphNode
var _name_to_id: Dictionary = {}  # GraphNode.name -> python node_id
var _spawn_count := 0
var _inspector_node_id: String = ""
var _save_path: String = ""
var _templates_menu: PopupMenu
var _template_labels: Array = []  # ordered list of template labels
var _term_tabs: TabContainer       # bottom panel tab bar
# Plugin panel builders: panel_name -> Callable(parent: TabContainer)
var _panel_builders: Dictionary = {}


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


# ── Dark Theme ───────────────────────────────────────────────────────

func _create_dark_theme() -> Theme:
	var t := Theme.new()

	# Default font color
	t.set_color("font_color", "Label", TXT)
	t.set_color("font_color", "Button", TXT)
	t.set_color("font_hover_color", "Button", TXT_BRIGHT)
	t.set_color("font_pressed_color", "Button", ACCENT)
	t.set_color("font_color", "LineEdit", TXT)
	t.set_color("font_placeholder_color", "LineEdit", TXT_DIM)
	t.set_color("font_color", "CheckBox", TXT)
	t.set_color("font_color", "OptionButton", TXT)
	t.set_color("font_color", "SpinBox", TXT)
	t.set_color("font_color", "MenuBar", TXT)
	t.set_color("font_color", "PopupMenu", TXT)
	t.set_color("font_hover_color", "PopupMenu", TXT_BRIGHT)
	t.set_color("font_color", "TabContainer", TXT_DIM)
	t.set_color("font_selected_color", "TabContainer", TXT_BRIGHT)
	t.set_color("font_hovered_color", "TabContainer", ACCENT)

	# Panels / backgrounds
	var panel_bg := _flat_box(BG_MID)
	t.set_stylebox("panel", "Panel", panel_bg)
	t.set_stylebox("panel", "PanelContainer", panel_bg)
	t.set_stylebox("panel", "TabContainer", _flat_box(BG_MID, 0, 0))

	# Tab bar styles
	var tab_selected := _flat_box(BG_LIGHT, 4, 2)
	tab_selected.border_color = ACCENT
	tab_selected.border_width_bottom = 2
	var tab_unselected := _flat_box(BG_DARK, 4, 2)
	var tab_hovered := _flat_box(BG_RAISED, 4, 2)
	t.set_stylebox("tab_selected", "TabContainer", tab_selected)
	t.set_stylebox("tab_unselected", "TabContainer", tab_unselected)
	t.set_stylebox("tab_hovered", "TabContainer", tab_hovered)
	t.set_stylebox("tab_selected", "TabBar", tab_selected)
	t.set_stylebox("tab_unselected", "TabBar", tab_unselected)
	t.set_stylebox("tab_hovered", "TabBar", tab_hovered)
	t.set_stylebox("tabbar_background", "TabContainer", _flat_box(BG_DARK, 0, 0))

	# Buttons
	var btn_normal := _flat_box(BG_LIGHT, 5, 4)
	btn_normal.border_color = BORDER_COL
	btn_normal.border_width_bottom = 1
	btn_normal.border_width_top = 1
	btn_normal.border_width_left = 1
	btn_normal.border_width_right = 1
	var btn_hover := _flat_box(BG_RAISED, 5, 4)
	btn_hover.border_color = ACCENT_DIM
	btn_hover.border_width_bottom = 1
	btn_hover.border_width_top = 1
	btn_hover.border_width_left = 1
	btn_hover.border_width_right = 1
	var btn_pressed := _flat_box(Color(0.16, 0.20, 0.28), 5, 4)
	btn_pressed.border_color = ACCENT
	btn_pressed.border_width_bottom = 1
	btn_pressed.border_width_top = 1
	btn_pressed.border_width_left = 1
	btn_pressed.border_width_right = 1
	t.set_stylebox("normal", "Button", btn_normal)
	t.set_stylebox("hover", "Button", btn_hover)
	t.set_stylebox("pressed", "Button", btn_pressed)
	t.set_stylebox("focus", "Button", StyleBoxEmpty.new())

	# LineEdit
	var le_normal := _flat_box(BG_DARK, 4, 4)
	le_normal.border_color = BORDER_COL
	le_normal.border_width_bottom = 1
	le_normal.border_width_top = 1
	le_normal.border_width_left = 1
	le_normal.border_width_right = 1
	var le_focus := _flat_box(BG_DARK, 4, 4)
	le_focus.border_color = ACCENT_DIM
	le_focus.border_width_bottom = 1
	le_focus.border_width_top = 1
	le_focus.border_width_left = 1
	le_focus.border_width_right = 1
	t.set_stylebox("normal", "LineEdit", le_normal)
	t.set_stylebox("focus", "LineEdit", le_focus)

	# OptionButton
	t.set_stylebox("normal", "OptionButton", btn_normal)
	t.set_stylebox("hover", "OptionButton", btn_hover)
	t.set_stylebox("pressed", "OptionButton", btn_pressed)
	t.set_stylebox("focus", "OptionButton", StyleBoxEmpty.new())

	# PopupMenu
	var popup_bg := _flat_box(Color(0.063, 0.071, 0.102), 4, 4)
	popup_bg.border_color = BORDER_COL
	popup_bg.border_width_bottom = 1
	popup_bg.border_width_top = 1
	popup_bg.border_width_left = 1
	popup_bg.border_width_right = 1
	t.set_stylebox("panel", "PopupMenu", popup_bg)
	var popup_hover := _flat_box(BG_RAISED, 0, 0)
	t.set_stylebox("hover", "PopupMenu", popup_hover)

	# ScrollContainer / ScrollBar
	var scroll_bg := _flat_box(BG_DARK, 0, 2)
	t.set_stylebox("scroll", "VScrollBar", scroll_bg)
	t.set_stylebox("scroll", "HScrollBar", scroll_bg)
	var grabber := _flat_box(Color(0.19, 0.22, 0.30), 3, 0)
	t.set_stylebox("grabber", "VScrollBar", grabber)
	t.set_stylebox("grabber", "HScrollBar", grabber)
	var grabber_h := _flat_box(Color(0.25, 0.29, 0.38), 3, 0)
	t.set_stylebox("grabber_highlight", "VScrollBar", grabber_h)
	t.set_stylebox("grabber_highlight", "HScrollBar", grabber_h)

	# Separator
	t.set_color("font_color", "HSeparator", BORDER_COL)
	var sep := StyleBoxLine.new()
	sep.color = BORDER_COL
	sep.thickness = 1
	t.set_stylebox("separator", "HSeparator", sep)

	# GraphEdit
	t.set_color("grid_major", "GraphEdit", Color(0.12, 0.14, 0.19, 0.6))
	t.set_color("grid_minor", "GraphEdit", Color(0.09, 0.11, 0.15, 0.4))
	var ge_panel := _flat_box(BG_DARK, 0, 0)
	t.set_stylebox("panel", "GraphEdit", ge_panel)

	# GraphNode
	var gn_frame := _flat_box(Color(0.094, 0.110, 0.153, 0.96), 8, 6)
	gn_frame.border_color = Color(0.20, 0.24, 0.33, 0.8)
	gn_frame.border_width_bottom = 1
	gn_frame.border_width_top = 1
	gn_frame.border_width_left = 1
	gn_frame.border_width_right = 1
	var gn_selected := _flat_box(Color(0.110, 0.133, 0.184, 0.98), 8, 6)
	gn_selected.border_color = ACCENT_DIM
	gn_selected.border_width_bottom = 2
	gn_selected.border_width_top = 2
	gn_selected.border_width_left = 2
	gn_selected.border_width_right = 2
	var gn_titlebar := _flat_box(Color(0.125, 0.149, 0.204), 8, 4)
	gn_titlebar.corner_detail = 8
	gn_titlebar.set_corner_radius(CORNER_BOTTOM_LEFT, 0)
	gn_titlebar.set_corner_radius(CORNER_BOTTOM_RIGHT, 0)
	var gn_titlebar_sel := _flat_box(Color(0.157, 0.184, 0.247), 8, 4)
	gn_titlebar_sel.corner_detail = 8
	gn_titlebar_sel.set_corner_radius(CORNER_BOTTOM_LEFT, 0)
	gn_titlebar_sel.set_corner_radius(CORNER_BOTTOM_RIGHT, 0)
	t.set_stylebox("panel", "GraphNode", gn_frame)
	t.set_stylebox("panel_selected", "GraphNode", gn_selected)
	t.set_stylebox("titlebar", "GraphNode", gn_titlebar)
	t.set_stylebox("titlebar_selected", "GraphNode", gn_titlebar_sel)
	t.set_color("title_color", "GraphNode", TXT_BRIGHT)

	# SpinBox inherits LineEdit — but we also style the sub-buttons
	# RichTextLabel
	t.set_color("default_color", "RichTextLabel", TXT)

	# SplitContainer grabber area
	t.set_constant("separation", "HSplitContainer", 4)
	t.set_constant("separation", "VSplitContainer", 4)
	t.set_constant("minimum_grab_thickness", "HSplitContainer", 6)
	t.set_constant("minimum_grab_thickness", "VSplitContainer", 6)

	# MenuBar
	var mb_normal := _flat_box(BG_HEADER, 0, 0)
	t.set_stylebox("normal", "MenuBar", mb_normal)
	t.set_stylebox("hover", "MenuBar", _flat_box(BG_RAISED, 0, 2))
	t.set_stylebox("pressed", "MenuBar", _flat_box(BG_LIGHT, 0, 2))

	return t


func _flat_box(color: Color, corner_r: int = 0, content_margin: int = 4) -> StyleBoxFlat:
	var sb := StyleBoxFlat.new()
	sb.bg_color = color
	sb.set_corner_radius_all(corner_r)
	sb.content_margin_left = content_margin
	sb.content_margin_right = content_margin
	sb.content_margin_top = content_margin
	sb.content_margin_bottom = content_margin
	return sb


func _graph_node_titlebar(color: Color) -> StyleBoxFlat:
	var sb := _flat_box(color, 8, 4)
	sb.corner_detail = 8
	sb.set_corner_radius(CORNER_BOTTOM_LEFT, 0)
	sb.set_corner_radius(CORNER_BOTTOM_RIGHT, 0)
	return sb


# ── UI Construction ──────────────────────────────────────────────────

func _build_ui() -> void:
	# Apply dark theme to the whole scene
	theme = _create_dark_theme()
	# Root vertical split: menu bar on top, main content below
	var root := VBoxContainer.new()
	root.set_anchors_and_offsets_preset(PRESET_FULL_RECT)
	root.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	root.size_flags_vertical = Control.SIZE_EXPAND_FILL
	root.grow_horizontal = Control.GROW_DIRECTION_BOTH
	root.grow_vertical = Control.GROW_DIRECTION_BOTH
	add_child(root)

	# ── Menu bar ─────────────────────────────────────────────────────
	var menu_bg := PanelContainer.new()
	var menu_style := _flat_box(BG_HEADER, 0, 0)
	menu_bg.add_theme_stylebox_override("panel", menu_style)
	root.add_child(menu_bg)

	var menu_row := HBoxContainer.new()
	menu_row.custom_minimum_size.y = 32
	menu_bg.add_child(menu_row)

	var menu_bar := MenuBar.new()
	menu_row.add_child(menu_bar)

	# File menu
	var file_menu := PopupMenu.new()
	file_menu.name = "File"
	file_menu.add_item("New", 0)
	file_menu.add_separator()
	file_menu.add_item("Open Graph...", 1)
	file_menu.add_item("Save Graph", 2)
	file_menu.add_item("Save Graph As...", 3)
	file_menu.add_separator()
	# Templates submenu
	var templates_menu := PopupMenu.new()
	templates_menu.name = "Templates"
	file_menu.add_child(templates_menu)
	file_menu.add_submenu_node_item("Templates", templates_menu)
	templates_menu.id_pressed.connect(_on_template_selected)
	file_menu.add_separator()
	file_menu.add_item("Export .py", 4)
	file_menu.id_pressed.connect(_on_file_menu)
	menu_bar.add_child(file_menu)

	# Action buttons
	var btn_spacer := Control.new()
	btn_spacer.custom_minimum_size.x = 16
	menu_row.add_child(btn_spacer)

	var run_btn := Button.new()
	run_btn.text = "  Run Graph  "
	run_btn.pressed.connect(_on_run_pressed)
	run_btn.add_theme_stylebox_override("normal", _flat_box(Color(0.18, 0.61, 0.38), 5, 6))
	run_btn.add_theme_stylebox_override("hover", _flat_box(Color(0.22, 0.72, 0.45), 5, 6))
	run_btn.add_theme_stylebox_override("pressed", _flat_box(Color(0.14, 0.50, 0.30), 5, 6))
	run_btn.add_theme_color_override("font_color", TXT_BRIGHT)
	menu_row.add_child(run_btn)

	var clear_btn := Button.new()
	clear_btn.text = "  Clear All  "
	clear_btn.pressed.connect(_on_clear_pressed)
	clear_btn.add_theme_stylebox_override("normal", _flat_box(Color(0.59, 0.24, 0.27), 5, 6))
	clear_btn.add_theme_stylebox_override("hover", _flat_box(Color(0.70, 0.30, 0.33), 5, 6))
	clear_btn.add_theme_stylebox_override("pressed", _flat_box(Color(0.48, 0.18, 0.22), 5, 6))
	clear_btn.add_theme_color_override("font_color", TXT_BRIGHT)
	menu_row.add_child(clear_btn)

	var spacer := Control.new()
	spacer.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	menu_row.add_child(spacer)

	status_label = Label.new()
	status_label.text = "Starting..."
	menu_row.add_child(status_label)

	# Store references for later
	_templates_menu = templates_menu

	# ── Main content: palette | (graph + terminal) | inspector ───────
	var hsplit := HSplitContainer.new()
	hsplit.size_flags_vertical = Control.SIZE_EXPAND_FILL
	root.add_child(hsplit)

	# Left: Palette with dark background
	var palette_bg := PanelContainer.new()
	palette_bg.custom_minimum_size.x = 210
	palette_bg.add_theme_stylebox_override("panel", _flat_box(BG_MID, 0, 2))
	hsplit.add_child(palette_bg)

	var palette_panel := VBoxContainer.new()
	palette_bg.add_child(palette_panel)

	# Palette header
	var pal_header := Label.new()
	pal_header.text = "  Nodes"
	pal_header.add_theme_color_override("font_color", TXT_BRIGHT)
	pal_header.add_theme_font_size_override("font_size", 14)
	var pal_hdr_bg := PanelContainer.new()
	pal_hdr_bg.add_theme_stylebox_override("panel", _flat_box(BG_HEADER, 0, 4))
	pal_hdr_bg.add_child(pal_header)
	palette_panel.add_child(pal_hdr_bg)

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

	# Bottom panel — Output, Code, Training tabs
	_term_tabs = TabContainer.new()
	_term_tabs.custom_minimum_size.y = 200
	center_vsplit.add_child(_term_tabs)

	# ── Output tab ───────────────────────────────────────────────
	terminal = RichTextLabel.new()
	terminal.name = "Output"
	terminal.scroll_following = true
	_term_tabs.add_child(terminal)

	# ── Code tab ─────────────────────────────────────────────────
	var code_panel := VBoxContainer.new()
	code_panel.name = "Code"
	_term_tabs.add_child(code_panel)

	var code_btns := HBoxContainer.new()
	code_panel.add_child(code_btns)
	var export_btn := Button.new()
	export_btn.text = "Export .py"
	export_btn.pressed.connect(_on_export_pressed)
	code_btns.add_child(export_btn)
	var copy_btn := Button.new()
	copy_btn.text = "Copy"
	copy_btn.pressed.connect(func(): DisplayServer.clipboard_set(code_text.get_parsed_text()))
	code_btns.add_child(copy_btn)

	var code_scroll := ScrollContainer.new()
	code_scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	code_panel.add_child(code_scroll)
	code_text = RichTextLabel.new()
	code_text.text = "# Run the graph to generate code preview."
	code_text.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	code_text.size_flags_vertical = Control.SIZE_EXPAND_FILL
	code_text.selection_enabled = true
	code_scroll.add_child(code_text)

	# ── Plugin panel tabs — built on demand when server reports them ──
	_panel_builders = {
		"Training": _build_training_tab,
		"Robotics": _build_robotics_tab,
	}

	# Right: Inspector with dark background
	var inspector_bg := PanelContainer.new()
	inspector_bg.custom_minimum_size.x = 280
	inspector_bg.add_theme_stylebox_override("panel", _flat_box(BG_MID, 0, 2))
	center_right.add_child(inspector_bg)

	var inspector_outer := VBoxContainer.new()
	inspector_bg.add_child(inspector_outer)

	# Inspector header
	var insp_header := Label.new()
	insp_header.text = "  Inspector"
	insp_header.add_theme_color_override("font_color", TXT_BRIGHT)
	insp_header.add_theme_font_size_override("font_size", 14)
	var insp_hdr_bg := PanelContainer.new()
	insp_hdr_bg.add_theme_stylebox_override("panel", _flat_box(BG_HEADER, 0, 4))
	insp_hdr_bg.add_child(insp_header)
	inspector_outer.add_child(insp_hdr_bg)

	var inspector_scroll := ScrollContainer.new()
	inspector_scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
	inspector_outer.add_child(inspector_scroll)

	inspector_box = VBoxContainer.new()
	inspector_box.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	inspector_scroll.add_child(inspector_box)

	var hint := Label.new()
	hint.text = "Select a node to inspect."
	hint.add_theme_color_override("font_color", TXT_DIM)
	inspector_box.add_child(hint)

	# Set initial split positions — palette ~220px, inspector ~300px from right
	hsplit.split_offset = 220
	center_right.split_offset = -300
	center_vsplit.split_offset = -220

	_log("NodeTool Godot frontend ready.")


# ── Plugin Panel Builders ────────────────────────────────────────────

func _fetch_plugin_panels() -> void:
	_rpc("get_plugin_panels", {}, func(result: Dictionary):
		var panels: Array = result.get("panels", [])
		for panel_name in panels:
			if _panel_builders.has(panel_name):
				_panel_builders[panel_name].call()
				_log("Panel loaded: %s" % panel_name)
			else:
				_log("Panel '%s' has no Godot builder — skipped" % panel_name)
	)


func _build_training_tab() -> void:
	var train_panel := HBoxContainer.new()
	train_panel.name = "Training"
	_term_tabs.add_child(train_panel)

	# Left column: status + loss info
	var train_left := VBoxContainer.new()
	train_left.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	train_left.size_flags_stretch_ratio = 0.4
	train_panel.add_child(train_left)

	var status_row := HBoxContainer.new()
	train_left.add_child(status_row)
	var dot := Label.new()
	dot.text = "*"
	dot.add_theme_color_override("font_color", TXT_DIM)
	status_row.add_child(dot)
	train_status_label = Label.new()
	train_status_label.text = "Idle"
	train_status_label.add_theme_color_override("font_color", TXT_DIM)
	status_row.add_child(train_status_label)

	train_epoch_label = Label.new()
	train_epoch_label.text = "Epoch 0 / 0"
	train_left.add_child(train_epoch_label)
	train_loss_label = Label.new()
	train_loss_label.text = "Best loss  —"
	train_left.add_child(train_loss_label)

	var plot_placeholder := Label.new()
	plot_placeholder.text = "[Loss plot — coming soon]"
	plot_placeholder.add_theme_color_override("font_color", TXT_DIM)
	plot_placeholder.size_flags_vertical = Control.SIZE_EXPAND_FILL
	train_left.add_child(plot_placeholder)

	# Center column: dataset config
	var train_center := VBoxContainer.new()
	train_center.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	train_center.size_flags_stretch_ratio = 0.3
	train_panel.add_child(train_center)

	var ds_header := Label.new()
	ds_header.text = "Datasets"
	ds_header.add_theme_color_override("font_color", ACCENT)
	train_center.add_child(ds_header)
	train_center.add_child(HSeparator.new())
	train_dataset_box = VBoxContainer.new()
	train_center.add_child(train_dataset_box)
	var ds_hint := Label.new()
	ds_hint.text = "Load a template to configure datasets."
	ds_hint.add_theme_color_override("font_color", TXT_DIM)
	ds_hint.autowrap_mode = TextServer.AUTOWRAP_WORD
	train_dataset_box.add_child(ds_hint)

	# Right column: hyperparams + controls
	var train_right := VBoxContainer.new()
	train_right.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	train_right.size_flags_stretch_ratio = 0.3
	train_panel.add_child(train_right)

	var ep_row := HBoxContainer.new()
	train_right.add_child(ep_row)
	var ep_lbl := Label.new()
	ep_lbl.text = "epochs"
	ep_lbl.custom_minimum_size.x = 60
	ep_row.add_child(ep_lbl)
	train_epochs_spin = SpinBox.new()
	train_epochs_spin.min_value = 1
	train_epochs_spin.max_value = 10000
	train_epochs_spin.value = 10
	train_epochs_spin.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	ep_row.add_child(train_epochs_spin)

	var lr_row := HBoxContainer.new()
	train_right.add_child(lr_row)
	var lr_lbl := Label.new()
	lr_lbl.text = "lr"
	lr_lbl.custom_minimum_size.x = 60
	lr_row.add_child(lr_lbl)
	train_lr_spin = SpinBox.new()
	train_lr_spin.min_value = 0.0000001
	train_lr_spin.max_value = 1.0
	train_lr_spin.step = 0.0001
	train_lr_spin.value = 0.001
	train_lr_spin.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	lr_row.add_child(train_lr_spin)

	var opt_row := HBoxContainer.new()
	train_right.add_child(opt_row)
	var opt_lbl := Label.new()
	opt_lbl.text = "optim"
	opt_lbl.custom_minimum_size.x = 60
	opt_row.add_child(opt_lbl)
	train_optim_btn = OptionButton.new()
	for o in ["adam", "adamw", "sgd", "rmsprop"]:
		train_optim_btn.add_item(o)
	train_optim_btn.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	opt_row.add_child(train_optim_btn)

	var loss_row := HBoxContainer.new()
	train_right.add_child(loss_row)
	var loss_lbl := Label.new()
	loss_lbl.text = "loss"
	loss_lbl.custom_minimum_size.x = 60
	loss_row.add_child(loss_lbl)
	train_loss_btn = OptionButton.new()
	for l in ["crossentropy", "mse", "bce", "bcewithlogits", "l1"]:
		train_loss_btn.add_item(l)
	train_loss_btn.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	loss_row.add_child(train_loss_btn)

	var dev_row := HBoxContainer.new()
	train_right.add_child(dev_row)
	var dev_lbl := Label.new()
	dev_lbl.text = "device"
	dev_lbl.custom_minimum_size.x = 60
	dev_row.add_child(dev_lbl)
	train_device_btn = OptionButton.new()
	for d in ["cpu", "cuda", "cuda:0", "cuda:1"]:
		train_device_btn.add_item(d)
	train_device_btn.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	dev_row.add_child(train_device_btn)

	train_right.add_child(HSeparator.new())
	var ctrl_row := HBoxContainer.new()
	train_right.add_child(ctrl_row)
	var start_btn := Button.new()
	start_btn.text = "Start"
	start_btn.pressed.connect(_on_train_start)
	ctrl_row.add_child(start_btn)
	var pause_btn := Button.new()
	pause_btn.text = "Pause"
	ctrl_row.add_child(pause_btn)
	var stop_btn := Button.new()
	stop_btn.text = "Stop"
	ctrl_row.add_child(stop_btn)


func _build_robotics_tab() -> void:
	var rob_panel := HBoxContainer.new()
	rob_panel.name = "Robotics"
	_term_tabs.add_child(rob_panel)

	# Left: serial monitor
	var rob_left := VBoxContainer.new()
	rob_left.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	rob_left.size_flags_stretch_ratio = 0.5
	rob_panel.add_child(rob_left)

	var rob_serial_hdr := Label.new()
	rob_serial_hdr.text = "Serial Monitor"
	rob_serial_hdr.add_theme_color_override("font_color", ACCENT)
	rob_left.add_child(rob_serial_hdr)

	var rob_port_row := HBoxContainer.new()
	rob_left.add_child(rob_port_row)
	var rob_port_lbl := Label.new()
	rob_port_lbl.text = "Port"
	rob_port_lbl.custom_minimum_size.x = 40
	rob_port_row.add_child(rob_port_lbl)
	var rob_port_opt := OptionButton.new()
	rob_port_opt.add_item("(none)")
	rob_port_opt.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	rob_port_row.add_child(rob_port_opt)
	var rob_baud_opt := OptionButton.new()
	for b in ["9600", "19200", "38400", "57600", "115200"]:
		rob_baud_opt.add_item(b)
	rob_baud_opt.select(4)
	rob_port_row.add_child(rob_baud_opt)

	var rob_btn_row := HBoxContainer.new()
	rob_left.add_child(rob_btn_row)
	var rob_conn_btn := Button.new()
	rob_conn_btn.text = "Connect"
	rob_conn_btn.pressed.connect(func(): _log("[Robotics] Serial connect not yet wired"))
	rob_btn_row.add_child(rob_conn_btn)
	var rob_disc_btn := Button.new()
	rob_disc_btn.text = "Disconnect"
	rob_disc_btn.pressed.connect(func(): _log("[Robotics] Disconnected"))
	rob_btn_row.add_child(rob_disc_btn)
	var rob_refresh_btn := Button.new()
	rob_refresh_btn.text = "Refresh"
	rob_btn_row.add_child(rob_refresh_btn)

	rob_left.add_child(HSeparator.new())
	var rob_serial_log := RichTextLabel.new()
	rob_serial_log.text = "(serial output will appear here)"
	rob_serial_log.add_theme_color_override("default_color", TXT_DIM)
	rob_serial_log.size_flags_vertical = Control.SIZE_EXPAND_FILL
	rob_serial_log.scroll_following = true
	rob_left.add_child(rob_serial_log)

	var rob_send_row := HBoxContainer.new()
	rob_left.add_child(rob_send_row)
	var rob_send_input := LineEdit.new()
	rob_send_input.placeholder_text = "Send command..."
	rob_send_input.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	rob_send_row.add_child(rob_send_input)
	var rob_send_btn := Button.new()
	rob_send_btn.text = "Send"
	rob_send_row.add_child(rob_send_btn)

	# Right: sensor plot placeholder
	var rob_right := VBoxContainer.new()
	rob_right.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	rob_right.size_flags_stretch_ratio = 0.5
	rob_panel.add_child(rob_right)

	var rob_plot_hdr := Label.new()
	rob_plot_hdr.text = "Live Sensor Plot"
	rob_plot_hdr.add_theme_color_override("font_color", ACCENT)
	rob_right.add_child(rob_plot_hdr)

	var rob_plot_hint := Label.new()
	rob_plot_hint.text = "Wire sensor nodes to see live data here."
	rob_plot_hint.add_theme_color_override("font_color", TXT_DIM)
	rob_right.add_child(rob_plot_hint)

	var rob_plot_area := Panel.new()
	rob_plot_area.size_flags_vertical = Control.SIZE_EXPAND_FILL
	rob_plot_area.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	rob_plot_area.add_theme_stylebox_override("panel", _flat_box(BG_DARK, 4, 8))
	rob_right.add_child(rob_plot_area)


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
	var parsed = JSON.parse_string(raw)
	if parsed == null:
		_log("[ERROR] Invalid JSON from server")
		return
	var msg: Dictionary = parsed
	var id = msg.get("id")
	if id is float:
		id = int(id)
	if msg.has("error"):
		var err = msg["error"]
		_log("[Error] %s" % str(err.get("message", err)))
		if id != null:
			_pending.erase(id)
		return
	if id != null and _pending.has(id):
		var cb: Callable = _pending[id]
		_pending.erase(id)
		cb.call(msg.get("result", {}))


func _log(text: String) -> void:
	if terminal:
		terminal.append_text(text + "\n")
	print(text)


# ── Registry / Palette ───────────────────────────────────────────────

func _on_registry_received(result: Dictionary) -> void:
	registry = result.get("categories", {})
	category_order = result.get("category_order", [])
	var total := 0
	for cat in registry:
		total += registry[cat].size()
	_log("Registry: %d categories, %d nodes" % [registry.size(), total])
	_build_palette()
	_fetch_templates()
	_fetch_plugin_panels()


func _build_palette() -> void:
	for child in palette_list.get_children():
		child.queue_free()

	for cat in category_order:
		if not registry.has(cat):
			continue
		var nodes_arr: Array = registry[cat]
		if nodes_arr.is_empty():
			continue

		# Category header with colored accent
		var cat_bg := PanelContainer.new()
		var cat_col: Color = CATEGORY_COLORS.get(cat, TXT_DIM)
		var cat_style := _flat_box(cat_col.darkened(0.7), 3, 3)
		cat_style.border_color = cat_col.darkened(0.3)
		cat_style.border_width_left = 3
		cat_bg.add_theme_stylebox_override("panel", cat_style)
		var cat_label := Label.new()
		cat_label.text = cat
		cat_label.add_theme_color_override("font_color", cat_col.lightened(0.3))
		cat_label.add_theme_font_size_override("font_size", 13)
		cat_bg.add_child(cat_label)
		palette_list.add_child(cat_bg)

		for node_def in nodes_arr:
			var btn := Button.new()
			btn.text = "  %s" % node_def["label"]
			btn.alignment = HORIZONTAL_ALIGNMENT_LEFT
			btn.set_meta("type_name", node_def["type_name"])
			btn.set_meta("node_def", node_def)
			btn.pressed.connect(_on_palette_btn.bind(node_def["type_name"]))
			# Flat palette buttons
			btn.add_theme_stylebox_override("normal", _flat_box(Color.TRANSPARENT, 3, 3))
			btn.add_theme_stylebox_override("hover", _flat_box(BG_RAISED, 3, 3))
			btn.add_theme_stylebox_override("pressed", _flat_box(BG_LIGHT, 3, 3))
			btn.add_theme_color_override("font_color", TXT_DIM)
			btn.add_theme_color_override("font_hover_color", TXT)
			btn.add_theme_font_size_override("font_size", 13)
			palette_list.add_child(btn)

	# Add a small spacer at the end
	var end_spacer := Control.new()
	end_spacer.custom_minimum_size.y = 20
	palette_list.add_child(end_spacer)
	_log("Palette built")


func _on_palette_btn(type_name: String) -> void:
	_rpc("add_node", {"type_name": type_name}, func(result: Dictionary):
		_add_graph_node(result)
		_log("Added: %s" % result.get("label", type_name))
		# Refresh dataset panel if a marker was added
		if type_name.begins_with("pt_input_marker") or type_name.begins_with("pt_train_marker"):
			_refresh_dataset_panel()
	)


func _on_search_changed(text: String) -> void:
	var query := text.strip_edges().to_lower()
	for child in palette_list.get_children():
		if child is Button:
			child.visible = query.is_empty() or query in child.text.to_lower()


# ── Graph Node Management ────────────────────────────────────────────

func _add_graph_node(node_data: Dictionary, pos_arr: Array = []) -> void:
	var node_id: String = node_data["id"]
	var gn := GraphNode.new()
	gn.title = node_data.get("label", "Node")
	gn.name = "gn_%s" % node_id.substr(0, 8)
	gn.set_meta("node_id", node_id)
	gn.set_meta("node_data", node_data)

	# Position: use provided or stagger
	if pos_arr.size() >= 2:
		gn.position_offset = Vector2(float(pos_arr[0]), float(pos_arr[1]))
	else:
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
		summary.name = "cfg_summary"
		summary.text = ", ".join(config_parts.slice(0, 5))
		summary.add_theme_color_override("font_color", TXT_DIM)
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

	# Apply category color to title bar
	var cat_name: String = node_data.get("category", "")
	if CATEGORY_COLORS.has(cat_name):
		var cat_col: Color = CATEGORY_COLORS[cat_name]
		gn.add_theme_stylebox_override("titlebar", _graph_node_titlebar(cat_col))
		var sel_col := cat_col.lightened(0.15)
		gn.add_theme_stylebox_override("titlebar_selected", _graph_node_titlebar(sel_col))

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
	title.add_theme_color_override("font_color", TXT_BRIGHT)
	inspector_box.add_child(title)

	var cat_name: String = node_data.get("category", "")
	var cat_lbl := Label.new()
	cat_lbl.text = cat_name
	var cat_col: Color = CATEGORY_COLORS.get(cat_name, TXT_DIM)
	cat_lbl.add_theme_color_override("font_color", cat_col)
	inspector_box.add_child(cat_lbl)

	var desc := Label.new()
	desc.text = node_data.get("description", "")
	desc.autowrap_mode = TextServer.AUTOWRAP_WORD
	desc.add_theme_color_override("font_color", TXT_DIM)
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
			hdr.add_theme_color_override("font_color", ACCENT)
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
		out_hdr.add_theme_color_override("font_color", TXT_DIM)
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
	# Update local node_data so the inspector shows the new value on revisit
	var gn = _nodes.get(node_id)
	if gn and is_instance_valid(gn):
		var data: Dictionary = gn.get_meta("node_data", {})
		if data.has("inputs") and data["inputs"].has(port_name):
			data["inputs"][port_name]["default_value"] = value
			gn.set_meta("node_data", data)
		# Also update the config summary on the canvas node
		var summary_parts: PackedStringArray = []
		for pname in data.get("inputs", {}):
			var port: Dictionary = data["inputs"][pname]
			if port.get("editable", false):
				var val = port.get("default_value")
				if val != null and str(val) != "" and str(val) != "false" and str(val) != "False" and str(val) != "0":
					summary_parts.append(str(val))
		var summary_node = gn.find_child("cfg_summary", false, false)
		if summary_node and summary_node is Label:
			summary_node.text = ", ".join(summary_parts.slice(0, 5))


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


# ── File Menu ─────────────────────────────────────────────────────────

func _on_file_menu(id: int) -> void:
	match id:
		0: _file_new()
		1: _file_open()
		2: _file_save()
		3: _file_save_as()
		4: _on_export_pressed()


func _file_new() -> void:
	_on_clear_pressed()
	_save_path = ""
	_log("New graph")


func _file_open() -> void:
	var fd := FileDialog.new()
	fd.file_mode = FileDialog.FILE_MODE_OPEN_FILE
	fd.access = FileDialog.ACCESS_FILESYSTEM
	fd.filters = PackedStringArray(["*.json ; Graph files"])
	fd.file_selected.connect(func(path: String):
		_rpc("load_graph", {"path": path}, func(result: Dictionary):
			# Clear canvas
			node_graph.clear_connections()
			for gn in _nodes.values():
				if is_instance_valid(gn):
					gn.queue_free()
			_nodes.clear()
			_name_to_id.clear()
			_spawn_count = 0
			# Rebuild from loaded data
			var positions: Dictionary = result.get("positions", {})
			var nodes_data: Dictionary = result.get("nodes", {})
			for nid in nodes_data:
				var nd: Dictionary = nodes_data[nid]
				_add_graph_node(nd, positions.get(nid, []))
			# Rebuild connections
			for conn in result.get("connections", []):
				var from_gn_name := _id_to_gn_name(conn["from_node"])
				var to_gn_name := _id_to_gn_name(conn["to_node"])
				var from_port := _output_name_to_idx(conn["from_node"], conn["from_port"])
				var to_port := _input_name_to_idx(conn["to_node"], conn["to_port"])
				if from_gn_name != "" and to_gn_name != "" and from_port >= 0 and to_port >= 0:
					node_graph.connect_node(from_gn_name, from_port, to_gn_name, to_port)
			_save_path = path
			_log("Loaded: %s (%d nodes)" % [path, nodes_data.size()])
			_refresh_dataset_panel()
		)
		fd.queue_free()
	)
	fd.canceled.connect(func(): fd.queue_free())
	add_child(fd)
	fd.popup_centered(Vector2i(700, 500))


func _file_save() -> void:
	if _save_path.is_empty():
		_file_save_as()
		return
	_do_save(_save_path)


func _file_save_as() -> void:
	var fd := FileDialog.new()
	fd.file_mode = FileDialog.FILE_MODE_SAVE_FILE
	fd.access = FileDialog.ACCESS_FILESYSTEM
	fd.filters = PackedStringArray(["*.json ; Graph files"])
	fd.file_selected.connect(func(path: String):
		_save_path = path
		_do_save(path)
		fd.queue_free()
	)
	fd.canceled.connect(func(): fd.queue_free())
	add_child(fd)
	fd.popup_centered(Vector2i(700, 500))


func _do_save(path: String) -> void:
	# Collect positions from canvas
	var positions := {}
	for nid in _nodes:
		var gn: GraphNode = _nodes[nid]
		if is_instance_valid(gn):
			positions[nid] = [gn.position_offset.x, gn.position_offset.y]
	_rpc("save_graph", {"path": path, "positions": positions}, func(result: Dictionary):
		_log("Saved: %s (%d nodes)" % [path, result.get("nodes", 0)])
	)


# ── Templates ────────────────────────────────────────────────────────

func _fetch_templates() -> void:
	_rpc("get_templates", {}, func(result: Dictionary):
		_template_labels.clear()
		_templates_menu.clear()
		var templates: Array = result.get("templates", [])
		for i in range(templates.size()):
			var t: Dictionary = templates[i]
			_templates_menu.add_item(t.get("label", "?"), i)
			_template_labels.append(t.get("label", "?"))
		_log("Templates: %d available" % templates.size())
	)


func _on_template_selected(id: int) -> void:
	if id < 0 or id >= _template_labels.size():
		return
	var label: String = _template_labels[id]
	_log("Loading template: %s" % label)
	_rpc("load_template", {"label": label}, func(result: Dictionary):
		# Clear canvas
		node_graph.clear_connections()
		for gn in _nodes.values():
			if is_instance_valid(gn):
				gn.queue_free()
		_nodes.clear()
		_name_to_id.clear()
		_spawn_count = 0
		# Rebuild
		var positions: Dictionary = result.get("positions", {})
		var nodes_data: Dictionary = result.get("nodes", {})
		for nid in nodes_data:
			var nd: Dictionary = nodes_data[nid]
			_add_graph_node(nd, positions.get(nid, []))
		for conn in result.get("connections", []):
			var from_gn := _id_to_gn_name(conn["from_node"])
			var to_gn := _id_to_gn_name(conn["to_node"])
			var fp := _output_name_to_idx(conn["from_node"], conn["from_port"])
			var tp := _input_name_to_idx(conn["to_node"], conn["to_port"])
			if from_gn != "" and to_gn != "" and fp >= 0 and tp >= 0:
				node_graph.connect_node(from_gn, fp, to_gn, tp)
		_log("Template loaded: %s (%d nodes)" % [label, nodes_data.size()])
		_refresh_dataset_panel()
	)


# ── Dataset Panel (dynamic from markers) ─────────────────────────────

func _refresh_dataset_panel() -> void:
	_rpc("get_marker_groups", {}, func(result: Dictionary):
		var groups: Dictionary = result.get("groups", {})
		# Clear dataset box
		for child in train_dataset_box.get_children():
			child.queue_free()
		if groups.is_empty():
			var hint := Label.new()
			hint.text = "Add Data In (A) marker nodes to configure datasets,\nor load a template from File > Templates."
			hint.add_theme_color_override("font_color", TXT_DIM)
			hint.autowrap_mode = TextServer.AUTOWRAP_WORD
			train_dataset_box.add_child(hint)
			return
		for group_name in groups:
			var group: Dictionary = groups[group_name]
			var mods: Array = group.get("modalities", [])
			var mods_str: String = ", ".join(mods) if mods.size() > 0 else "?"
			# Group header
			var hdr := Label.new()
			hdr.text = "[%s] %s" % [group_name, mods_str]
			hdr.add_theme_color_override("font_color", ACCENT)
			train_dataset_box.add_child(hdr)
			# Path
			var path_row := HBoxContainer.new()
			train_dataset_box.add_child(path_row)
			var path_lbl := Label.new()
			path_lbl.text = "path"
			path_lbl.custom_minimum_size.x = 40
			path_row.add_child(path_lbl)
			var path_edit := LineEdit.new()
			path_edit.placeholder_text = "mnist, cifar10, /path/to/data..."
			path_edit.size_flags_horizontal = Control.SIZE_EXPAND_FILL
			path_edit.name = "ds_%s_path" % group_name
			path_row.add_child(path_edit)
			# Batch + split
			var bs_row := HBoxContainer.new()
			train_dataset_box.add_child(bs_row)
			var batch_lbl := Label.new()
			batch_lbl.text = "batch"
			batch_lbl.custom_minimum_size.x = 40
			bs_row.add_child(batch_lbl)
			var batch_spin := SpinBox.new()
			batch_spin.min_value = 1
			batch_spin.max_value = 4096
			batch_spin.value = 32
			batch_spin.name = "ds_%s_batch" % group_name
			bs_row.add_child(batch_spin)
			var split_lbl := Label.new()
			split_lbl.text = "split"
			bs_row.add_child(split_lbl)
			var split_opt := OptionButton.new()
			split_opt.add_item("train")
			split_opt.add_item("test")
			split_opt.add_item("val")
			split_opt.name = "ds_%s_split" % group_name
			bs_row.add_child(split_opt)
			# Separator between groups
			train_dataset_box.add_child(HSeparator.new())
	)


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


func _on_export_pressed() -> void:
	_log("--- Exporting code ---")
	_rpc("export_code", {}, func(result: Dictionary):
		var code: String = result.get("code", "# Export failed")
		code_text.clear()
		code_text.append_text(code)
		_log("Code exported (%d chars)" % code.length())
	)


func _on_train_start() -> void:
	_log("--- Starting training ---")
	train_status_label.text = "Running"
	train_status_label.add_theme_color_override("font_color", OK_GREEN)
	_rpc("train_start", {
		"epochs": int(train_epochs_spin.value),
		"lr": train_lr_spin.value,
		"optimizer": train_optim_btn.get_item_text(train_optim_btn.selected),
		"loss": train_loss_btn.get_item_text(train_loss_btn.selected),
		"device": train_device_btn.get_item_text(train_device_btn.selected),
	}, func(result: Dictionary):
		if result.has("error"):
			_log("[Train Error] %s" % str(result["error"]))
			train_status_label.text = "Error"
			train_status_label.add_theme_color_override("font_color", ERR_RED)
		else:
			_log("[Train] %s" % result.get("message", "Started"))
	)


# ── Helpers ──────────────────────────────────────────────────────────

func _id_to_gn_name(node_id: String) -> String:
	"""Map Python node_id to the GraphNode name on the canvas."""
	var gn = _nodes.get(node_id)
	if gn and is_instance_valid(gn):
		return gn.name
	return ""


func _output_name_to_idx(node_id: String, port_name: String) -> int:
	"""Map output port name to its visual slot index."""
	var gn = _nodes.get(node_id)
	if gn == null:
		return -1
	var data: Dictionary = gn.get_meta("node_data", {})
	var keys: Array = data.get("outputs", {}).keys()
	return keys.find(port_name)


func _input_name_to_idx(node_id: String, port_name: String) -> int:
	"""Map data input port name to its visual slot index."""
	var gn = _nodes.get(node_id)
	if gn == null:
		return -1
	var data: Dictionary = gn.get_meta("node_data", {})
	var inputs: Dictionary = data.get("inputs", {})
	var data_ports: Array = []
	for pname in inputs:
		if not inputs[pname].get("editable", false):
			data_ports.append(pname)
	return data_ports.find(port_name)


func _color_from_array(arr: Array) -> Color:
	return Color(
		arr[0] / 255.0 if arr.size() > 0 else 0.6,
		arr[1] / 255.0 if arr.size() > 1 else 0.6,
		arr[2] / 255.0 if arr.size() > 2 else 0.7,
		arr[3] / 255.0 if arr.size() > 3 else 1.0)
