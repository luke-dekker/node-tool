"""Dark theme for the node tool — cool neutral with cyan accent."""

from __future__ import annotations
import dearpygui.dearpygui as dpg

# -- Palette -----------------------------------------------------------------
# Cool desaturated neutrals (slight blue undertone) + single cyan accent.
BG_DARK    = (12,  14,  20, 255)   # editor canvas / deepest
BG_MID     = (20,  23,  31, 255)   # window bg
BG_LIGHT   = (28,  32,  42, 255)   # frame bg (inputs)
BG_POPUP   = (16,  18,  26, 250)
BG_HEADER  = (24,  28,  38, 255)   # section header bar
BG_RAISED  = (34,  39,  52, 255)   # hovered / raised surfaces

ACCENT     = (88,  196, 245, 255)  # primary cyan
ACCENT2    = (130, 110, 240, 255)  # secondary violet (used sparingly)
ACCENT_DIM = (55,  120, 160, 255)
ACCENT_GLOW= (88,  196, 245,  60)

TEXT       = (224, 228, 238, 255)
TEXT_DIM   = (132, 142, 162, 255)
TEXT_FAINT = ( 92, 100, 118, 255)
TEXT_BRIGHT= (250, 252, 255, 255)

BORDER     = (44,  52,  72,  200)
BORDER_H   = (88,  196, 245, 220)
SCROLLBAR  = (48,  56,  76,  200)

OK_GREEN   = ( 80, 200, 130, 255)
WARN_AMBER = (240, 180,  70, 255)
ERR_RED    = (235,  90,  90, 255)

# Node header colors per category
MATH_COLOR   = (40,  160, 80,  255)
LOGIC_COLOR  = (60,  120, 220, 255)
STRING_COLOR = (220, 140, 40,  255)
DATA_COLOR   = (160, 60,  200, 255)
IO_COLOR     = (200, 60,  80,  255)

# Pin colors per type
FLOAT_PIN      = (80,  200, 120, 255)
INT_PIN        = (80,  140, 220, 255)
BOOL_PIN       = (220, 100, 80,  255)
STRING_PIN     = (220, 180, 80,  255)
ANY_PIN        = (160, 160, 180, 255)
TENSOR_PIN     = (255, 120,  40, 255)   # orange
MODULE_PIN     = (160,  80, 255, 255)   # violet
DATALOADER_PIN = ( 40, 200, 200, 255)   # teal
OPTIMIZER_PIN  = (255, 200,  40, 255)   # gold
LOSS_FN_PIN    = (220,  60, 120, 255)   # rose

PYTORCH_COLOR  = (220,  90,  30)        # PyTorch orange (no alpha)

# New pin colors
DATAFRAME_PIN    = ( 50, 205, 120, 255)   # emerald green
NDARRAY_PIN      = ( 80, 180, 255, 255)   # sky blue
SERIES_PIN       = (150, 230,  80, 255)   # lime
SKLEARN_PIN      = (255, 160,  50, 255)   # amber
IMAGE_PIN        = (255,  80, 180, 255)   # hot pink
SCHEDULER_PIN    = (160, 230,  60, 255)   # lime-green
DATASET_PIN   = ( 80, 220, 180, 255)   # mint green
TRANSFORM_PIN = (200, 130, 255, 255)   # lavender

# New category colors (no alpha - used in create_node_theme)
NUMPY_COLOR      = ( 70, 130, 200)        # numpy blue
PANDAS_COLOR     = (140, 100, 200)        # pandas purple
SKLEARN_COLOR    = (230, 140,  40)        # sklearn orange
SCIPY_COLOR      = ( 60, 180, 130)        # scipy teal
VIZ_COLOR        = (220,  60, 120)        # viz pink

PYTHON_COLOR     = (100, 180,  80)         # unified green

CATEGORY_COLORS = {
    "Python":  (PYTHON_COLOR[0], PYTHON_COLOR[1], PYTHON_COLOR[2], 255),
    "PyTorch": (PYTORCH_COLOR[0], PYTORCH_COLOR[1], PYTORCH_COLOR[2], 255),
    "NumPy":   (NUMPY_COLOR[0],   NUMPY_COLOR[1],   NUMPY_COLOR[2],   255),
    "Pandas":  (PANDAS_COLOR[0],  PANDAS_COLOR[1],  PANDAS_COLOR[2],  255),
    "Sklearn": (SKLEARN_COLOR[0], SKLEARN_COLOR[1], SKLEARN_COLOR[2], 255),
    "SciPy":   (SCIPY_COLOR[0],   SCIPY_COLOR[1],   SCIPY_COLOR[2],   255),
    "Viz":     (VIZ_COLOR[0],     VIZ_COLOR[1],     VIZ_COLOR[2],     255),
}


def create_fonts() -> dict[str, int]:
    """Load Segoe UI (regular + bold) and Consolas as DPG fonts.

    Returns a dict with keys: 'default', 'bold', 'mono', 'small'.
    Caller should bind 'default' globally and use the others where needed.
    """
    import os
    fonts_dir = "C:/Windows/Fonts"
    paths = {
        "regular": os.path.join(fonts_dir, "segoeui.ttf"),
        "bold":    os.path.join(fonts_dir, "segoeuib.ttf"),
        "mono":    os.path.join(fonts_dir, "consola.ttf"),
    }

    out: dict[str, int] = {}
    with dpg.font_registry():
        # Slightly larger sizes — DPG renders TTF crisply at native px.
        if os.path.exists(paths["regular"]):
            with dpg.font(paths["regular"], 16) as f:
                dpg.add_font_range_hint(dpg.mvFontRangeHint_Default)
            out["default"] = f
            with dpg.font(paths["regular"], 13) as f_small:
                dpg.add_font_range_hint(dpg.mvFontRangeHint_Default)
            out["small"] = f_small
        if os.path.exists(paths["bold"]):
            with dpg.font(paths["bold"], 16) as f_bold:
                dpg.add_font_range_hint(dpg.mvFontRangeHint_Default)
            out["bold"] = f_bold
        if os.path.exists(paths["mono"]):
            with dpg.font(paths["mono"], 14) as f_mono:
                dpg.add_font_range_hint(dpg.mvFontRangeHint_Default)
            out["mono"] = f_mono
    return out


def create_global_theme() -> int:
    """Create and return the global theme tag."""
    with dpg.theme() as theme_id:
        with dpg.theme_component(dpg.mvAll):
            # Window / frame backgrounds
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg,       BG_MID)
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg,        BG_DARK)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg,        BG_LIGHT)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, BG_RAISED)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive,  (44, 52, 70, 255))
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg,        BG_POPUP)
            dpg.add_theme_color(dpg.mvThemeCol_Border,         BORDER)
            dpg.add_theme_color(dpg.mvThemeCol_BorderShadow,   (0, 0, 0, 0))

            # Title bars
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg,           BG_HEADER)
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive,     BG_HEADER)
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgCollapsed,  BG_HEADER)

            # Menu bar
            dpg.add_theme_color(dpg.mvThemeCol_MenuBarBg,      BG_HEADER)

            # Text
            dpg.add_theme_color(dpg.mvThemeCol_Text,           TEXT)
            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled,   TEXT_DIM)

            # Buttons (subtle by default — Run/Clear get explicit themes)
            dpg.add_theme_color(dpg.mvThemeCol_Button,         (40, 48, 66, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,  (56, 68, 92, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,   (72, 88, 118, 255))

            # Headers (collapsing headers, selectables)
            dpg.add_theme_color(dpg.mvThemeCol_Header,         (38, 46, 64, 255))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered,  (54, 66, 90, 255))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive,   (68, 84, 114, 255))

            # Separator / border
            dpg.add_theme_color(dpg.mvThemeCol_Separator,         BORDER)
            dpg.add_theme_color(dpg.mvThemeCol_SeparatorHovered,  BORDER_H)
            dpg.add_theme_color(dpg.mvThemeCol_SeparatorActive,   ACCENT)

            # Sliders / checkboxes
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab,       ACCENT)
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, ACCENT2)
            dpg.add_theme_color(dpg.mvThemeCol_CheckMark,        ACCENT)

            # Scrollbar
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg,          (0, 0, 0, 0))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab,        SCROLLBAR)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered, (72, 84, 108, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabActive,  ACCENT_DIM)

            # Tabs
            dpg.add_theme_color(dpg.mvThemeCol_Tab,              BG_DARK)
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered,       BG_RAISED)
            dpg.add_theme_color(dpg.mvThemeCol_TabActive,        (38, 46, 64, 255))

            # Resize grip — invisible, hover only
            dpg.add_theme_color(dpg.mvThemeCol_ResizeGrip,         (0, 0, 0, 0))
            dpg.add_theme_color(dpg.mvThemeCol_ResizeGripHovered,  (88, 196, 245, 120))
            dpg.add_theme_color(dpg.mvThemeCol_ResizeGripActive,   ACCENT)

            # Docking
            dpg.add_theme_color(dpg.mvThemeCol_DockingEmptyBg, BG_DARK)

            # Rounding / spacing — slightly tighter, more consistent
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding,    8.0)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding,     6.0)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,     5.0)
            dpg.add_theme_style(dpg.mvStyleVar_PopupRounding,     6.0)
            dpg.add_theme_style(dpg.mvStyleVar_ScrollbarRounding, 6.0)
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding,      4.0)
            dpg.add_theme_style(dpg.mvStyleVar_TabRounding,       5.0)
            dpg.add_theme_style(dpg.mvStyleVar_WindowBorderSize,  1.0)
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize,   0.0)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding,     12.0, 10.0)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding,      9.0, 6.0)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing,       8.0, 7.0)
            dpg.add_theme_style(dpg.mvStyleVar_ItemInnerSpacing,  6.0, 5.0)
            dpg.add_theme_style(dpg.mvStyleVar_IndentSpacing,     14.0)
            dpg.add_theme_style(dpg.mvStyleVar_ScrollbarSize,     11.0)
            dpg.add_theme_style(dpg.mvStyleVar_GrabMinSize,       10.0)

        # -- Node editor colors (use mvThemeCat_Nodes category) -------------
        with dpg.theme_component(dpg.mvNodeEditor):
            dpg.add_theme_color(dpg.mvNodeCol_GridBackground,      BG_DARK,              category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_GridLine,            (32, 38,  52, 180),   category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_Link,                (130, 165, 200, 220), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_LinkHovered,         (180, 220, 255, 255), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_LinkSelected,        ACCENT,               category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_BoxSelector,         (88, 196, 245,  30),  category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_BoxSelectorOutline,  (88, 196, 245, 160),  category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_style(dpg.mvNodeStyleVar_NodeCornerRounding,     6.0, category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_style(dpg.mvNodeStyleVar_NodePadding,            8.0, 6.0, category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_style(dpg.mvNodeStyleVar_NodeBorderThickness,    1.0, category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_style(dpg.mvNodeStyleVar_LinkThickness,          2.6, category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_style(dpg.mvNodeStyleVar_LinkHoverDistance,      8.0, category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_style(dpg.mvNodeStyleVar_PinCircleRadius,        4.5, category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_style(dpg.mvNodeStyleVar_PinHoverRadius,         9.0, category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_style(dpg.mvNodeStyleVar_GridSpacing,           28.0, category=dpg.mvThemeCat_Nodes)

        with dpg.theme_component(dpg.mvNode):
            dpg.add_theme_color(dpg.mvNodeCol_NodeBackground,         (28, 32, 44, 245), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundHovered,  (34, 40, 54, 250), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundSelected, (38, 46, 62, 255), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_NodeOutline,            (60, 70, 92, 220), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_TitleBar,               (32, 38, 52, 255), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_TitleBarHovered,        (42, 50, 68, 255), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_TitleBarSelected,       (50, 60, 82, 255), category=dpg.mvThemeCat_Nodes)

    return theme_id


def create_node_theme(category: str) -> int:
    """Create a per-category node theme (colored title bar)."""
    color = CATEGORY_COLORS.get(category, DATA_COLOR)
    h_color = tuple(min(255, int(c * 1.2)) if i < 3 else c for i, c in enumerate(color))
    s_color = tuple(min(255, int(c * 1.4)) if i < 3 else c for i, c in enumerate(color))

    with dpg.theme() as theme_id:
        with dpg.theme_component(dpg.mvNode):
            dpg.add_theme_color(dpg.mvNodeCol_TitleBar,          color,   category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_TitleBarHovered,   h_color, category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_TitleBarSelected,  s_color, category=dpg.mvThemeCat_Nodes)
    return theme_id


def create_button_theme(color: tuple) -> int:
    """Create a custom-colored button theme."""
    h = tuple(min(255, int(c * 1.3)) if i < 3 else c for i, c in enumerate(color))
    a = tuple(min(255, int(c * 1.6)) if i < 3 else c for i, c in enumerate(color))
    with dpg.theme() as t:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button,        color)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, h)
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,  a)
    return t


def create_run_button_theme() -> int:
    return create_button_theme((46, 156, 96, 255))


def create_clear_button_theme() -> int:
    return create_button_theme((150, 60, 70, 255))


def create_section_header_theme() -> int:
    """Theme for the small 'header bar' child windows in each panel."""
    with dpg.theme() as t:
        with dpg.theme_component(dpg.mvChildWindow):
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg, BG_HEADER)
            dpg.add_theme_color(dpg.mvThemeCol_Border,  BORDER)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding,    5.0)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding,    10.0, 6.0)
            dpg.add_theme_style(dpg.mvStyleVar_ChildBorderSize,  1.0)
    return t


def create_palette_button_theme() -> int:
    """Flat, left-aligned palette buttons."""
    with dpg.theme() as t:
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_color(dpg.mvThemeCol_Button,        (0, 0, 0, 0))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (54, 66, 90, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,  (68, 84, 114, 255))
            dpg.add_theme_style(dpg.mvStyleVar_ButtonTextAlign, 0.0, 0.5)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding,    8.0, 5.0)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,   4.0)
    return t
