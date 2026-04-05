"""Cyberpunk dark theme for the node tool."""

from __future__ import annotations
import dearpygui.dearpygui as dpg

# -- Palette -----------------------------------------------------------------
BG_DARK    = (8,   8,  16, 255)
BG_MID     = (16,  16, 28, 255)
BG_LIGHT   = (24,  24, 40, 255)
BG_POPUP   = (12,  12, 22, 255)
ACCENT     = (100, 200, 255, 255)
ACCENT2    = (180, 100, 255, 255)
ACCENT_DIM = (60, 130, 180, 255)
TEXT       = (220, 220, 240, 255)
TEXT_DIM   = (140, 140, 160, 255)
TEXT_BRIGHT= (255, 255, 255, 255)
BORDER     = (50,  50, 80,  200)
BORDER_H   = (100, 200, 255, 200)
SCROLLBAR  = (40,  40, 60,  200)

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

CATEGORY_COLORS = {
    "Math":    MATH_COLOR,
    "Logic":   LOGIC_COLOR,
    "String":  STRING_COLOR,
    "Data":    DATA_COLOR,
    "PyTorch": (PYTORCH_COLOR[0], PYTORCH_COLOR[1], PYTORCH_COLOR[2], 255),
    "NumPy":   (NUMPY_COLOR[0],   NUMPY_COLOR[1],   NUMPY_COLOR[2],   255),
    "Pandas":  (PANDAS_COLOR[0],  PANDAS_COLOR[1],  PANDAS_COLOR[2],  255),
    "Sklearn": (SKLEARN_COLOR[0], SKLEARN_COLOR[1], SKLEARN_COLOR[2], 255),
    "SciPy":   (SCIPY_COLOR[0],   SCIPY_COLOR[1],   SCIPY_COLOR[2],   255),
    "Viz":     (VIZ_COLOR[0],     VIZ_COLOR[1],     VIZ_COLOR[2],     255),
}


def create_global_theme() -> int:
    """Create and return the global cyberpunk theme tag."""
    with dpg.theme() as theme_id:
        with dpg.theme_component(dpg.mvAll):
            # Window / frame backgrounds
            dpg.add_theme_color(dpg.mvThemeCol_WindowBg,       BG_MID)
            dpg.add_theme_color(dpg.mvThemeCol_ChildBg,        BG_DARK)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBg,        BG_LIGHT)
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, (35, 35, 55, 255))
            dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive,  (45, 45, 70, 255))
            dpg.add_theme_color(dpg.mvThemeCol_PopupBg,        BG_POPUP)

            # Titles
            dpg.add_theme_color(dpg.mvThemeCol_TitleBg,           BG_DARK)
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive,     (20, 20, 35, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TitleBgCollapsed,  BG_DARK)

            # Text
            dpg.add_theme_color(dpg.mvThemeCol_Text,           TEXT)
            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled,   TEXT_DIM)

            # Buttons
            dpg.add_theme_color(dpg.mvThemeCol_Button,         (30, 60, 90, 220))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered,  (50, 120, 180, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive,   (70, 160, 220, 255))

            # Headers
            dpg.add_theme_color(dpg.mvThemeCol_Header,         (30, 60, 100, 200))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered,  (50, 100, 160, 220))
            dpg.add_theme_color(dpg.mvThemeCol_HeaderActive,   (70, 140, 210, 255))

            # Separator / border
            dpg.add_theme_color(dpg.mvThemeCol_Separator,         BORDER)
            dpg.add_theme_color(dpg.mvThemeCol_SeparatorHovered,  BORDER_H)
            dpg.add_theme_color(dpg.mvThemeCol_SeparatorActive,   ACCENT)

            # Sliders / checkboxes
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrab,       ACCENT)
            dpg.add_theme_color(dpg.mvThemeCol_SliderGrabActive, ACCENT2)
            dpg.add_theme_color(dpg.mvThemeCol_CheckMark,        ACCENT)

            # Scrollbar
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg,          BG_DARK)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab,        SCROLLBAR)
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered, (60, 60, 90, 255))
            dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabActive,  ACCENT_DIM)

            # Tab (using mvThemeCol_TabActive, not TabSelected which doesn't exist)
            dpg.add_theme_color(dpg.mvThemeCol_Tab,              (20, 20, 35, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TabHovered,       (40, 80, 130, 255))
            dpg.add_theme_color(dpg.mvThemeCol_TabActive,        (30, 60, 100, 255))

            # Resize grip
            dpg.add_theme_color(dpg.mvThemeCol_ResizeGrip,        (100, 200, 255, 60))
            dpg.add_theme_color(dpg.mvThemeCol_ResizeGripHovered,  (100, 200, 255, 160))
            dpg.add_theme_color(dpg.mvThemeCol_ResizeGripActive,   (100, 200, 255, 255))

            # Docking
            dpg.add_theme_color(dpg.mvThemeCol_DockingEmptyBg, BG_DARK)

            # Rounding / spacing
            dpg.add_theme_style(dpg.mvStyleVar_WindowRounding,    6.0)
            dpg.add_theme_style(dpg.mvStyleVar_ChildRounding,     4.0)
            dpg.add_theme_style(dpg.mvStyleVar_FrameRounding,     4.0)
            dpg.add_theme_style(dpg.mvStyleVar_PopupRounding,     4.0)
            dpg.add_theme_style(dpg.mvStyleVar_ScrollbarRounding, 4.0)
            dpg.add_theme_style(dpg.mvStyleVar_GrabRounding,      4.0)
            dpg.add_theme_style(dpg.mvStyleVar_TabRounding,       4.0)
            dpg.add_theme_style(dpg.mvStyleVar_WindowBorderSize,  1.0)
            dpg.add_theme_style(dpg.mvStyleVar_FrameBorderSize,   0.0)
            dpg.add_theme_style(dpg.mvStyleVar_WindowPadding,     10.0, 8.0)
            dpg.add_theme_style(dpg.mvStyleVar_FramePadding,      6.0, 4.0)
            dpg.add_theme_style(dpg.mvStyleVar_ItemSpacing,       8.0, 4.0)
            dpg.add_theme_style(dpg.mvStyleVar_IndentSpacing,     14.0)
            dpg.add_theme_style(dpg.mvStyleVar_ScrollbarSize,     12.0)

        # -- Node editor colors (use mvThemeCat_Nodes category) -------------
        with dpg.theme_component(dpg.mvNodeEditor):
            dpg.add_theme_color(dpg.mvNodeCol_GridBackground,      (8,  8,  16, 255),   category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_GridLine,            (28, 28, 45, 255),   category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_Link,                (100, 200, 255, 200), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_LinkHovered,         (150, 230, 255, 255), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_LinkSelected,        (180, 100, 255, 255), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_BoxSelector,         (100, 200, 255, 30),  category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_BoxSelectorOutline,  (100, 200, 255, 150), category=dpg.mvThemeCat_Nodes)

        with dpg.theme_component(dpg.mvNode):
            dpg.add_theme_color(dpg.mvNodeCol_NodeBackground,         (22, 22, 36, 240), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundHovered,  (28, 28, 46, 255), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_NodeBackgroundSelected, (30, 30, 50, 255), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_NodeOutline,            (50, 50, 80, 200), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_TitleBar,               (25, 25, 42, 255), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_TitleBarHovered,        (35, 35, 58, 255), category=dpg.mvThemeCat_Nodes)
            dpg.add_theme_color(dpg.mvNodeCol_TitleBarSelected,       (40, 40, 65, 255), category=dpg.mvThemeCat_Nodes)

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
    return create_button_theme((30, 140, 60, 255))


def create_clear_button_theme() -> int:
    return create_button_theme((120, 40, 40, 255))
