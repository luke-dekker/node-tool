"""Shared layout helpers for template builders."""
from __future__ import annotations


def grid(start_x: int = 60, start_y: int = 80,
         step_x: int = 220, step_y: int = 160):
    """Return a function that yields successive (x, y) positions in a grid.

    Use as `pos = grid(); pos(); pos()` for sequential placement, or
    `pos(col, row)` for explicit grid coordinates. Templates typically
    place nodes left-to-right along their data flow.
    """
    state = {"col": 0, "row": 0}

    def _next(col: int | None = None, row: int | None = None) -> tuple[int, int]:
        if col is None and row is None:
            col, row = state["col"], state["row"]
            state["col"] += 1
        return (start_x + col * step_x, start_y + row * step_y)

    return _next
