"""Entry point for the Node Tool."""

from __future__ import annotations
import sys
import os
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from gui.app import NodeApp


def main() -> None:
    screenshot_path: str | None = None

    if "--screenshot" in sys.argv:
        screenshot_path = str(Path(__file__).parent / "screenshot.png")

    app = NodeApp()
    app.run(screenshot_path=screenshot_path)


if __name__ == "__main__":
    main()
