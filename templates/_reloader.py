"""TemplatesReloader — file-mtime polling for the templates/ directory.

Mirrors nodes/subgraphs/_reloader.SubgraphReloader but watches Python template
files instead of JSON subgraph files. On change, the corresponding template
module is re-imported via importlib.reload and the registry in
templates/__init__.py is updated; the GUI's File -> Templates menu refreshes
in place via gui/mixins/polling._poll_template_reload.

Like the subgraph reloader, this returns events but doesn't mutate state on
its own — caller decides via apply_event(). Keeps the watcher decoupled
from the registry's lifecycle.
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import Literal

from templates import TEMPLATES_DIR, _is_template_file, reload_template, remove_template


# Event kinds: (kind, stem, message, entry)
# - kind == 'added':   newly discovered file → entry is the freshly built (label, desc, builder)
# - kind == 'changed': existing file modified → entry is the rebuilt one
# - kind == 'removed': file deleted → entry is None
TemplateEvent = tuple[Literal["added", "changed", "removed"], str, str, tuple | None]


class TemplatesReloader:
    """Polls templates/ once per second; reloads changed .py files."""

    def __init__(self):
        TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
        self._mtimes: dict[str, float] = {}
        self._last_check = 0.0

    def poll(self) -> list[TemplateEvent]:
        """Call once per frame. Returns events for changed files.

        Caller is responsible for applying events to the registry and refreshing
        the menu UI. apply_event() handles the registry update; the GUI's
        polling mixin handles the menu refresh.
        """
        now = time.monotonic()
        if now - self._last_check < 1.0:
            return []
        self._last_check = now

        events: list[TemplateEvent] = []
        seen_paths: set[str] = set()

        for path in sorted(TEMPLATES_DIR.glob("*.py")):
            if not _is_template_file(path):
                continue
            path_str = str(path)
            seen_paths.add(path_str)
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue

            stem = path.stem
            if path_str not in self._mtimes:
                # New file (or first scan — apply silently if entry already in registry)
                self._mtimes[path_str] = mtime
                from templates import _REGISTRY
                if stem not in _REGISTRY:
                    events.append(("added", stem,
                                   f"[Templates] Loaded {path.name}", None))
                    # The entry will be loaded by apply_event
            elif mtime > self._mtimes[path_str]:
                self._mtimes[path_str] = mtime
                events.append(("changed", stem,
                               f"[Templates] Reloaded {path.name}", None))

        # Detect deletions
        deleted = [p for p in self._mtimes if p not in seen_paths]
        for p in deleted:
            self._mtimes.pop(p, None)
            stem = Path(p).stem
            events.append(("removed", stem,
                           f"[Templates] Removed {Path(p).name}", None))

        return events

    @staticmethod
    def apply_event(event: TemplateEvent) -> None:
        """Mutate the templates registry for one event."""
        kind, stem, _msg, _entry = event
        if kind in ("added", "changed"):
            reload_template(stem)
        elif kind == "removed":
            remove_template(stem)
