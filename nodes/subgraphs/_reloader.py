"""SubgraphReloader — file-mtime polling for the subgraphs/ directory.

Mirrors core/custom.HotReloader but specialized for .subgraph.json files.
On file changes, the old generated class is removed from NODE_REGISTRY and a
fresh one (potentially with updated ports) is registered. Deleted files are
unregistered. New files are loaded.

The reloader returns event tuples so the app loop can log them and refresh the
palette UI in place.
"""
from __future__ import annotations
import time
from pathlib import Path
from typing import Literal

from core.subgraph import SubgraphFile, SUBGRAPHS_DIR
from nodes.subgraphs._base import SubgraphNode
from nodes.subgraphs import _make_subgraph_class


# Event kinds: (kind, type_name, message, cls)
# - kind=='added':   newly discovered file → cls is the freshly built class
# - kind=='changed': existing file modified → cls is the rebuilt class
# - kind=='removed': file deleted → cls is None
SubgraphEvent = tuple[Literal["added", "changed", "removed"], str, str, type | None]


def _type_name_for(path: Path) -> str:
    """Stable type_name from a subgraph file path."""
    stem = path.name
    if stem.endswith(".subgraph.json"):
        stem = stem[: -len(".subgraph.json")]
    return f"subgraph_{stem}"


class SubgraphReloader:
    """Polls subgraphs/ once per second; reloads changed .subgraph.json files."""

    def __init__(self):
        SUBGRAPHS_DIR.mkdir(parents=True, exist_ok=True)
        self._mtimes: dict[str, float] = {}
        self._known: set[str] = set()
        self._last_check = 0.0

    def poll(self) -> list[SubgraphEvent]:
        """Call once per frame. Returns a list of events for changed files.

        Caller is responsible for updating NODE_REGISTRY and the palette UI
        based on the returned events. Doing the registry mutation here would
        couple the reloader to the registry's lifecycle in unhelpful ways.
        """
        now = time.monotonic()
        if now - self._last_check < 1.0:
            return []
        self._last_check = now

        events: list[SubgraphEvent] = []

        # Detect added / changed
        seen_paths: set[str] = set()
        for path in sorted(SUBGRAPHS_DIR.glob("*.subgraph.json")):
            path_str = str(path)
            seen_paths.add(path_str)
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue

            type_name = _type_name_for(path)

            if path_str not in self._mtimes:
                # Newly discovered
                cls = self._build_class(path, type_name)
                if cls is not None:
                    self._mtimes[path_str] = mtime
                    self._known.add(type_name)
                    events.append(("added", type_name,
                                   f"[Subgraphs] Loaded {path.name}", cls))
            elif mtime > self._mtimes[path_str]:
                # Existing file changed — rebuild
                cls = self._build_class(path, type_name)
                if cls is not None:
                    self._mtimes[path_str] = mtime
                    events.append(("changed", type_name,
                                   f"[Subgraphs] Reloaded {path.name}", cls))

        # Detect deletions
        deleted = [p for p in self._mtimes if p not in seen_paths]
        for p in deleted:
            self._mtimes.pop(p, None)
            type_name = _type_name_for(Path(p))
            self._known.discard(type_name)
            events.append(("removed", type_name,
                           f"[Subgraphs] Removed {Path(p).name}", None))

        return events

    def _build_class(self, path: Path, type_name: str) -> type | None:
        try:
            sf = SubgraphFile.load(path)
        except Exception as exc:
            print(f"[subgraphs] failed to reload {path.name}: {exc}")
            return None
        try:
            return _make_subgraph_class(sf, type_name)
        except Exception as exc:
            print(f"[subgraphs] failed to build class for {path.name}: {exc}")
            return None

    # ── Helpers for the app to apply events ─────────────────────────────

    @staticmethod
    def apply_event(event: SubgraphEvent) -> None:
        """Mutate NODE_REGISTRY (and the subgraphs module globals) for one event."""
        from nodes import NODE_REGISTRY
        from nodes import subgraphs as sg_mod
        kind, type_name, _msg, cls = event
        if kind in ("added", "changed"):
            if cls is not None:
                NODE_REGISTRY[type_name] = cls
                sg_mod._GENERATED[type_name] = cls
                # Inject into the subgraphs module's namespace so attribute
                # lookups (used by _discover) work consistently
                setattr(sg_mod, cls.__name__, cls)
        elif kind == "removed":
            NODE_REGISTRY.pop(type_name, None)
            old = sg_mod._GENERATED.pop(type_name, None)
            if old is not None:
                # Strip from module globals so it can be GC'd
                try:
                    delattr(sg_mod, old.__name__)
                except AttributeError:
                    pass
