"""Prebuilt graph templates with hot-reload support.

Each template lives in its own .py file under templates/ and provides:

    LABEL       = "Display Name"          # required
    DESCRIPTION = "..."                   # optional, used as menu tooltip
    def build(graph: Graph) -> dict[str, tuple[int, int]]: ...

Templates are discovered automatically at app startup by scanning this
directory. Add a new template by dropping a new .py file in templates/.
The file watcher in templates/_reloader.py picks up additions and
modifications without restarting the app.

The TEMPLATES dict below is built once at import time from the file system.
Use `get_templates()` to access the current state — it returns a fresh list
every call so callers always see the latest registry.
"""
from __future__ import annotations
import importlib
import importlib.util
from pathlib import Path
from typing import Callable
from core.graph import Graph

TemplateBuilder = Callable[[Graph], dict[str, tuple[int, int]]]

# (label, description, builder)
TemplateEntry = tuple[str, str, TemplateBuilder]


TEMPLATES_DIR = Path(__file__).parent

# Module-level state populated at import + maintained by the reloader.
# Keyed by file stem so the reloader can update individual entries.
_REGISTRY: dict[str, TemplateEntry] = {}


def _is_template_file(path: Path) -> bool:
    """Filter for files that look like template modules."""
    if path.suffix != ".py":
        return False
    if path.name.startswith("_"):
        return False
    if path.name == "__init__.py":
        return False
    return True


def _load_template_file(path: Path) -> TemplateEntry | None:
    """Import a template .py file and extract its (label, description, build).

    Returns None if the file doesn't have a build() function (which means it
    isn't a template even if it ended up in templates/). Re-importing an
    already-imported module uses importlib.reload so live edits take effect.
    """
    mod_name = f"templates.{path.stem}"
    try:
        if mod_name in __import__("sys").modules:
            module = importlib.reload(__import__("sys").modules[mod_name])
        else:
            spec = importlib.util.spec_from_file_location(mod_name, path)
            if spec is None or spec.loader is None:
                return None
            module = importlib.util.module_from_spec(spec)
            __import__("sys").modules[mod_name] = module
            spec.loader.exec_module(module)
    except Exception as exc:
        print(f"[templates] failed to import {path.name}: {exc}")
        return None

    builder = getattr(module, "build", None)
    if not callable(builder):
        return None
    label = getattr(module, "LABEL", path.stem.replace("_", " ").title())
    description = getattr(module, "DESCRIPTION", "")
    return (label, description, builder)


def _discover() -> None:
    """Scan TEMPLATES_DIR and rebuild _REGISTRY from scratch."""
    _REGISTRY.clear()
    for path in sorted(TEMPLATES_DIR.glob("*.py")):
        if not _is_template_file(path):
            continue
        entry = _load_template_file(path)
        if entry is not None:
            _REGISTRY[path.stem] = entry


def get_templates() -> list[TemplateEntry]:
    """Return the current registry as a list, sorted by label.

    Returns a fresh copy each call so callers always see the latest set —
    the reloader can mutate _REGISTRY between calls.
    """
    return sorted(_REGISTRY.values(), key=lambda t: t[0].lower())


def reload_template(stem: str) -> TemplateEntry | None:
    """Re-import a single template by its file stem and update the registry.
    Used by TemplatesReloader.apply_event."""
    path = TEMPLATES_DIR / f"{stem}.py"
    if not path.exists():
        _REGISTRY.pop(stem, None)
        return None
    entry = _load_template_file(path)
    if entry is not None:
        _REGISTRY[stem] = entry
    return entry


def remove_template(stem: str) -> None:
    """Drop a template from the registry — used when a file is deleted."""
    _REGISTRY.pop(stem, None)


# Initial discovery at import time
_discover()


# Backward-compat shim: anyone reading TEMPLATES as a constant gets the
# current state. (The GUI now calls get_templates() in the menu builder so
# it sees live updates after the reloader runs.)
class _TemplatesProxy:
    """Sequence-like proxy that always reflects the current registry."""
    def __iter__(self):
        return iter(get_templates())
    def __len__(self):
        return len(_REGISTRY)
    def __getitem__(self, i):
        return get_templates()[i]


TEMPLATES = _TemplatesProxy()
