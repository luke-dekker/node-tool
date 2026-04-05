"""Custom node support — hot-reload watcher and @node function decorator.

Drop a .py file into nodes/custom/ and it will be loaded (and reloaded on save)
automatically. Use the @node decorator to define nodes as plain functions.

    from core.custom import node

    @node(label="Double", category="Math")
    def double(value: float = 1.0) -> float:
        return value * 2
"""
from __future__ import annotations
import importlib
import importlib.util
import inspect
import sys
import time
from pathlib import Path
from typing import get_type_hints, Any

from core.node import BaseNode, Port, PortType

CUSTOM_DIR = Path(__file__).parent.parent / "nodes" / "custom"


# ── Hot-reloader ──────────────────────────────────────────────────────────────

class HotReloader:
    """Polls nodes/custom/ once per second; reloads changed .py files."""

    def __init__(self):
        CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
        self._mtimes:  dict[str, float] = {}
        self._modules: dict[str, object] = {}
        self._last_check = 0.0

    def poll(self) -> list[tuple[str, list[str]]]:
        """Call once per frame. Returns [(message, [new_type_names])] for changed files."""
        now = time.monotonic()
        if now - self._last_check < 1.0:
            return []
        self._last_check = now

        results: list[tuple[str, list[str]]] = []
        for py_file in sorted(CUSTOM_DIR.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            path_str = str(py_file)
            try:
                mtime = py_file.stat().st_mtime
            except OSError:
                continue

            if path_str not in self._mtimes:
                msg, new_types = self._load(py_file)
                results.append((msg, new_types))
                self._mtimes[path_str] = mtime
            elif mtime > self._mtimes[path_str]:
                msg, new_types = self._reload(py_file)
                results.append((msg, new_types))
                self._mtimes[path_str] = mtime

        return results

    def _snapshot(self) -> set[str]:
        from nodes import NODE_REGISTRY
        return set(NODE_REGISTRY.keys())

    def _new_since(self, before: set[str]) -> list[str]:
        from nodes import NODE_REGISTRY
        return [t for t in NODE_REGISTRY if t not in before]

    def _load(self, path: Path) -> tuple[str, list[str]]:
        before = self._snapshot()
        try:
            mod_name = f"nodes.custom.{path.stem}"
            spec = importlib.util.spec_from_file_location(mod_name, path)
            if spec is None or spec.loader is None:
                return f"[Custom] Could not load {path.name}", []
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)  # type: ignore[union-attr]
            self._modules[str(path)] = module
            new_types = self._new_since(before)
            return f"[Custom] Loaded {path.name} (+{len(new_types)} node(s))", new_types
        except Exception as exc:
            return f"[Custom] Error loading {path.name}: {exc}", []

    def _reload(self, path: Path) -> tuple[str, list[str]]:
        before = self._snapshot()
        module = self._modules.get(str(path))
        if module is None:
            return self._load(path)
        try:
            spec = importlib.util.spec_from_file_location(module.__name__, path)
            if spec is None or spec.loader is None:
                return f"[Custom] Could not reload {path.name}", []
            spec.loader.exec_module(module)  # type: ignore[union-attr]
            new_types = self._new_since(before)
            return f"[Custom] Reloaded {path.name} (+{len(new_types)} new node(s))", new_types
        except Exception as exc:
            return f"[Custom] Error reloading {path.name}: {exc}", []


# ── @node decorator ───────────────────────────────────────────────────────────

_TYPE_MAP: dict[type, PortType] = {
    float: PortType.FLOAT,
    int:   PortType.INT,
    bool:  PortType.BOOL,
    str:   PortType.STRING,
}

def _pt(t) -> PortType:
    return _TYPE_MAP.get(t, PortType.ANY)


def _make_node_class(fn, type_name, label, category, description,
                     input_defs, output_defs, multi_output) -> type:
    _input_defs  = input_defs
    _output_defs = output_defs
    _fn          = fn
    _multi       = multi_output

    def _setup_ports(self):
        for name, ptype, default in _input_defs:
            self.add_input(name, ptype, default=default)
        for name, ptype in _output_defs:
            self.add_output(name, ptype)

    def execute(self, inputs):
        try:
            kwargs = {name: inputs.get(name) for name, _, _ in _input_defs}
            result = _fn(**kwargs)
            if _multi:
                return result if isinstance(result, dict) else {_output_defs[0][0]: result}
            return {_output_defs[0][0]: result}
        except Exception:
            return {name: None for name, _ in _output_defs}

    return type(
        f"_SimpleNode_{type_name}",
        (BaseNode,),
        {
            "type_name":    type_name,
            "label":        label,
            "category":     category,
            "description":  description,
            "_setup_ports": _setup_ports,
            "execute":      execute,
        },
    )


def node(
    label: str,
    category: str = "Custom",
    description: str = "",
    outputs: dict[str, type] | None = None,
    type_name: str | None = None,
):
    """Decorator to register a plain function as a node.

    Parameters
    ----------
    label       Display name on the canvas.
    category    Palette group (default "Custom").
    description Tooltip / inspector text.
    outputs     Dict of {port_name: type} for multi-output nodes.
    type_name   Override the auto-derived key (default = function name).
    """
    def decorator(fn):
        hints  = get_type_hints(fn)
        sig    = inspect.signature(fn)
        _name  = type_name or fn.__name__
        _desc  = description or (fn.__doc__ or "").strip()

        input_defs: list[tuple[str, PortType, Any]] = []
        for param_name, param in sig.parameters.items():
            ptype   = _pt(hints.get(param_name))
            default = (param.default
                       if param.default is not inspect.Parameter.empty
                       else ptype.default_value())
            input_defs.append((param_name, ptype, default))

        if outputs is not None:
            output_defs  = [(n, _pt(t)) for n, t in outputs.items()]
            multi_output = True
        else:
            ret = hints.get("return")
            if ret is not None and ret is not dict:
                output_defs = [("result", _pt(ret))]
            else:
                output_defs = [("result", PortType.ANY)]
            multi_output = False

        cls = _make_node_class(fn, _name, label, category, _desc,
                               input_defs, output_defs, multi_output)

        from nodes import NODE_REGISTRY
        NODE_REGISTRY[_name] = cls
        return fn

    return decorator
