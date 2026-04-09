"""ClassModuleImportNode — instantiate a class from a Python file as a Module.

The other half of the class-export loop. Workflow:
  1. Build a graph and export it as a class via File -> Export As Class...
     (or GraphExporter().export(g, mode='class', class_name='MyModel'))
  2. In a new graph, drop a Class Module Import node
  3. Set `path` to the .py file and `class_name` to the class inside it
  4. Optionally pass init_args as a JSON object to the constructor
  5. The output `model` port carries the instantiated nn.Module — wire it
     anywhere a MODULE is expected (Adam, training config, etc.)

Re-imports happen lazily: if the file's mtime or class_name changes, the
class is reloaded so you can iterate on the source file and see changes
without restarting the app.
"""
from __future__ import annotations
import os
from typing import Any
from core.node import BaseNode, PortType


class ClassModuleImportNode(BaseNode):
    type_name   = "pt_class_module_import"
    label       = "Class Module Import"
    category    = "Models"
    subcategory = "Pretrained"
    description = (
        "Import a Python class from a .py file and instantiate it as a Module. "
        "Use it to drop a class-exported graph into a new graph as a single node. "
        "init_args is a JSON object passed as keyword arguments to the constructor."
    )

    def __init__(self):
        self._cached_module = None
        self._cached_cfg: tuple = ()
        self._cached_mtime: float = 0.0
        super().__init__()

    def _setup_ports(self) -> None:
        # Pure model factory: imports + instantiates a class as a MODULE.
        # To run a tensor through it, wire the `model` output into an
        # ApplyModuleNode (which handles forward + shape introspection).
        self.add_input("path",        PortType.STRING, default="my_model.py",
                       description="Path to a .py file containing the class")
        self.add_input("class_name",  PortType.STRING, default="GraphModel",
                       description="Name of the class inside the file")
        self.add_input("init_args",   PortType.STRING, default="{}",
                       description="JSON object of constructor kwargs, e.g. '{\"hidden\": 128}'")
        self.add_input("device",      PortType.STRING, default="cpu")

        self.add_output("model",        PortType.MODULE,
                        description="The instantiated module — wire to ApplyModule for forward, "
                                    "into Adam, into TrainingConfig, etc.")
        self.add_output("info",         PortType.STRING)
        self.add_output("param_count",  PortType.INT)

    def get_layers(self) -> list:
        """Expose the imported module as a layer for GraphAsModule's parameter
        collection. Mirrors LoadModelNode's pattern."""
        return [self._cached_module] if self._cached_module is not None else []

    def _resolve_path(self, path: str) -> str:
        """Allow paths relative to the project root, the templates dir, or absolute."""
        if os.path.isabs(path) and os.path.exists(path):
            return path
        for root in (".", os.path.dirname(__file__) + "/../..", "templates"):
            candidate = os.path.join(root, path)
            if os.path.exists(candidate):
                return candidate
        return path

    def _import_class(self, path: str, class_name: str):
        """Import a class from a .py file by path. Re-imports on file change."""
        import importlib.util
        resolved = self._resolve_path(path)
        if not os.path.exists(resolved):
            raise FileNotFoundError(f"File not found: {path}")
        mtime = os.path.getmtime(resolved)
        cfg = (resolved, class_name)
        if (self._cached_module is not None
                and self._cached_cfg == cfg
                and self._cached_mtime == mtime):
            return type(self._cached_module)  # already loaded, unchanged

        spec = importlib.util.spec_from_file_location(
            f"_class_module_import_{abs(hash(resolved)) & 0xffff}", resolved
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create import spec for {resolved}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls = getattr(module, class_name, None)
        if cls is None:
            raise AttributeError(f"Class {class_name!r} not found in {resolved}")
        self._cached_cfg = cfg
        self._cached_mtime = mtime
        return cls

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import json
        empty = {"model": None, "info": "", "param_count": 0}
        path       = (inputs.get("path") or "").strip()
        class_name = (inputs.get("class_name") or "").strip()
        init_args  = (inputs.get("init_args") or "{}").strip() or "{}"
        device     = (inputs.get("device") or "cpu").strip() or "cpu"
        if not path or not class_name:
            return {**empty, "info": "Set path and class_name"}

        try:
            cls = self._import_class(path, class_name)
        except Exception as exc:
            self._cached_module = None
            return {**empty, "info": f"Import failed: {exc}"}

        try:
            kwargs = json.loads(init_args) if init_args else {}
            if not isinstance(kwargs, dict):
                kwargs = {}
        except json.JSONDecodeError as exc:
            return {**empty, "info": f"init_args JSON error: {exc}"}

        try:
            self._cached_module = cls(**kwargs)
            try:
                self._cached_module.to(device)
            except Exception:
                pass
        except Exception as exc:
            self._cached_module = None
            return {**empty, "info": f"Constructor failed: {exc}"}

        params = sum(p.numel() for p in self._cached_module.parameters())
        info = f"{class_name} | {params:,} params | from {os.path.basename(path)}"
        return {"model": self._cached_module, "info": info, "param_count": params}

    def export(self, iv, ov):
        path       = self._val(iv, "path")
        class_name = self._val(iv, "class_name")
        init_args  = self._val(iv, "init_args")
        device     = self._val(iv, "device")
        m_var      = ov.get("model",       "_imported")
        info_var   = ov.get("info",        "_imp_info")
        p_var      = ov.get("param_count", "_imp_params")

        return ["import json", "import importlib.util"], [
            f"import importlib.util as _ilu",
            f"_spec = _ilu.spec_from_file_location('_imported_mod', {path})",
            f"_mod  = _ilu.module_from_spec(_spec)",
            f"_spec.loader.exec_module(_mod)",
            f"_cls  = getattr(_mod, {class_name})",
            f"{m_var} = _cls(**json.loads({init_args} or '{{}}'))",
            f"{m_var}.to({device})",
            f"{p_var} = sum(p.numel() for p in {m_var}.parameters())",
            f"{info_var} = f'{{{class_name}}} | {{{p_var}:,}} params'",
        ]
