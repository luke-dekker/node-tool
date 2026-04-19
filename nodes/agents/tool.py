"""ToolNode — wraps a Python callable resolved from a dotted path."""
from __future__ import annotations
import importlib
import json
from typing import Any

from core.node import BaseNode, PortType


class ToolNode(BaseNode):
    type_name   = "ag_tool"
    label       = "Tool"
    category    = "Agents"
    subcategory = "Tools"
    description = ("Bind a Python callable (referenced by dotted path) as a tool "
                   "the agent can call.")

    def _setup_ports(self) -> None:
        self.add_input("name", PortType.STRING, default="my_tool",
                       description="Tool name the LLM will reference")
        self.add_input("description", PortType.STRING, default="",
                       description="Drives the LLM's choice of when to call this tool")
        self.add_input("input_schema", PortType.STRING, default="",
                       description=("JSON Schema for the tool's arguments. Empty = "
                                    "free-form object."))
        self.add_input("python_callable", PortType.STRING, default="",
                       description="Dotted path, e.g. 'datetime.datetime.now'")
        self.add_input("side_effect", PortType.BOOL, default=False,
                       description=("True if the tool can mutate the system "
                                    "(shell, file, network)."))
        self.add_output("tool", "TOOL")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        from plugins.agents._llm.protocol import ToolDef  # deferred

        name = (inputs.get("name") or "tool").strip() or "tool"
        desc = (inputs.get("description") or "").strip()
        schema_raw = (inputs.get("input_schema") or "").strip()
        path = (inputs.get("python_callable") or "").strip()

        if schema_raw:
            try:
                schema = json.loads(schema_raw)
                if not isinstance(schema, dict):
                    raise ValueError("input_schema must decode to an object")
            except (json.JSONDecodeError, ValueError) as exc:
                raise RuntimeError(f"ToolNode {name!r}: invalid input_schema JSON: {exc}")
        else:
            schema = {"type": "object", "properties": {}, "additionalProperties": True}

        if not path:
            raise RuntimeError(f"ToolNode {name!r}: python_callable is empty")
        callable_obj = _resolve_dotted(path)

        return {"tool": ToolDef(
            name=name, description=desc, input_schema=schema,
            callable=callable_obj, side_effect=bool(inputs.get("side_effect")),
        )}

    def export(self, iv, ov):
        return [], [f"# ToolNode export pending Phase D"]


def _resolve_dotted(path: str) -> Any:
    """Resolve 'pkg.mod.attr' or 'pkg.mod:attr' to a Python object."""
    if ":" in path:
        mod_name, attr_path = path.split(":", 1)
    else:
        # Walk left until import succeeds.
        parts = path.split(".")
        for i in range(len(parts), 0, -1):
            mod_name = ".".join(parts[:i])
            try:
                importlib.import_module(mod_name)
                attr_path = ".".join(parts[i:])
                break
            except ImportError:
                continue
        else:
            raise RuntimeError(f"Cannot import any prefix of {path!r}")
    obj = importlib.import_module(mod_name)
    if attr_path:
        for attr in attr_path.split("."):
            obj = getattr(obj, attr)
    return obj
