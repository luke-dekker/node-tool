"""PythonFunctionToolNode — compile inline Python into a tool callable.

Phase B v1: raw `exec` behind `allow_side_effect_tools` flag, no isolation.
Per Decision 1 of plugins/agents/DESIGN.md (solo-use posture). Revisit
sandboxing if/when graphs are shared publicly.
"""
from __future__ import annotations
import json
import textwrap
from typing import Any

from core.node import BaseNode, PortType


class PythonFunctionToolNode(BaseNode):
    type_name   = "ag_python_function_tool"
    label       = "Python Function Tool"
    category    = "Agents"
    subcategory = "Tools"
    description = ("Bind an inline Python function as a tool. Body has access "
                   "to the kwargs the LLM passes. WARNING: runs as raw exec; "
                   "side_effect=True by default.")

    _DEFAULT_BODY = textwrap.dedent("""\
        # Inline tool body. Available: any kwargs the LLM passes.
        # Return value becomes the tool result the LLM sees.
        from datetime import datetime
        return datetime.now().isoformat(timespec='seconds')
    """)

    def _setup_ports(self) -> None:
        self.add_input("name", PortType.STRING, default="get_time",
                       description="Tool name the LLM will reference")
        self.add_input("description", PortType.STRING,
                       default="Return the current local time.",
                       description="Drives LLM's choice of when to call this tool")
        self.add_input("input_schema", PortType.STRING, default="",
                       description=("JSON Schema for arguments. Empty = "
                                    "no arguments."))
        self.add_input("code", PortType.STRING, default=self._DEFAULT_BODY,
                       description="Function body. Receives **kwargs from the LLM.")
        self.add_input("side_effect", PortType.BOOL, default=True,
                       description=("Inline exec is dangerous. Default True. "
                                    "Flip to False only if your code is pure."))
        self.add_output("tool", "TOOL")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        from plugins.agents._llm.protocol import ToolDef  # deferred

        name = (inputs.get("name") or "tool").strip() or "tool"
        desc = (inputs.get("description") or "").strip()
        schema_raw = (inputs.get("input_schema") or "").strip()
        body = inputs.get("code") or ""

        if schema_raw:
            try:
                schema = json.loads(schema_raw)
                if not isinstance(schema, dict):
                    raise ValueError("input_schema must decode to an object")
            except (json.JSONDecodeError, ValueError) as exc:
                raise RuntimeError(
                    f"PythonFunctionToolNode {name!r}: invalid input_schema JSON: {exc}"
                )
        else:
            schema = {"type": "object", "properties": {}, "additionalProperties": True}

        callable_obj = _compile_inline_function(name, body)

        return {"tool": ToolDef(
            name=name, description=desc, input_schema=schema,
            callable=callable_obj, side_effect=bool(inputs.get("side_effect", True)),
        )}

    def export(self, iv, ov):
        name = (self.inputs["name"].default_value or "tool").strip() or "tool"
        desc = (self.inputs["description"].default_value or "").strip()
        schema_raw = (self.inputs["input_schema"].default_value or "").strip()
        if schema_raw:
            try:
                schema = json.loads(schema_raw)
            except (json.JSONDecodeError, ValueError):
                schema = {"type": "object", "properties": {}, "additionalProperties": True}
        else:
            schema = {"type": "object", "properties": {}, "additionalProperties": True}
        body = self.inputs["code"].default_value or "return None"
        side = bool(self.inputs["side_effect"].default_value)

        out = ov.get("tool", f"_tool_{name}")
        safe_fn = "_inline_" + "".join(
            c if (c.isalnum() or c == "_") else "_" for c in name
        )
        fn_def = "def " + safe_fn + "(**kwargs):\n" + textwrap.indent(body, "    ")
        lines = fn_def.splitlines() + [
            f"{out} = {{'name': {name!r}, 'description': {desc!r}, "
            f"'input_schema': {schema!r}, 'side_effect': {side!r}, "
            f"'callable': {safe_fn}}}"
        ]
        return [], lines


def _compile_inline_function(name: str, body: str):
    """Compile `body` as the body of `def <name>(**kwargs): ...`.

    Returns a callable. Body has full access to imports, globals, etc. — this
    is intentional per the v1 sandbox decision.
    """
    safe_name = "_inline_" + "".join(c if (c.isalnum() or c == "_") else "_" for c in name)
    src = "def " + safe_name + "(**kwargs):\n" + textwrap.indent(body or "    return None", "    ")
    namespace: dict[str, Any] = {}
    try:
        exec(src, namespace, namespace)
    except SyntaxError as exc:
        raise RuntimeError(
            f"PythonFunctionToolNode {name!r}: syntax error: {exc}"
        )
    fn = namespace.get(safe_name)
    if fn is None:
        raise RuntimeError(
            f"PythonFunctionToolNode {name!r}: failed to compile inline function"
        )
    return fn
