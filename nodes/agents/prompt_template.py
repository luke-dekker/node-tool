"""PromptTemplateNode — render a string template with {var} substitution."""
from __future__ import annotations
from typing import Any

from core.node import BaseNode, PortType


class PromptTemplateNode(BaseNode):
    type_name   = "ag_prompt_template"
    label       = "Prompt Template"
    category    = "Agents"
    subcategory = "Prompts"
    description = "Render a {var}-templated string. Vars come from the `vars` input dict."

    def _setup_ports(self) -> None:
        self.add_input("template", "PROMPT_TEMPLATE", default="",
                       description="Template with {var} slots")
        self.add_input("vars", PortType.ANY, default=None,
                       description="dict of substitution values (optional)")
        self.add_output("text", PortType.STRING, description="Rendered string")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        tpl = inputs.get("template") or ""
        vars_ = inputs.get("vars") or {}
        if not isinstance(vars_, dict):
            vars_ = {}
        try:
            text = tpl.format(**vars_)
        except (KeyError, IndexError):
            # Missing slot → leave the template intact rather than raising.
            text = tpl
        return {"text": text}

    def export(self, iv, ov):
        tpl = self._val(iv, "template")
        vars_expr = iv.get("vars") or "{}"
        out = ov.get("text", "_text")
        return [], [f"{out} = {tpl}.format(**({vars_expr} or {{}}))"]
