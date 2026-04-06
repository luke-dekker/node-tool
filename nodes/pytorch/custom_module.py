"""Custom Module node."""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


class CustomModuleNode(BaseNode):
    type_name   = "pt_custom_module"
    label       = "Custom Module"
    category    = "Models"
    subcategory = "Architecture"
    description = (
        "Write the body of forward(self, x). "
        "Wired sub-modules are self.mod_1 ... self.mod_4. "
        "torch, nn, and F (functional) are pre-imported."
    )

    _DEFAULT_CODE = "return self.mod_1(x)"

    def _setup_ports(self) -> None:
        self.add_input("forward_code", PortType.STRING,  default=self._DEFAULT_CODE,
                       description="Body of forward(self, x)")
        self.add_input("mod_1", PortType.MODULE, default=None)
        self.add_input("mod_2", PortType.MODULE, default=None)
        self.add_input("mod_3", PortType.MODULE, default=None)
        self.add_input("mod_4", PortType.MODULE, default=None)
        self.add_output("model", PortType.MODULE)
        self.add_output("error", PortType.STRING,
                        description="Build error if the code is invalid")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        code = (inputs.get("forward_code") or self._DEFAULT_CODE).strip()
        mods = {
            f"mod_{i}": inputs.get(f"mod_{i}")
            for i in range(1, 5)
            if inputs.get(f"mod_{i}") is not None
        }
        try:
            import textwrap
            import torch.nn as nn

            indented = textwrap.indent(code, "        ")
            src = (
                "import torch\n"
                "import torch.nn as nn\n"
                "import torch.nn.functional as F\n\n"
                "class _Dyn(nn.Module):\n"
                "    def __init__(self, **kw):\n"
                "        super().__init__()\n"
                "        for k, v in kw.items(): setattr(self, k, v)\n"
                "    def forward(self, x):\n"
                f"{indented}\n"
            )
            ns: dict = {}
            exec(compile(src, "<custom_module>", "exec"), ns)
            model = ns["_Dyn"](**mods)
            return {"model": model, "error": ""}
        except Exception as exc:
            return {"model": None, "error": str(exc)}

    def export(self, iv, ov):
        import textwrap
        code = self.inputs["forward_code"].default_value or "return self.mod_1(x)"
        mods = {f"mod_{i}": iv.get(f"mod_{i}") for i in range(1, 5) if iv.get(f"mod_{i}")}
        out  = ov.get("model", "_custom_mod")
        indented = textwrap.indent(code.strip(), "        ")
        mod_attrs = "\n".join(f"        self.{k} = {v}" for k, v in mods.items())
        lines = [
            "import torch",
            "import torch.nn as nn",
            "import torch.nn.functional as F",
            f"class _CustomMod_{out}(nn.Module):",
            f"    def __init__(self):",
            f"        super().__init__()",
            mod_attrs,
            f"    def forward(self, x):",
            indented,
            f"{out} = _CustomMod_{out}()",
        ]
        return [], lines
