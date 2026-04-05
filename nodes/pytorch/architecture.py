"""Non-sequential architecture nodes — Residual, Concat, Add, Custom Module."""

from __future__ import annotations
from typing import Any
from core.node import BaseNode
from core.node import PortType

CATEGORY = "Models"


# ── Residual Block ─────────────────────────────────────────────────────────────

class ResidualBlockNode(BaseNode):
    type_name   = "pt_residual_block"
    label       = "Residual Block"
    category    = CATEGORY
    subcategory = "Architecture"
    description = (
        "x + block(x). Wire an optional projection module to match dimensions "
        "(e.g. a 1×1 Conv when channels change)."
    )

    def _setup_ports(self) -> None:
        self.add_input("block",      PortType.MODULE, default=None,
                       description="F(x) — the residual path")
        self.add_input("projection", PortType.MODULE, default=None,
                       description="Optional shortcut projection (1×1 conv / linear)")
        self.add_output("model", PortType.MODULE)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        block = inputs.get("block")
        proj  = inputs.get("projection")
        if block is None:
            return {"model": None}
        try:
            import torch.nn as nn
            _block, _proj = block, proj

            class _ResBlock(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.block = _block
                    self.projection = _proj

                def forward(self, x):
                    identity = x if self.projection is None else self.projection(x)
                    return identity + self.block(x)

            return {"model": _ResBlock()}
        except Exception:
            return {"model": None}


# ── Concat Branches ────────────────────────────────────────────────────────────

class ConcatBranchesNode(BaseNode):
    type_name   = "pt_concat_branches"
    label       = "Concat Branches"
    category    = CATEGORY
    subcategory = "Architecture"
    description = (
        "torch.cat([branch_1(x), branch_2(x), ...], dim=dim). "
        "Useful for Inception-style parallel paths."
    )

    def _setup_ports(self) -> None:
        self.add_input("branch_1", PortType.MODULE, default=None)
        self.add_input("branch_2", PortType.MODULE, default=None)
        self.add_input("branch_3", PortType.MODULE, default=None)
        self.add_input("branch_4", PortType.MODULE, default=None)
        self.add_input("dim",      PortType.INT,    default=1,
                       description="Concat dim (1 = channel axis for images)")
        self.add_output("model",   PortType.MODULE)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        branches = [inputs.get(f"branch_{i}") for i in range(1, 5)
                    if inputs.get(f"branch_{i}") is not None]
        if not branches:
            return {"model": None}
        try:
            import torch
            import torch.nn as nn
            _branches = branches
            _dim = int(inputs.get("dim") or 1)

            class _ConcatMod(nn.Module):
                def __init__(self):
                    super().__init__()
                    for i, b in enumerate(_branches):
                        setattr(self, f"branch_{i}", b)
                    self._n   = len(_branches)
                    self._dim = _dim

                def forward(self, x):
                    return torch.cat(
                        [getattr(self, f"branch_{i}")(x) for i in range(self._n)],
                        dim=self._dim,
                    )

            return {"model": _ConcatMod()}
        except Exception:
            return {"model": None}


# ── Add Branches ───────────────────────────────────────────────────────────────

class AddBranchesNode(BaseNode):
    type_name   = "pt_add_branches"
    label       = "Add Branches"
    category    = CATEGORY
    subcategory = "Architecture"
    description = "branch_1(x) + branch_2(x). For parallel paths summed together."

    def _setup_ports(self) -> None:
        self.add_input("branch_1", PortType.MODULE, default=None)
        self.add_input("branch_2", PortType.MODULE, default=None)
        self.add_output("model",   PortType.MODULE)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        b1 = inputs.get("branch_1")
        b2 = inputs.get("branch_2")
        if b1 is None or b2 is None:
            return {"model": None}
        try:
            import torch.nn as nn
            _b1, _b2 = b1, b2

            class _AddMod(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.branch_1 = _b1
                    self.branch_2 = _b2

                def forward(self, x):
                    return self.branch_1(x) + self.branch_2(x)

            return {"model": _AddMod()}
        except Exception:
            return {"model": None}


# ── Custom Module ──────────────────────────────────────────────────────────────

class CustomModuleNode(BaseNode):
    type_name   = "pt_custom_module"
    label       = "Custom Module"
    category    = CATEGORY
    subcategory = "Architecture"
    description = (
        "Write the body of forward(self, x). "
        "Wired sub-modules are self.mod_1 … self.mod_4. "
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
