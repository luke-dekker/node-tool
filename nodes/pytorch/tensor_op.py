"""Consolidated tensor compute node — replaces TensorBinaryOpNode,
ArgmaxNode, SoftmaxOpNode, TensorEinsumNode, TensorMuxNode.

Pick `op`:
  add | sub | mul | div  — element-wise binary (a, b → result)
  argmax | softmax       — single-tensor reduction along `dim` (a → result)
  einsum                 — torch.einsum(equation, a, b) → result
  mux                    — select(a, b) by `select` key — runtime branch

Output: result (TENSOR).
"""
from __future__ import annotations
from core.node import BaseNode, PortType


_BINARY = ["add", "sub", "mul", "div"]
_REDUCE = ["argmax", "softmax"]
_OPS    = _BINARY + _REDUCE + ["einsum", "mux"]
_PY_OPS = {"add": "+", "sub": "-", "mul": "*", "div": "/"}


class TensorOpNode(BaseNode):
    type_name   = "pt_tensor_op"
    label       = "Tensor Op"
    category    = "Tensor Ops"
    subcategory = ""
    description = (
        "Tensor operation. Pick `op`:\n"
        "  add/sub/mul/div — element-wise binary on (a, b)\n"
        "  argmax/softmax  — reduce a along `dim`\n"
        "  einsum          — torch.einsum(equation, a, b)\n"
        "  mux             — select between (a, b) by `select`"
    )

    def relevant_inputs(self, values):
        op = (values.get("op") or "add").strip()
        if op in _BINARY:    return ["op"]                       # a, b are wired
        if op in _REDUCE:    return ["op", "dim"]
        if op == "einsum":   return ["op", "equation"]
        if op == "mux":      return ["op", "select"]
        return ["op"]

    def _setup_ports(self):
        self.add_input("a",        PortType.TENSOR, default=None)
        self.add_input("b",        PortType.TENSOR, default=None, optional=True)
        self.add_input("op",       PortType.STRING, default="add", choices=_OPS)
        self.add_input("dim",      PortType.INT,    default=-1, optional=True)
        self.add_input("equation", PortType.STRING, default="ij,jk->ik", optional=True)
        self.add_input("select",   PortType.STRING, default="a", optional=True,
                       description="mux: 'a'/'b' or 0/1")
        self.add_output("result", PortType.TENSOR)

    def execute(self, inputs):
        try:
            import torch
            op = (inputs.get("op") or "add").strip()
            a, b = inputs.get("a"), inputs.get("b")
            if op in _BINARY:
                if a is None or b is None:
                    return {"result": None}
                if op == "add": return {"result": a + b}
                if op == "sub": return {"result": a - b}
                if op == "mul": return {"result": a * b}
                if op == "div": return {"result": a / b}
            if op in _REDUCE:
                if a is None:
                    return {"result": None}
                dim = int(inputs.get("dim", -1))
                fn = torch.argmax if op == "argmax" else torch.softmax
                return {"result": fn(a, dim=dim)}
            if op == "einsum":
                if a is None or b is None:
                    return {"result": None}
                eq = str(inputs.get("equation") or "ij,jk->ik")
                return {"result": torch.einsum(eq, a, b)}
            if op == "mux":
                sel = inputs.get("select", "a")
                pick_a = (
                    isinstance(sel, bool) and not sel
                    or isinstance(sel, (int, float)) and int(sel) == 0
                    or str(sel).strip().lower() in ("a", "0", "", "left", "first")
                )
                chosen = a if pick_a else b
                if chosen is None:
                    chosen = a if a is not None else b
                return {"result": chosen}
            return {"result": None}
        except Exception:
            return {"result": None}

    def export(self, iv, ov):
        op = (self.inputs["op"].default_value or "add")
        out = ov["result"]
        a = self._val(iv, "a")
        if op in _BINARY:
            return [], [f"{out} = {a} {_PY_OPS[op]} {self._val(iv, 'b')}"]
        if op in _REDUCE:
            fn = "torch.argmax" if op == "argmax" else "torch.softmax"
            return ["import torch"], [f"{out} = {fn}({a}, dim={self._val(iv, 'dim')})"]
        if op == "einsum":
            return ["import torch"], [
                f"{out} = torch.einsum({self._val(iv, 'equation')}, {a}, {self._val(iv, 'b')})"
            ]
        if op == "mux":
            sel = self._val(iv, "select")
            return [], [
                f"{out} = ({a}) if str({sel}).strip().lower() in ('a','0','','left','first') else ({self._val(iv, 'b')})"
            ]
        return [], [f"# unknown tensor op {op!r}"]
