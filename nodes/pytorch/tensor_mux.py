"""Tensor Mux node — selects between two tensor inputs at runtime.

Used by dual-mode graphs (e.g. live imitation) where the same graph is
evaluated against either a buffered source or a live source depending on
a mode flag.
"""
from __future__ import annotations
from core.node import BaseNode, PortType


class TensorMuxNode(BaseNode):
    type_name   = "pt_tensor_mux"
    label       = "Tensor Mux"
    category    = "Tensor Ops"
    subcategory = ""
    description = "Select between two tensor inputs by a runtime key (0/'a' → a, 1/'b' → b)"

    def _setup_ports(self):
        self.add_input("a",      PortType.TENSOR, default=None)
        self.add_input("b",      PortType.TENSOR, default=None)
        self.add_input("select", PortType.STRING, default="a",
                       description="'a'/'b' or 0/1 — chooses which input to forward")
        self.add_output("tensor_out", PortType.TENSOR)

    @staticmethod
    def _pick_a(select) -> bool:
        if isinstance(select, bool):
            return not select
        if isinstance(select, (int, float)):
            return int(select) == 0
        s = str(select).strip().lower()
        return s in ("a", "0", "", "left", "first")

    def execute(self, inputs):
        a = inputs.get("a")
        b = inputs.get("b")
        select = inputs.get("select", "a")
        chosen = a if self._pick_a(select) else b
        if chosen is None:
            chosen = a if a is not None else b
        return {"tensor_out": chosen}

    def export(self, iv, ov):
        sel = self._val(iv, "select")
        a   = self._val(iv, "a")
        b   = self._val(iv, "b")
        out = ov["tensor_out"]
        return [], [
            f"{out} = ({a}) if str({sel}).strip().lower() in ('a','0','','left','first') else ({b})",
        ]
