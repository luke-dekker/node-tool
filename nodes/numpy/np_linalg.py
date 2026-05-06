"""Consolidated NumPy linear-algebra node.

Replaces five per-op nodes (inv, svd, eig, dot, matmul). Each kind reads
a different subset of inputs and writes a different subset of outputs:

    inv    — input: matrix          → result = inv(matrix)
    svd    — input: matrix          → result = U, secondary = S, tertiary = Vt
    eig    — input: matrix          → result = eigenvalues, secondary = eigenvectors
    dot    — input: a, b            → result = np.dot(a, b)
    matmul — input: a, b            → result = np.matmul(a, b)

Unused outputs are None — they pass through the graph unwired without
breaking downstream nodes that handle None.
"""
from __future__ import annotations
import numpy as np

from core.node import BaseNode, PortType


_KINDS = ["inv", "svd", "eig", "dot", "matmul"]


class NpLinalgNode(BaseNode):
    type_name   = "np_linalg"
    label       = "Linalg"
    category    = "NumPy"
    subcategory = ""
    description = (
        "Linear-algebra ops. Pick `kind`:\n"
        "  inv    — matrix → result\n"
        "  svd    — matrix → result=U, secondary=S, tertiary=Vt\n"
        "  eig    — matrix → result=eigenvalues, secondary=eigenvectors\n"
        "  dot    — a, b → result\n"
        "  matmul — a, b → result"
    )

    def _setup_ports(self):
        self.add_input("kind",   PortType.STRING, default="matmul", choices=_KINDS)
        self.add_input("matrix", PortType.NDARRAY,
                       description="Input for unary ops (inv / svd / eig)")
        self.add_input("a",      PortType.NDARRAY,
                       description="LHS for binary ops (dot / matmul)")
        self.add_input("b",      PortType.NDARRAY,
                       description="RHS for binary ops (dot / matmul)")
        self.add_output("result",    PortType.NDARRAY,
                        description="Primary output (inv result / U / eigenvalues / dot / matmul)")
        self.add_output("secondary", PortType.NDARRAY,
                        description="(svd) S, (eig) eigenvectors, else None")
        self.add_output("tertiary",  PortType.NDARRAY,
                        description="(svd) Vt, else None")

    def execute(self, inputs):
        kind = (inputs.get("kind") or "matmul").strip().lower()
        null = {"result": None, "secondary": None, "tertiary": None}
        try:
            if kind == "inv":
                m = inputs.get("matrix")
                return null if m is None else {**null, "result": np.linalg.inv(m)}
            if kind == "svd":
                m = inputs.get("matrix")
                if m is None:
                    return null
                U, S, Vt = np.linalg.svd(m, full_matrices=False)
                return {"result": U, "secondary": S, "tertiary": Vt}
            if kind == "eig":
                m = inputs.get("matrix")
                if m is None:
                    return null
                vals, vecs = np.linalg.eig(m)
                return {"result": vals, "secondary": vecs, "tertiary": None}
            if kind in ("dot", "matmul"):
                a, b = inputs.get("a"), inputs.get("b")
                if a is None or b is None:
                    return null
                fn = np.dot if kind == "dot" else np.matmul
                return {**null, "result": fn(a, b)}
            return null
        except Exception:
            return null

    def export(self, iv, ov):
        kind = (self.inputs["kind"].default_value or "matmul").strip().lower()
        imp  = ["import numpy as np"]
        out_r = ov.get("result", "_lin_result")
        out_s = ov.get("secondary", "_lin_secondary")
        out_t = ov.get("tertiary", "_lin_tertiary")
        if kind == "inv":
            return imp, [f"{out_r} = np.linalg.inv({self._val(iv, 'matrix')})"]
        if kind == "svd":
            return imp, [
                f"{out_r}, {out_s}, {out_t} = "
                f"np.linalg.svd({self._val(iv, 'matrix')}, full_matrices=False)"
            ]
        if kind == "eig":
            return imp, [
                f"{out_r}, {out_s} = np.linalg.eig({self._val(iv, 'matrix')})"
            ]
        if kind in ("dot", "matmul"):
            return imp, [
                f"{out_r} = np.{kind}({self._val(iv, 'a')}, {self._val(iv, 'b')})"
            ]
        return imp, [f"{out_r} = None"]
