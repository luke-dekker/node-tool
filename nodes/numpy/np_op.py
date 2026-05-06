"""Consolidated NumPy operations node — replaces 9 standalone nodes plus
the older NpReduceNode and NpArrayFuncNode mode-dispatchers.

Supported `op`:
  shape transforms (single array → array):
    clip, slice, reshape, transpose, flatten, normalize, cumsum, diff
  binary (a, b → array):
    concat, stack
  conditional (condition, x, y → array):
    where
  reductions (array, axis → array/scalar):
    sum, mean, std, var, min, max, median, prod, any, all
  element-wise (array → array):
    abs, sqrt, log, exp, sign
  inspections (array → string on `info`):
    shape, info

Outputs:
  result — np.ndarray (transforms / reductions / element-wise)
  info   — str         (shape / info ops)

NpCreateNode (creator, no array input) and NpLinalgNode (multi-output)
remain standalone.
"""
from __future__ import annotations
import numpy as np
from core.node import BaseNode, PortType


_TRANSFORM_OPS  = ["clip", "slice", "reshape", "transpose", "flatten",
                   "normalize", "cumsum", "diff"]
_BINARY_OPS     = ["concat", "stack"]
_WHERE_OPS      = ["where"]
_REDUCE_OPS     = ["sum", "mean", "std", "var", "min", "max", "median",
                   "prod", "any", "all"]
_ELEMENTWISE    = ["abs", "sqrt", "log", "exp", "sign"]
_INSPECT_OPS    = ["shape", "info"]
_OPS = (_TRANSFORM_OPS + _BINARY_OPS + _WHERE_OPS + _REDUCE_OPS
        + _ELEMENTWISE + _INSPECT_OPS)

_AXIS_SENTINEL = -99   # axis input value meaning "all"


class NpOpNode(BaseNode):
    type_name   = "np_op"
    label       = "NumPy Op"
    category    = "NumPy"
    subcategory = ""
    description = (
        "NumPy operation. Pick `op` from the dropdown — the inspector hides "
        "fields that don't apply. Transform/reduce/element-wise ops produce "
        "`result`; shape/info populate `info`."
    )

    def relevant_inputs(self, values):
        op = (values.get("op") or "abs").strip()
        base = ["op", "array"]   # array is a wired data port; surfaced for clarity
        per_op = {
            "clip":      ["a_min", "a_max"],
            "slice":     ["start", "end", "step"],
            "reshape":   ["shape"],
            "concat":    ["b", "axis"],
            "stack":     ["b", "axis"],
            "where":     ["x_arr", "y_arr"],
            **{r: ["axis"] for r in _REDUCE_OPS},
        }
        return ["op"] + per_op.get(op, [])

    def _setup_ports(self):
        self.add_input("array", PortType.NDARRAY)
        self.add_input("op",    PortType.STRING, default="abs", choices=_OPS)
        # union of per-op params
        self.add_input("b",     PortType.NDARRAY,           optional=True,
                       description="concat / stack: second array")
        self.add_input("x_arr", PortType.NDARRAY,           optional=True,
                       description="where: value where condition>0")
        self.add_input("y_arr", PortType.NDARRAY,           optional=True,
                       description="where: value otherwise")
        self.add_input("axis",  PortType.INT,    _AXIS_SENTINEL, optional=True,
                       description="reductions / concat / stack: axis (-99 = all)")
        self.add_input("a_min", PortType.FLOAT,  0.0, optional=True)
        self.add_input("a_max", PortType.FLOAT,  1.0, optional=True)
        self.add_input("start", PortType.INT,    0,   optional=True)
        self.add_input("end",   PortType.INT,    10,  optional=True)
        self.add_input("step",  PortType.INT,    1,   optional=True)
        self.add_input("shape", PortType.STRING, "2,3", optional=True)
        self.add_output("result", PortType.NDARRAY)
        self.add_output("info",   PortType.STRING)

    def execute(self, inputs):
        out = {"result": None, "info": ""}
        try:
            arr = inputs.get("array")
            if arr is None:
                return out
            op = (inputs.get("op") or "abs").strip().lower()

            if op == "clip":
                out["result"] = np.clip(arr, float(inputs.get("a_min", 0.0)),
                                              float(inputs.get("a_max", 1.0)))
                return out
            if op == "slice":
                s, e, st = int(inputs.get("start", 0)), int(inputs.get("end", 10)), int(inputs.get("step", 1))
                out["result"] = arr[s:e:st]
                return out
            if op == "reshape":
                shape = tuple(int(x) for x in (inputs.get("shape") or "2,3").split(",") if x.strip())
                out["result"] = arr.reshape(shape)
                return out
            if op == "transpose":
                out["result"] = np.transpose(arr); return out
            if op == "flatten":
                out["result"] = arr.flatten(); return out
            if op == "normalize":
                out["result"] = ((arr - arr.min()) / (arr.max() - arr.min())
                                 if arr.max() != arr.min()
                                 else np.zeros_like(arr, dtype=float))
                return out
            if op == "cumsum":
                out["result"] = np.cumsum(arr); return out
            if op == "diff":
                out["result"] = np.diff(arr); return out

            if op in _BINARY_OPS:
                b = inputs.get("b")
                if b is None:
                    return out
                ax = int(inputs.get("axis", 0) or 0)
                fn = np.concatenate if op == "concat" else np.stack
                out["result"] = fn([arr, b], axis=ax)
                return out

            if op == "where":
                x_arr = inputs.get("x_arr"); y_arr = inputs.get("y_arr")
                if x_arr is None or y_arr is None:
                    return out
                out["result"] = np.where(arr > 0, x_arr, y_arr)
                return out

            if op in _REDUCE_OPS:
                ax = inputs.get("axis", _AXIS_SENTINEL)
                ax = None if (ax is None or int(ax) == _AXIS_SENTINEL) else int(ax)
                out["result"] = getattr(np, op)(arr, axis=ax)
                return out

            if op in _ELEMENTWISE:
                if op == "log":
                    out["result"] = np.log(np.clip(arr, 1e-12, None))
                else:
                    out["result"] = getattr(np, op)(arr)
                return out

            if op == "shape":
                out["info"] = str(list(arr.shape)); return out
            if op == "info":
                a = np.asarray(arr, dtype=float)
                out["info"] = (f"shape={list(arr.shape)} dtype={arr.dtype} "
                               f"min={a.min():.4g} max={a.max():.4g} mean={a.mean():.4g}")
                return out

            return out
        except Exception:
            return out

    def export(self, iv, ov):
        op = (self.inputs["op"].default_value or "abs")
        arr = self._val(iv, "array")
        out = ov["result"]; info = ov["info"]
        imp = ["import numpy as np"]

        if op == "clip":
            return imp, [f"{out} = np.clip({arr}, {self._val(iv, 'a_min')}, {self._val(iv, 'a_max')})"]
        if op == "slice":
            return imp, [f"{out} = {arr}[{self._val(iv, 'start')}:{self._val(iv, 'end')}:{self._val(iv, 'step')}]"]
        if op == "reshape":
            shape = self._val(iv, "shape")
            return imp, [f"{out} = {arr}.reshape(tuple(int(x) for x in {shape}.split(',') if x.strip()))"]
        if op == "transpose":
            return imp, [f"{out} = np.transpose({arr})"]
        if op == "flatten":
            return imp, [f"{out} = {arr}.flatten()"]
        if op == "normalize":
            return imp, [
                f"{out} = ({arr} - {arr}.min()) / ({arr}.max() - {arr}.min()) "
                f"if {arr}.max() != {arr}.min() else np.zeros_like({arr}, dtype=float)"
            ]
        if op == "cumsum":  return imp, [f"{out} = np.cumsum({arr})"]
        if op == "diff":    return imp, [f"{out} = np.diff({arr})"]
        if op in _BINARY_OPS:
            b = self._val(iv, "b"); ax = self._val(iv, "axis")
            fn = "np.concatenate" if op == "concat" else "np.stack"
            return imp, [f"{out} = {fn}([{arr}, {b}], axis={ax})"]
        if op == "where":
            x = self._val(iv, "x_arr"); y = self._val(iv, "y_arr")
            return imp, [f"{out} = np.where({arr} > 0, {x}, {y})"]
        if op in _REDUCE_OPS:
            ax = self._axis(iv)
            return imp, [f"{out} = np.{op}({arr}, axis={ax})"]
        if op in _ELEMENTWISE:
            if op == "log":
                return imp, [f"{out} = np.log(np.clip({arr}, 1e-12, None))"]
            return imp, [f"{out} = np.{op}({arr})"]
        if op == "shape":
            return imp, [f"{info} = str(list({arr}.shape))"]
        if op == "info":
            return imp, [
                f"{info} = f\"shape={{list({arr}.shape)}} dtype={{{arr}.dtype}} "
                f"min={{{arr}.astype(float).min():.4g}} max={{{arr}.astype(float).max():.4g}} mean={{{arr}.astype(float).mean():.4g}}\""
            ]
        return imp, [f"# unknown op {op!r}"]
