"""Consolidated tensor reshape/combine node — replaces TensorCatNode,
TensorStackNode, TensorSplitNode, TensorShapeOpNode.

Pick `op`:
  cat        — torch.cat([t1..t4], dim) → tensor
  stack      — torch.stack([t1..t4], dim) → tensor
  split      — torch.split(t, split_size, dim) → chunk_0, chunk_1
  reshape    — t.reshape(shape) → tensor
  squeeze    — t.squeeze(dim) (dim<0 = all) → tensor
  unsqueeze  — t.unsqueeze(dim) → tensor
  permute    — t.permute(*shape ints) → tensor
  transpose  — t.transpose(dim, dim_b) → tensor

Outputs:
  tensor   — populated for cat/stack/reshape/squeeze/unsqueeze/permute/transpose
  chunk_0  — populated for split
  chunk_1  — populated for split (None if split yields only 1 chunk)
"""
from __future__ import annotations
from core.node import BaseNode, PortType


_MULTI_IN = ["cat", "stack"]
_SHAPE    = ["reshape", "squeeze", "unsqueeze", "permute", "transpose"]
_OPS      = _MULTI_IN + ["split"] + _SHAPE


class TensorReshapeNode(BaseNode):
    type_name   = "pt_tensor_reshape"
    label       = "Tensor Reshape"
    category    = "Tensor Ops"
    subcategory = ""
    description = (
        "Combine or reshape tensors. Pick `op`:\n"
        "  cat / stack — combine up to 4 tensors along `dim`\n"
        "  split — split single tensor by `split_size` along `dim` (2 outputs)\n"
        "  reshape / squeeze / unsqueeze / permute / transpose — single-tensor shape ops"
    )

    def relevant_inputs(self, values):
        op = (values.get("op") or "reshape").strip()
        if op in _MULTI_IN:    return ["op", "dim"]   # t1..t4 wired
        if op == "split":      return ["op", "split_size", "dim"]
        if op == "reshape":    return ["op", "shape"]
        if op == "squeeze":    return ["op", "dim"]
        if op == "unsqueeze":  return ["op", "dim"]
        if op == "permute":    return ["op", "shape"]
        if op == "transpose":  return ["op", "dim", "dim_b"]
        return ["op"]

    def _setup_ports(self):
        # multi-input (cat / stack) — t1 also serves as the single-tensor source
        # for the reshape/split modes
        self.add_input("t1", PortType.TENSOR, default=None)
        self.add_input("t2", PortType.TENSOR, default=None, optional=True)
        self.add_input("t3", PortType.TENSOR, default=None, optional=True)
        self.add_input("t4", PortType.TENSOR, default=None, optional=True)
        self.add_input("op",         PortType.STRING, default="reshape", choices=_OPS)
        self.add_input("dim",        PortType.INT,    default=0, optional=True)
        self.add_input("dim_b",      PortType.INT,    default=1, optional=True)
        self.add_input("shape",      PortType.STRING, default="-1", optional=True)
        self.add_input("split_size", PortType.INT,    default=1, optional=True)
        self.add_output("tensor",  PortType.TENSOR)
        self.add_output("chunk_0", PortType.TENSOR)
        self.add_output("chunk_1", PortType.TENSOR)

    def execute(self, inputs):
        try:
            import torch
            op = (inputs.get("op") or "reshape").strip()
            t  = inputs.get("t1")
            null = {"tensor": None, "chunk_0": None, "chunk_1": None}

            if op in _MULTI_IN:
                ts = [inputs.get(f"t{i}") for i in range(1, 5)
                      if inputs.get(f"t{i}") is not None]
                if not ts:
                    return null
                dim = int(inputs.get("dim") or 0)
                fn = torch.cat if op == "cat" else torch.stack
                if op == "stack" and len(ts) < 2:
                    return null
                return {**null, "tensor": fn(ts, dim=dim)}

            if op == "split":
                if t is None:
                    return null
                chunks = torch.split(t, int(inputs.get("split_size") or 1),
                                     dim=int(inputs.get("dim") or 0))
                return {
                    "tensor":  None,
                    "chunk_0": chunks[0] if len(chunks) > 0 else None,
                    "chunk_1": chunks[1] if len(chunks) > 1 else None,
                }

            if t is None:
                return null

            if op == "reshape":
                shape = [int(x.strip()) for x in str(inputs.get("shape") or "-1").split(",")]
                return {**null, "tensor": t.reshape(shape)}
            if op == "squeeze":
                dim = int(inputs.get("dim") if inputs.get("dim") is not None else -1)
                return {**null, "tensor": t.squeeze(dim) if dim >= 0 else t.squeeze()}
            if op == "unsqueeze":
                return {**null, "tensor": t.unsqueeze(int(inputs.get("dim") or 0))}
            if op == "permute":
                dims = [int(x.strip()) for x in str(inputs.get("shape") or "0").split(",")]
                return {**null, "tensor": t.permute(dims)}
            if op == "transpose":
                return {**null, "tensor": t.transpose(int(inputs.get("dim") or 0),
                                                     int(inputs.get("dim_b") or 1))}
            return null
        except Exception:
            return {"tensor": None, "chunk_0": None, "chunk_1": None}

    def export(self, iv, ov):
        op = (self.inputs["op"].default_value or "reshape")
        out = ov.get("tensor", "_t_out")
        imp = ["import torch"]

        if op in _MULTI_IN:
            ts = [iv.get(f"t{i}") for i in range(1, 5) if iv.get(f"t{i}")]
            if op == "stack" and len(ts) < 2:
                return imp, [f"{out} = None  # stack needs >=2 input tensors"]
            if not ts:
                return imp, [f"{out} = None"]
            fn = "torch.cat" if op == "cat" else "torch.stack"
            return imp, [f"{out} = {fn}([{', '.join(ts)}], dim={self._val(iv, 'dim')})"]

        t = iv.get("t1") or "None"

        if op == "split":
            var = f"_split_{self.safe_id}"
            lines = [f"{var} = torch.split({t}, {self._val(iv, 'split_size')}, "
                     f"dim={self._val(iv, 'dim')})"]
            if "chunk_0" in ov: lines.append(f"{ov['chunk_0']} = {var}[0] if len({var}) > 0 else None")
            if "chunk_1" in ov: lines.append(f"{ov['chunk_1']} = {var}[1] if len({var}) > 1 else None")
            return imp, lines

        if op == "reshape":
            shape_str = str((self.inputs["shape"].default_value or "-1"))
            shape_args = ", ".join(s.strip() for s in shape_str.split(","))
            return imp, [f"{out} = {t}.reshape({shape_args})"]
        if op == "squeeze":
            d = self._val(iv, "dim")
            return imp, [f"{out} = {t}.squeeze({d}) if {d} >= 0 else {t}.squeeze()"]
        if op == "unsqueeze":
            return imp, [f"{out} = {t}.unsqueeze({self._val(iv, 'dim')})"]
        if op == "permute":
            shape_str = str((self.inputs["shape"].default_value or "0"))
            dims_args = ", ".join(s.strip() for s in shape_str.split(","))
            return imp, [f"{out} = {t}.permute({dims_args})"]
        if op == "transpose":
            return imp, [f"{out} = {t}.transpose({self._val(iv, 'dim')}, {self._val(iv, 'dim_b')})"]
        return imp, [f"# unknown reshape op {op!r}"]
