"""Tensor operation nodes — reshape, combine, split, einsum."""
from __future__ import annotations
from core.node import BaseNode
from core.node import PortType

CATEGORY = "Analyze"


class TensorCatNode(BaseNode):
    type_name = "pt_tensor_cat"
    label = "Concatenate"
    category = CATEGORY
    description = "torch.cat([t1, t2, t3, t4], dim). Concatenates up to 4 tensors along dim."

    def _setup_ports(self):
        self.add_input("t1",  PortType.TENSOR, default=None)
        self.add_input("t2",  PortType.TENSOR, default=None)
        self.add_input("t3",  PortType.TENSOR, default=None)
        self.add_input("t4",  PortType.TENSOR, default=None)
        self.add_input("dim", PortType.INT,    default=0)
        self.add_output("tensor", PortType.TENSOR)

    def execute(self, inputs):
        try:
            import torch
            ts = [inputs.get(f"t{i}") for i in range(1, 5) if inputs.get(f"t{i}") is not None]
            if not ts:
                return {"tensor": None}
            return {"tensor": torch.cat(ts, dim=int(inputs.get("dim") or 0))}
        except Exception:
            return {"tensor": None}


class TensorStackNode(BaseNode):
    type_name = "pt_tensor_stack"
    label = "Stack"
    category = CATEGORY
    description = "torch.stack([t1, t2, t3, t4], dim). Stacks tensors along a NEW dimension."

    def _setup_ports(self):
        self.add_input("t1",  PortType.TENSOR, default=None)
        self.add_input("t2",  PortType.TENSOR, default=None)
        self.add_input("t3",  PortType.TENSOR, default=None)
        self.add_input("t4",  PortType.TENSOR, default=None)
        self.add_input("dim", PortType.INT,    default=0)
        self.add_output("tensor", PortType.TENSOR)

    def execute(self, inputs):
        try:
            import torch
            ts = [inputs.get(f"t{i}") for i in range(1, 5) if inputs.get(f"t{i}") is not None]
            if len(ts) < 2:
                return {"tensor": None}
            return {"tensor": torch.stack(ts, dim=int(inputs.get("dim") or 0))}
        except Exception:
            return {"tensor": None}


class TensorSplitNode(BaseNode):
    type_name = "pt_tensor_split"
    label = "Split"
    category = CATEGORY
    description = "torch.split(tensor, split_size, dim). Returns first two chunks (chunk_0, chunk_1)."

    def _setup_ports(self):
        self.add_input("tensor",     PortType.TENSOR, default=None)
        self.add_input("split_size", PortType.INT,    default=1)
        self.add_input("dim",        PortType.INT,    default=0)
        self.add_output("chunk_0", PortType.TENSOR)
        self.add_output("chunk_1", PortType.TENSOR)

    def execute(self, inputs):
        try:
            import torch
            t = inputs.get("tensor")
            if t is None:
                return {"chunk_0": None, "chunk_1": None}
            chunks = torch.split(t, int(inputs.get("split_size") or 1),
                                 dim=int(inputs.get("dim") or 0))
            return {
                "chunk_0": chunks[0] if len(chunks) > 0 else None,
                "chunk_1": chunks[1] if len(chunks) > 1 else None,
            }
        except Exception:
            return {"chunk_0": None, "chunk_1": None}


class TensorReshapeNode(BaseNode):
    type_name = "pt_tensor_reshape"
    label = "Reshape"
    category = CATEGORY
    description = "tensor.reshape(shape). Enter shape as comma-separated ints, e.g. '32,-1' or '2,3,4'. Use -1 for inferred dim."

    def _setup_ports(self):
        self.add_input("tensor", PortType.TENSOR, default=None)
        self.add_input("shape",  PortType.STRING, default="-1")
        self.add_output("tensor", PortType.TENSOR)

    def execute(self, inputs):
        try:
            t = inputs.get("tensor")
            if t is None:
                return {"tensor": None}
            shape = [int(x.strip()) for x in str(inputs.get("shape") or "-1").split(",")]
            return {"tensor": t.reshape(shape)}
        except Exception:
            return {"tensor": None}


class TensorUnsqueezeNode(BaseNode):
    type_name = "pt_tensor_unsqueeze"
    label = "Unsqueeze"
    category = CATEGORY
    description = "tensor.unsqueeze(dim). Inserts a size-1 dimension at position dim."

    def _setup_ports(self):
        self.add_input("tensor", PortType.TENSOR, default=None)
        self.add_input("dim",    PortType.INT,    default=0)
        self.add_output("tensor", PortType.TENSOR)

    def execute(self, inputs):
        try:
            t = inputs.get("tensor")
            if t is None:
                return {"tensor": None}
            return {"tensor": t.unsqueeze(int(inputs.get("dim") or 0))}
        except Exception:
            return {"tensor": None}


class TensorSqueezeNode(BaseNode):
    type_name = "pt_tensor_squeeze"
    label = "Squeeze"
    category = CATEGORY
    description = "tensor.squeeze(dim). Removes size-1 dimensions. Leave dim=-1 to squeeze all."

    def _setup_ports(self):
        self.add_input("tensor", PortType.TENSOR, default=None)
        self.add_input("dim",    PortType.INT,    default=-1)
        self.add_output("tensor", PortType.TENSOR)

    def execute(self, inputs):
        try:
            t = inputs.get("tensor")
            if t is None:
                return {"tensor": None}
            dim = int(inputs.get("dim") if inputs.get("dim") is not None else -1)
            return {"tensor": t.squeeze(dim) if dim >= 0 else t.squeeze()}
        except Exception:
            return {"tensor": None}


class TensorTransposeNode(BaseNode):
    type_name = "pt_tensor_transpose"
    label = "Transpose"
    category = CATEGORY
    description = "tensor.transpose(dim0, dim1). Swap two dimensions."

    def _setup_ports(self):
        self.add_input("tensor", PortType.TENSOR, default=None)
        self.add_input("dim0",   PortType.INT,    default=0)
        self.add_input("dim1",   PortType.INT,    default=1)
        self.add_output("tensor", PortType.TENSOR)

    def execute(self, inputs):
        try:
            t = inputs.get("tensor")
            if t is None:
                return {"tensor": None}
            return {"tensor": t.transpose(int(inputs.get("dim0") or 0),
                                          int(inputs.get("dim1") or 1))}
        except Exception:
            return {"tensor": None}


class TensorPermuteNode(BaseNode):
    type_name = "pt_tensor_permute"
    label = "Permute"
    category = CATEGORY
    description = "tensor.permute(dims). Reorder all dimensions. Enter as comma-separated ints, e.g. '0,2,1'."

    def _setup_ports(self):
        self.add_input("tensor", PortType.TENSOR, default=None)
        self.add_input("dims",   PortType.STRING, default="0,1,2")
        self.add_output("tensor", PortType.TENSOR)

    def execute(self, inputs):
        try:
            t = inputs.get("tensor")
            if t is None:
                return {"tensor": None}
            dims = [int(x.strip()) for x in str(inputs.get("dims") or "0,1,2").split(",")]
            return {"tensor": t.permute(dims)}
        except Exception:
            return {"tensor": None}


class TensorEinsumNode(BaseNode):
    type_name = "pt_tensor_einsum"
    label = "Einsum"
    category = CATEGORY
    description = "torch.einsum(equation, t1, t2). E.g. 'ij,jk->ik' for matmul, 'bij,bjk->bik' for batched."

    def _setup_ports(self):
        self.add_input("equation", PortType.STRING, default="ij,jk->ik")
        self.add_input("t1",       PortType.TENSOR, default=None)
        self.add_input("t2",       PortType.TENSOR, default=None)
        self.add_output("tensor", PortType.TENSOR)

    def execute(self, inputs):
        try:
            import torch
            eq = str(inputs.get("equation") or "ij,jk->ik")
            t1 = inputs.get("t1")
            t2 = inputs.get("t2")
            if t1 is None or t2 is None:
                return {"tensor": None}
            return {"tensor": torch.einsum(eq, t1, t2)}
        except Exception:
            return {"tensor": None}


# Subcategory stamp
_SC = "Tensors"
from core.node import BaseNode as _BN
for _n, _c in list(globals().items()):
    if isinstance(_c, type) and issubclass(_c, _BN) and _c is not _BN:
        _c.subcategory = _SC
