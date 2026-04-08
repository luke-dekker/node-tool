"""Unpack Sequence node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class UnpackSequenceNode(BaseNode):
    type_name   = "pt_unpack_sequence"
    label       = "Unpack Sequence"
    category    = "Layers"
    subcategory = "Recurrent"
    description = "torch.nn.utils.rnn.pad_packed_sequence — inverse of Pack Sequence."

    def _setup_ports(self):
        self.add_input("packed",      PortType.ANY,  default=None)
        self.add_input("batch_first", PortType.BOOL, default=True)
        self.add_output("tensor",  PortType.TENSOR)
        self.add_output("lengths", PortType.TENSOR)

    def execute(self, inputs):
        try:
            from torch.nn.utils.rnn import pad_packed_sequence
            packed = inputs.get("packed")
            if packed is None:
                return {"tensor": None, "lengths": None}
            tensor, lengths = pad_packed_sequence(packed,
                                batch_first=bool(inputs.get("batch_first", True)))
            return {"tensor": tensor, "lengths": lengths}
        except Exception:
            return {"tensor": None, "lengths": None}

    def export(self, iv, ov):
        packed = iv.get("packed") or "None  # TODO: connect a packed sequence"
        t_var = ov.get("tensor",  "_tensor")
        l_var = ov.get("lengths", "_lengths")
        return ["from torch.nn.utils.rnn import pad_packed_sequence"], [
            f"{t_var}, {l_var} = pad_packed_sequence("
            f"{packed}, batch_first={self._val(iv, 'batch_first')})",
        ]
