"""Pack Sequence node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class PackSequenceNode(BaseNode):
    type_name   = "pt_pack_sequence"
    label       = "Pack Sequence"
    category    = "Layers"
    subcategory = "Recurrent"
    description = "torch.nn.utils.rnn.pack_padded_sequence for variable-length sequences. lengths: 1D tensor of sequence lengths."

    def _setup_ports(self):
        self.add_input("tensor",      PortType.TENSOR, default=None)
        self.add_input("lengths",     PortType.TENSOR, default=None)
        self.add_input("batch_first", PortType.BOOL,   default=True)
        self.add_output("packed", PortType.ANY)

    def execute(self, inputs):
        try:
            from torch.nn.utils.rnn import pack_padded_sequence
            t = inputs.get("tensor")
            l = inputs.get("lengths")
            if t is None or l is None:
                return {"packed": None}
            return {"packed": pack_padded_sequence(t, l.cpu(),
                               batch_first=bool(inputs.get("batch_first", True)),
                               enforce_sorted=False)}
        except Exception:
            return {"packed": None}

    def export(self, iv, ov):
        t = iv.get("tensor") or "None  # TODO: connect padded tensor"
        l = iv.get("lengths") or "None  # TODO: connect lengths tensor"
        return ["from torch.nn.utils.rnn import pack_padded_sequence"], [
            f"{ov['packed']} = pack_padded_sequence(",
            f"    {t}, {l}.cpu(),",
            f"    batch_first={self._val(iv, 'batch_first')},",
            f"    enforce_sorted=False,",
            f")",
        ]
