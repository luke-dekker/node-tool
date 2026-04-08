"""RNN Forward node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class RNNForwardNode(BaseNode):
    type_name   = "pt_rnn_forward"
    label       = "RNN Forward"
    category    = "Layers"
    subcategory = "Recurrent"
    description = "Run an RNN or GRU module. Returns full output sequence and final hidden state. Input shape: (batch, seq_len, input_size) if batch_first=True."

    def _setup_ports(self):
        self.add_input("module", PortType.MODULE, default=None)
        self.add_input("x",      PortType.TENSOR, default=None)
        self.add_input("h0",     PortType.TENSOR, default=None)
        self.add_output("output", PortType.TENSOR)
        self.add_output("hidden", PortType.TENSOR)

    def execute(self, inputs):
        try:
            module = inputs.get("module")
            x      = inputs.get("x")
            h0     = inputs.get("h0")
            if module is None or x is None:
                return {"output": None, "hidden": None}
            out, hidden = module(x) if h0 is None else module(x, h0)
            return {"output": out, "hidden": hidden}
        except Exception:
            return {"output": None, "hidden": None}

    def export(self, iv, ov):
        module = iv.get("module") or "None  # TODO: connect an RNN/GRU module"
        x  = iv.get("x")  or "None  # TODO: connect input tensor"
        h0 = iv.get("h0")
        out_var = ov.get("output", "_out")
        hid_var = ov.get("hidden", "_hidden")
        call = f"{module}({x})" if not h0 else f"{module}({x}, {h0})"
        return ["import torch", "import torch.nn as nn"], [
            f"{out_var}, {hid_var} = {call}",
        ]
