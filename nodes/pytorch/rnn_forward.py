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
