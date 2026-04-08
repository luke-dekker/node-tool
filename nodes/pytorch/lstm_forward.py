"""LSTM Forward node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class LSTMForwardNode(BaseNode):
    type_name   = "pt_lstm_forward"
    label       = "LSTM Forward"
    category    = "Layers"
    subcategory = "Recurrent"
    description = "Run an LSTM module. Returns full output sequence, final hidden state h_n, and cell state c_n. Input shape: (batch, seq_len, input_size) if batch_first=True."

    def _setup_ports(self):
        self.add_input("module", PortType.MODULE, default=None)
        self.add_input("x",      PortType.TENSOR, default=None)
        self.add_input("h0",     PortType.TENSOR, default=None)
        self.add_input("c0",     PortType.TENSOR, default=None)
        self.add_output("output", PortType.TENSOR)
        self.add_output("hidden", PortType.TENSOR)
        self.add_output("cell",   PortType.TENSOR)

    def execute(self, inputs):
        try:
            module = inputs.get("module")
            x      = inputs.get("x")
            h0     = inputs.get("h0")
            c0     = inputs.get("c0")
            if module is None or x is None:
                return {"output": None, "hidden": None, "cell": None}
            if h0 is not None and c0 is not None:
                out, (hn, cn) = module(x, (h0, c0))
            else:
                out, (hn, cn) = module(x)
            return {"output": out, "hidden": hn, "cell": cn}
        except Exception:
            return {"output": None, "hidden": None, "cell": None}

    def export(self, iv, ov):
        module = iv.get("module") or "None  # TODO: connect an LSTM module"
        x  = iv.get("x")  or "None  # TODO: connect input tensor"
        h0 = iv.get("h0")
        c0 = iv.get("c0")
        out_var = ov.get("output", "_out")
        hid_var = ov.get("hidden", "_hidden")
        cell_var = ov.get("cell",  "_cell")
        if h0 and c0:
            call = f"{module}({x}, ({h0}, {c0}))"
        else:
            call = f"{module}({x})"
        return ["import torch", "import torch.nn as nn"], [
            f"{out_var}, ({hid_var}, {cell_var}) = {call}",
        ]
