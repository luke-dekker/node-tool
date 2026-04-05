"""Recurrent network nodes — RNN, GRU, LSTM layer creation and forward pass."""
from __future__ import annotations
from core.node import BaseNode
from core.node import PortType

CATEGORY = "Layers"


class RNNLayerNode(BaseNode):
    type_name = "pt_rnn_layer"
    label = "RNN Layer"
    category = CATEGORY
    description = "nn.RNN — vanilla recurrent network layer. Outputs an nn.Module."

    def _setup_ports(self):
        self.add_input("input_size",   PortType.INT,  default=64)
        self.add_input("hidden_size",  PortType.INT,  default=128)
        self.add_input("num_layers",   PortType.INT,  default=1)
        self.add_input("nonlinearity", PortType.STRING, default="tanh")
        self.add_input("dropout",      PortType.FLOAT, default=0.0)
        self.add_input("bidirectional",PortType.BOOL,  default=False)
        self.add_input("batch_first",  PortType.BOOL,  default=True)
        self.add_input("freeze",       PortType.BOOL,  default=False)
        self.add_output("module", PortType.MODULE)

    def execute(self, inputs):
        try:
            import torch.nn as nn
            module = nn.RNN(
                input_size=int(inputs.get("input_size") or 64),
                hidden_size=int(inputs.get("hidden_size") or 128),
                num_layers=int(inputs.get("num_layers") or 1),
                nonlinearity=str(inputs.get("nonlinearity") or "tanh"),
                dropout=float(inputs.get("dropout") or 0.0),
                bidirectional=bool(inputs.get("bidirectional", False)),
                batch_first=bool(inputs.get("batch_first", True)),
            )
            if bool(inputs.get("freeze", False)):
                for p in module.parameters():
                    p.requires_grad = False
            return {"module": module}
        except Exception:
            return {"module": None}


class GRULayerNode(BaseNode):
    type_name = "pt_gru_layer"
    label = "GRU Layer"
    category = CATEGORY
    description = "nn.GRU — Gated Recurrent Unit layer. Outputs an nn.Module."

    def _setup_ports(self):
        self.add_input("input_size",   PortType.INT,   default=64)
        self.add_input("hidden_size",  PortType.INT,   default=128)
        self.add_input("num_layers",   PortType.INT,   default=1)
        self.add_input("dropout",      PortType.FLOAT, default=0.0)
        self.add_input("bidirectional",PortType.BOOL,  default=False)
        self.add_input("batch_first",  PortType.BOOL,  default=True)
        self.add_input("freeze",       PortType.BOOL,  default=False)
        self.add_output("module", PortType.MODULE)

    def execute(self, inputs):
        try:
            import torch.nn as nn
            module = nn.GRU(
                input_size=int(inputs.get("input_size") or 64),
                hidden_size=int(inputs.get("hidden_size") or 128),
                num_layers=int(inputs.get("num_layers") or 1),
                dropout=float(inputs.get("dropout") or 0.0),
                bidirectional=bool(inputs.get("bidirectional", False)),
                batch_first=bool(inputs.get("batch_first", True)),
            )
            if bool(inputs.get("freeze", False)):
                for p in module.parameters():
                    p.requires_grad = False
            return {"module": module}
        except Exception:
            return {"module": None}


class LSTMLayerNode(BaseNode):
    type_name = "pt_lstm_layer"
    label = "LSTM Layer"
    category = CATEGORY
    description = "nn.LSTM — Long Short-Term Memory layer. Outputs an nn.Module."

    def _setup_ports(self):
        self.add_input("input_size",   PortType.INT,   default=64)
        self.add_input("hidden_size",  PortType.INT,   default=128)
        self.add_input("num_layers",   PortType.INT,   default=1)
        self.add_input("dropout",      PortType.FLOAT, default=0.0)
        self.add_input("bidirectional",PortType.BOOL,  default=False)
        self.add_input("batch_first",  PortType.BOOL,  default=True)
        self.add_input("freeze",       PortType.BOOL,  default=False)
        self.add_output("module", PortType.MODULE)

    def execute(self, inputs):
        try:
            import torch.nn as nn
            module = nn.LSTM(
                input_size=int(inputs.get("input_size") or 64),
                hidden_size=int(inputs.get("hidden_size") or 128),
                num_layers=int(inputs.get("num_layers") or 1),
                dropout=float(inputs.get("dropout") or 0.0),
                bidirectional=bool(inputs.get("bidirectional", False)),
                batch_first=bool(inputs.get("batch_first", True)),
            )
            if bool(inputs.get("freeze", False)):
                for p in module.parameters():
                    p.requires_grad = False
            return {"module": module}
        except Exception:
            return {"module": None}


class RNNForwardNode(BaseNode):
    type_name = "pt_rnn_forward"
    label = "RNN Forward"
    category = CATEGORY
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


class LSTMForwardNode(BaseNode):
    type_name = "pt_lstm_forward"
    label = "LSTM Forward"
    category = CATEGORY
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


class PackSequenceNode(BaseNode):
    type_name = "pt_pack_sequence"
    label = "Pack Sequence"
    category = CATEGORY
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


class UnpackSequenceNode(BaseNode):
    type_name = "pt_unpack_sequence"
    label = "Unpack Sequence"
    category = CATEGORY
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


# Subcategory stamp
_SC = "Recurrent"
from core.node import BaseNode as _BN
for _n, _c in list(globals().items()):
    if isinstance(_c, type) and issubclass(_c, _BN) and _c is not _BN:
        _c.subcategory = _SC
