"""RNN Layer node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class RNNLayerNode(BaseNode):
    type_name   = "pt_rnn_layer"
    label       = "RNN Layer"
    category    = "Layers"
    subcategory = "Recurrent"
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
