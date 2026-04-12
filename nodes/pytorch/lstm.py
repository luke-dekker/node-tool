"""LSTM node — build and run nn.LSTM in one node. Replaces lstm_layer + lstm_forward."""
from __future__ import annotations
from core.node import BaseNode, PortType


class LSTMNode(BaseNode):
    type_name   = "pt_lstm"
    label       = "LSTM"
    category    = "Layers"
    subcategory = "Recurrent"
    description = "nn.LSTM — build and run LSTM in one node. Caches the module across ticks."

    def __init__(self):
        self._layer = None
        self._layer_cfg: tuple | None = None
        super().__init__()

    def _setup_ports(self):
        self.add_input("input_size",    PortType.INT,    default=64)
        self.add_input("hidden_size",   PortType.INT,    default=128)
        self.add_input("num_layers",    PortType.INT,    default=1)
        self.add_input("dropout",       PortType.FLOAT,  default=0.0)
        self.add_input("bidirectional", PortType.BOOL,   default=False)
        self.add_input("batch_first",   PortType.BOOL,   default=True)
        self.add_input("freeze",        PortType.BOOL,   default=False)
        self.add_input("x",             PortType.TENSOR, default=None)
        self.add_input("h0",            PortType.TENSOR, default=None)
        self.add_input("c0",            PortType.TENSOR, default=None)
        self.add_output("output", PortType.TENSOR)
        self.add_output("hidden", PortType.TENSOR)
        self.add_output("cell",   PortType.TENSOR)
        self.add_output("module", PortType.MODULE)

    def execute(self, inputs):
        try:
            import torch.nn as nn
            cfg = (
                int(inputs.get("input_size") or 64),
                int(inputs.get("hidden_size") or 128),
                int(inputs.get("num_layers") or 1),
                float(inputs.get("dropout") or 0.0),
                bool(inputs.get("bidirectional", False)),
                bool(inputs.get("batch_first", True)),
                bool(inputs.get("freeze", False)),
            )
            if self._layer is None or self._layer_cfg != cfg:
                self._layer = nn.LSTM(
                    input_size=cfg[0], hidden_size=cfg[1], num_layers=cfg[2],
                    dropout=cfg[3], bidirectional=cfg[4], batch_first=cfg[5],
                )
                if cfg[6]:
                    for p in self._layer.parameters():
                        p.requires_grad = False
                self._layer_cfg = cfg

            x = inputs.get("x")
            if x is None:
                return {"output": None, "hidden": None, "cell": None, "module": self._layer}
            h0 = inputs.get("h0")
            c0 = inputs.get("c0")
            if h0 is not None and c0 is not None:
                out, (hn, cn) = self._layer(x, (h0, c0))
            else:
                out, (hn, cn) = self._layer(x)
            return {"output": out, "hidden": hn, "cell": cn, "module": self._layer}
        except Exception:
            return {"output": None, "hidden": None, "cell": None, "module": None}

    def export(self, iv, ov):
        lv = f"_lstm_{self.safe_id}"
        x = iv.get("x") or "None  # TODO: connect input tensor"
        h0 = iv.get("h0")
        c0 = iv.get("c0")
        lines = [
            f"{lv} = nn.LSTM(",
            f"    input_size={self._val(iv, 'input_size')},",
            f"    hidden_size={self._val(iv, 'hidden_size')},",
            f"    num_layers={self._val(iv, 'num_layers')},",
            f"    dropout={self._val(iv, 'dropout')},",
            f"    bidirectional={self._val(iv, 'bidirectional')},",
            f"    batch_first={self._val(iv, 'batch_first')},",
            f")",
        ]
        if self.inputs["freeze"].default_value:
            lines.append(f"for _p in {lv}.parameters(): _p.requires_grad = False")
        out_var = ov.get("output", "_out")
        hid_var = ov.get("hidden", "_hidden")
        cell_var = ov.get("cell", "_cell")
        if h0 and c0:
            call = f"{lv}({x}, ({h0}, {c0}))"
        else:
            call = f"{lv}({x})"
        lines.append(f"{out_var}, ({hid_var}, {cell_var}) = {call}")
        lines.append(f"{ov.get('module', '_lstm_module')} = {lv}")
        return ["import torch", "import torch.nn as nn"], lines
