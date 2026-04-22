"""RNN node — build and run nn.RNN in one node. Replaces rnn_layer + rnn_forward."""
from __future__ import annotations
from core.node import BaseNode, PortType


class RNNNode(BaseNode):
    type_name   = "pt_rnn"
    label       = "RNN"
    category    = "Layers"
    subcategory = "Recurrent"
    description = "nn.RNN — build and run vanilla RNN in one node. Caches the module across ticks."

    def __init__(self):
        self._layer = None
        self._layer_cfg: tuple | None = None
        super().__init__()

    def _setup_ports(self):
        self.add_input("hidden_size",   PortType.INT,    default=128)
        self.add_input("num_layers",    PortType.INT,    default=1)
        self.add_input("nonlinearity",  PortType.STRING, default="tanh")
        self.add_input("dropout",       PortType.FLOAT,  default=0.0)
        self.add_input("bidirectional", PortType.BOOL,   default=False)
        self.add_input("batch_first",   PortType.BOOL,   default=True)
        self.add_input("freeze",        PortType.BOOL,   default=False)
        self.add_input("x",             PortType.TENSOR, default=None)
        self.add_input("h0",            PortType.TENSOR, default=None)
        # Legacy: input_size inferred from x.shape[-1].
        self.add_input("input_size",    PortType.INT,    default=0,
                       description="(legacy; ignored) — inferred from x")
        self.add_output("output", PortType.TENSOR)
        self.add_output("hidden", PortType.TENSOR)
        self.add_output("module", PortType.MODULE)

    def execute(self, inputs):
        try:
            import torch.nn as nn
            from nodes.pytorch._helpers import _infer_feature_dim
            x = inputs.get("x")
            in_f = _infer_feature_dim(x, inputs.get("input_size"), axis=-1)
            if in_f <= 0:
                return {"output": None, "hidden": None, "module": self._layer}
            cfg = (
                in_f,
                int(inputs.get("hidden_size") or 128),
                int(inputs.get("num_layers") or 1),
                str(inputs.get("nonlinearity") or "tanh"),
                float(inputs.get("dropout") or 0.0),
                bool(inputs.get("bidirectional", False)),
                bool(inputs.get("batch_first", True)),
                bool(inputs.get("freeze", False)),
            )
            if self._layer is None or self._layer_cfg != cfg:
                self._layer = nn.RNN(
                    input_size=cfg[0], hidden_size=cfg[1], num_layers=cfg[2],
                    nonlinearity=cfg[3], dropout=cfg[4],
                    bidirectional=cfg[5], batch_first=cfg[6],
                )
                if cfg[7]:
                    for p in self._layer.parameters():
                        p.requires_grad = False
                self._layer_cfg = cfg

            if x is None:
                return {"output": None, "hidden": None, "module": self._layer}
            h0 = inputs.get("h0")
            out, hidden = self._layer(x) if h0 is None else self._layer(x, h0)
            return {"output": out, "hidden": hidden, "module": self._layer}
        except Exception:
            return {"output": None, "hidden": None, "module": None}

    def export(self, iv, ov):
        lv = f"_rnn_{self.safe_id}"
        x = iv.get("x") or "None  # TODO: connect input tensor"
        h0 = iv.get("h0")
        lines = [
            f"{lv} = nn.RNN(",
            f"    input_size={x}.shape[-1],",
            f"    hidden_size={self._val(iv, 'hidden_size')},",
            f"    num_layers={self._val(iv, 'num_layers')},",
            f"    nonlinearity={self._val(iv, 'nonlinearity')},",
            f"    dropout={self._val(iv, 'dropout')},",
            f"    bidirectional={self._val(iv, 'bidirectional')},",
            f"    batch_first={self._val(iv, 'batch_first')},",
            f")",
        ]
        if self.inputs["freeze"].default_value:
            lines.append(f"for _p in {lv}.parameters(): _p.requires_grad = False")
        call = f"{lv}({x})" if not h0 else f"{lv}({x}, {h0})"
        lines.append(f"{ov.get('output', '_out')}, {ov.get('hidden', '_hidden')} = {call}")
        lines.append(f"{ov.get('module', '_rnn_module')} = {lv}")
        return ["import torch", "import torch.nn as nn"], lines
