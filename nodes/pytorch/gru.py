"""GRU node — build and run nn.GRU in one node. Replaces gru_layer + rnn_forward (for GRU)."""
from __future__ import annotations
from core.node import BaseNode, PortType


class GRUNode(BaseNode):
    type_name   = "pt_gru"
    label       = "GRU"
    category    = "Layers"
    subcategory = "Recurrent"
    description = "nn.GRU — build and run GRU in one node. Caches the module across ticks."

    def __init__(self):
        self._layer = None
        self._layer_cfg: tuple | None = None
        super().__init__()

    def _setup_ports(self):
        self.add_input("hidden_size",   PortType.INT,    default=128)
        self.add_input("num_layers",    PortType.INT,    default=1)
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
                float(inputs.get("dropout") or 0.0),
                bool(inputs.get("bidirectional", False)),
                bool(inputs.get("batch_first", True)),
                bool(inputs.get("freeze", False)),
            )
            if self._layer is None or self._layer_cfg != cfg:
                self._layer = nn.GRU(
                    input_size=cfg[0], hidden_size=cfg[1], num_layers=cfg[2],
                    dropout=cfg[3], bidirectional=cfg[4], batch_first=cfg[5],
                )
                if cfg[6]:
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
        lv = f"_gru_{self.safe_id}"
        x = iv.get("x") or "None  # TODO: connect input tensor"
        h0 = iv.get("h0")
        lines = [
            f"{lv} = nn.GRU(",
            f"    input_size={x}.shape[-1],",
            f"    hidden_size={self._val(iv, 'hidden_size')},",
            f"    num_layers={self._val(iv, 'num_layers')},",
            f"    dropout={self._val(iv, 'dropout')},",
            f"    bidirectional={self._val(iv, 'bidirectional')},",
            f"    batch_first={self._val(iv, 'batch_first')},",
            f")",
        ]
        if self.inputs["freeze"].default_value:
            lines.append(f"for _p in {lv}.parameters(): _p.requires_grad = False")
        call = f"{lv}({x})" if not h0 else f"{lv}({x}, {h0})"
        lines.append(f"{ov.get('output', '_out')}, {ov.get('hidden', '_hidden')} = {call}")
        lines.append(f"{ov.get('module', '_gru_module')} = {lv}")
        return ["import torch", "import torch.nn as nn"], lines
