"""Consolidated recurrent layer node — replaces RNNNode, LSTMNode, GRUNode.

Pick `kind`:
  rnn   — nn.RNN(nonlinearity=tanh|relu)
  lstm  — nn.LSTM (also outputs `cell` final state)
  gru   — nn.GRU

Outputs:
  output  — per-timestep hidden states from the last layer (B, T, H * dirs)
  hidden  — final hidden state for every layer (L * dirs, B, H)
  cell    — final cell state, lstm only (None for rnn/gru)
  module  — the underlying nn.Module (for save/load/freeze)

The cell output is always present in the port list but only populated for
kind=lstm — saves us declaring different outputs per kind.
"""
from __future__ import annotations
from core.node import BaseNode, PortType


_KINDS = ["rnn", "lstm", "gru"]


class RecurrentLayerNode(BaseNode):
    type_name   = "pt_recurrent"
    label       = "Recurrent Layer"
    category    = "Layers"
    subcategory = "Recurrent"
    description = (
        "RNN / LSTM / GRU layer in one node. Pick `kind`. Input feature dim "
        "is inferred from `input_seq.shape[-1]`. `cell` output is populated "
        "only for kind=lstm; the others leave it as None."
    )

    def __init__(self):
        self._layer = None
        self._layer_cfg: tuple | None = None
        super().__init__()

    def relevant_inputs(self, values):
        kind = (values.get("kind") or "rnn").strip()
        base = ["kind", "hidden_size", "num_layers", "dropout",
                "bidirectional", "batch_first", "freeze"]
        return base + (["nonlinearity"] if kind == "rnn" else [])

    def _setup_ports(self):
        self.add_input("input_seq",     PortType.TENSOR, default=None,
                       description="(batch, time, features) if batch_first else (time, batch, features)")
        self.add_input("init_hidden",   PortType.TENSOR, default=None, optional=True,
                       description="Optional h0; shape (L * dirs, B, hidden_size)")
        self.add_input("init_cell",     PortType.TENSOR, default=None, optional=True,
                       description="LSTM only: optional c0; same shape as init_hidden")
        self.add_input("kind",          PortType.STRING, default="lstm", choices=_KINDS)
        self.add_input("hidden_size",   PortType.INT,    default=128)
        self.add_input("num_layers",    PortType.INT,    default=1)
        self.add_input("nonlinearity",  PortType.STRING, default="tanh",
                       choices=["tanh", "relu"], optional=True,
                       description="rnn only")
        self.add_input("dropout",       PortType.FLOAT,  default=0.0)
        self.add_input("bidirectional", PortType.BOOL,   default=False)
        self.add_input("batch_first",   PortType.BOOL,   default=True)
        self.add_input("freeze",        PortType.BOOL,   default=False)
        self.add_output("output", PortType.TENSOR,
                        description="Per-timestep hidden states from last layer")
        self.add_output("hidden", PortType.TENSOR,
                        description="Final hidden state, all layers stacked dim 0")
        self.add_output("cell",   PortType.TENSOR,
                        description="LSTM final cell state (None for rnn/gru)")
        self.add_output("module", PortType.MODULE)

    def execute(self, inputs):
        try:
            import torch.nn as nn
            from nodes.pytorch._helpers import _infer_feature_dim
            x = inputs.get("input_seq")
            in_f = _infer_feature_dim(x, None, axis=-1)
            null = {"output": None, "hidden": None, "cell": None,
                    "module": self._layer}
            if in_f <= 0:
                return null

            kind = (inputs.get("kind") or "lstm").strip()
            cfg_common = (
                kind, in_f,
                int(inputs.get("hidden_size") or 128),
                int(inputs.get("num_layers") or 1),
                float(inputs.get("dropout") or 0.0),
                bool(inputs.get("bidirectional", False)),
                bool(inputs.get("batch_first", True)),
                bool(inputs.get("freeze", False)),
                str(inputs.get("nonlinearity") or "tanh") if kind == "rnn" else None,
            )
            if self._layer is None or self._layer_cfg != cfg_common:
                kw = dict(input_size=in_f, hidden_size=cfg_common[2],
                          num_layers=cfg_common[3], dropout=cfg_common[4],
                          bidirectional=cfg_common[5], batch_first=cfg_common[6])
                if kind == "lstm":
                    self._layer = nn.LSTM(**kw)
                elif kind == "gru":
                    self._layer = nn.GRU(**kw)
                else:  # rnn
                    self._layer = nn.RNN(nonlinearity=cfg_common[8], **kw)
                if cfg_common[7]:
                    for p in self._layer.parameters():
                        p.requires_grad = False
                self._layer_cfg = cfg_common

            if x is None:
                return null

            h0 = inputs.get("init_hidden")
            if kind == "lstm":
                c0 = inputs.get("init_cell")
                if h0 is not None and c0 is not None:
                    out, (hn, cn) = self._layer(x, (h0, c0))
                else:
                    out, (hn, cn) = self._layer(x)
                return {"output": out, "hidden": hn, "cell": cn, "module": self._layer}
            else:
                out, hn = self._layer(x) if h0 is None else self._layer(x, h0)
                return {"output": out, "hidden": hn, "cell": None, "module": self._layer}
        except Exception:
            return {"output": None, "hidden": None, "cell": None, "module": None}

    def export(self, iv, ov):
        kind = (self.inputs["kind"].default_value or "lstm")
        lv = f"_{kind}_{self.safe_id}"
        x = iv.get("input_seq") or "None  # TODO: connect input tensor"
        h0 = iv.get("init_hidden")
        cls = {"rnn": "RNN", "lstm": "LSTM", "gru": "GRU"}.get(kind, "LSTM")

        ctor_args = [
            f"    input_size={x}.shape[-1],",
            f"    hidden_size={self._val(iv, 'hidden_size')},",
            f"    num_layers={self._val(iv, 'num_layers')},",
            f"    dropout={self._val(iv, 'dropout')},",
            f"    bidirectional={self._val(iv, 'bidirectional')},",
            f"    batch_first={self._val(iv, 'batch_first')},",
        ]
        if kind == "rnn":
            ctor_args.insert(0, f"    nonlinearity={self._val(iv, 'nonlinearity')},")

        lines = [f"{lv} = nn.{cls}("] + ctor_args + [")"]
        if self.inputs["freeze"].default_value:
            lines.append(f"for _p in {lv}.parameters(): _p.requires_grad = False")

        out_var = ov.get("output", "_out"); hid_var = ov.get("hidden", "_hidden")
        if kind == "lstm":
            c0 = iv.get("init_cell")
            cell_var = ov.get("cell", "_cell")
            call = f"{lv}({x}, ({h0}, {c0}))" if (h0 and c0) else f"{lv}({x})"
            lines.append(f"{out_var}, ({hid_var}, {cell_var}) = {call}")
        else:
            call = f"{lv}({x}, {h0})" if h0 else f"{lv}({x})"
            lines.append(f"{out_var}, {hid_var} = {call}")
        lines.append(f"{ov.get('module', f'_{kind}_module')} = {lv}")
        return ["import torch", "import torch.nn as nn"], lines
