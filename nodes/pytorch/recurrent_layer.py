"""Consolidated recurrent layer node — RNN / LSTM / GRU + research variants.

Pick `kind`:
  rnn        — nn.RNN(nonlinearity=tanh|relu)
  lstm       — nn.LSTM (also outputs `cell` final state)
  gru        — nn.GRU
  clockwork  — Koutník 2014: hidden split into K modules ticking at rates
               1, 2, 4, …, 2^(K-1). Slower modules feed faster ones via
               an upper-triangular block-mask on the recurrent matrix, so
               long-range info is retained in slow modules without
               vanishing through dense recurrence.
  indrnn     — Li 2018: diagonal recurrent matrix (each hidden unit only
               sees its own previous value). No vanishing/exploding via
               cross-unit interaction → trains very deep stacks (20+
               layers) where vanilla RNN/LSTM stalls past 3.

Outputs:
  output  — per-timestep hidden states from the last layer (B, T, H * dirs)
  hidden  — final hidden state for every layer (L * dirs, B, H)
  cell    — final cell state, lstm only (None for the others)
  module  — the underlying nn.Module (for save/load/freeze)

Bidirectional / multi-layer / dropout: full support for rnn/lstm/gru.
Clockwork is single-layer, unidirectional (the rate hierarchy already
gives multi-scale recurrence; stacking + bidir is rarely used). IndRNN
supports num_layers and dropout-between-layers but not bidirectional.
"""
from __future__ import annotations
import inspect
import math
from core.node import BaseNode, PortType


_KINDS = ["rnn", "lstm", "gru", "clockwork", "indrnn"]


# ── Custom recurrence modules ────────────────────────────────────────────────
# Wrapped in builder functions so torch is imported lazily (the node module
# itself must remain importable without torch). Built once at import time
# below so `ClockworkRNN`/`IndRNN` are real nn.Module classes thereafter.


def _build_clockwork_class():
    import torch
    import torch.nn as nn

    class ClockworkRNN(nn.Module):
        def __init__(self, input_size, hidden_size,
                     num_modules=8, batch_first=True):
            super().__init__()
            if hidden_size % num_modules != 0:
                raise ValueError(
                    f"hidden_size ({hidden_size}) must be divisible by "
                    f"num_modules ({num_modules})"
                )
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_modules = num_modules
            self.module_size = hidden_size // num_modules
            self.batch_first = batch_first
            # Periods: [1, 2, 4, …, 2^(K-1)]
            periods = torch.tensor([2 ** k for k in range(num_modules)],
                                   dtype=torch.long)
            self.register_buffer("periods", periods, persistent=False)
            # Block-upper-triangular mask on the recurrent weight: row-block
            # i (target group) zeroed for column-blocks j < i (faster groups).
            mask = torch.zeros(hidden_size, hidden_size)
            for i in range(num_modules):
                for j in range(i, num_modules):
                    mask[i * self.module_size:(i + 1) * self.module_size,
                         j * self.module_size:(j + 1) * self.module_size] = 1.0
            self.register_buffer("rec_mask", mask, persistent=False)
            self.W_in = nn.Linear(input_size, hidden_size)
            self.W_h  = nn.Linear(hidden_size, hidden_size)
            # Init scaled by group rank — slow modules get smaller variance
            # so they don't dominate the recurrence early. Standard trick.
            with torch.no_grad():
                bound = 1.0 / math.sqrt(hidden_size)
                self.W_in.weight.uniform_(-bound, bound)
                self.W_h.weight.uniform_(-bound, bound)

        def forward(self, x, h0=None):
            if not self.batch_first:
                x = x.transpose(0, 1)
            B, T, _ = x.shape
            h = h0.squeeze(0) if h0 is not None else x.new_zeros(B, self.hidden_size)
            W_h_masked = self.W_h.weight * self.rec_mask
            outputs = []
            for t in range(T):
                inp_proj = self.W_in(x[:, t])
                rec      = h @ W_h_masked.t() + self.W_h.bias
                h_new    = torch.tanh(inp_proj + rec)
                # Active modules update; others keep h. Boolean mask
                # broadcast over the module_size unit groups.
                active = (t % self.periods == 0).repeat_interleave(self.module_size)
                gate = active.to(h.dtype).unsqueeze(0)
                h = h_new * gate + h * (1 - gate)
                outputs.append(h)
            out = torch.stack(outputs, dim=1)
            if not self.batch_first:
                out = out.transpose(0, 1)
            return out, h.unsqueeze(0)

    return ClockworkRNN


def _build_indrnn_class():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class IndRNN(nn.Module):
        """Independently Recurrent NN (Li et al. 2018).

        Each hidden unit has a single scalar recurrent weight (diagonal
        U). Stacking multiple layers with ReLU activations + diagonal
        recurrence trains stably to depths where LSTM stalls.
        """
        def __init__(self, input_size, hidden_size,
                     num_layers=1, dropout=0.0, batch_first=True):
            super().__init__()
            self.batch_first = batch_first
            self.num_layers  = num_layers
            self.dropout     = dropout
            self.hidden_size = hidden_size
            self.W = nn.ModuleList([
                nn.Linear(input_size if i == 0 else hidden_size, hidden_size)
                for i in range(num_layers)
            ])
            # Diagonal recurrence weights, init in [-1, 1] (Li 2018 §3.1).
            self.U = nn.ParameterList([
                nn.Parameter(torch.empty(hidden_size).uniform_(-1.0, 1.0))
                for _ in range(num_layers)
            ])

        def forward(self, x, h0=None):
            if not self.batch_first:
                x = x.transpose(0, 1)
            B, T, _ = x.shape
            if h0 is None:
                h = [x.new_zeros(B, self.hidden_size) for _ in range(self.num_layers)]
            else:
                h = [h0[i] for i in range(self.num_layers)]
            out_all = []
            for t in range(T):
                inp = x[:, t]
                for i in range(self.num_layers):
                    h[i] = F.relu(self.W[i](inp) + self.U[i] * h[i])
                    inp = h[i]
                    if self.dropout > 0 and i < self.num_layers - 1 and self.training:
                        inp = F.dropout(inp, p=self.dropout, training=True)
                out_all.append(inp)
            out = torch.stack(out_all, dim=1)
            if not self.batch_first:
                out = out.transpose(0, 1)
            h_final = torch.stack(h, dim=0)
            return out, h_final

    return IndRNN


# Build once at import time so we get clean nn.Module classes (not factories).
ClockworkRNN = _build_clockwork_class()
IndRNN       = _build_indrnn_class()


# ── Node ─────────────────────────────────────────────────────────────────────


class RecurrentLayerNode(BaseNode):
    type_name   = "pt_recurrent"
    label       = "Recurrent Layer"
    category    = "Layers"
    subcategory = "Recurrent"
    description = (
        "RNN / LSTM / GRU / Clockwork / IndRNN in one node. Pick `kind`. "
        "Input feature dim is inferred from `input_seq.shape[-1]`. `cell` "
        "output is populated only for kind=lstm; the others leave it None.\n"
        "  • clockwork: hidden split into `num_modules` groups ticking at\n"
        "    rates 1, 2, 4, …; hidden_size must be divisible by num_modules.\n"
        "    Single-layer, unidirectional only.\n"
        "  • indrnn: diagonal recurrence + ReLU; supports num_layers and\n"
        "    dropout between layers, no bidirectional."
    )

    def __init__(self):
        self._layer = None
        self._layer_cfg: tuple | None = None
        super().__init__()

    def relevant_inputs(self, values):
        kind = (values.get("kind") or "rnn").strip()
        base = ["kind", "hidden_size", "freeze"]
        if kind == "clockwork":
            return base + ["num_modules", "batch_first"]
        if kind == "indrnn":
            return base + ["num_layers", "dropout", "batch_first"]
        # rnn / lstm / gru
        common = base + ["num_layers", "dropout", "bidirectional", "batch_first"]
        return common + (["nonlinearity"] if kind == "rnn" else [])

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
        self.add_input("num_modules",   PortType.INT,    default=8, optional=True,
                       description="clockwork only — number of rate-modules; "
                                   "hidden_size must be divisible by this.")
        self.add_output("output", PortType.TENSOR,
                        description="Per-timestep hidden states from last layer")
        self.add_output("hidden", PortType.TENSOR,
                        description="Final hidden state, all layers stacked dim 0")
        self.add_output("cell",   PortType.TENSOR,
                        description="LSTM final cell state (None for others)")
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
            hidden = int(inputs.get("hidden_size") or 128)
            batch_first = bool(inputs.get("batch_first", True))
            freeze = bool(inputs.get("freeze", False))

            # Cfg tuple varies by kind — only fields that affect construction.
            if kind == "clockwork":
                num_modules = int(inputs.get("num_modules") or 8)
                cfg = (kind, in_f, hidden, num_modules, batch_first, freeze)
            elif kind == "indrnn":
                num_layers = int(inputs.get("num_layers") or 1)
                dropout    = float(inputs.get("dropout") or 0.0)
                cfg = (kind, in_f, hidden, num_layers, dropout, batch_first, freeze)
            else:
                cfg = (
                    kind, in_f, hidden,
                    int(inputs.get("num_layers") or 1),
                    float(inputs.get("dropout") or 0.0),
                    bool(inputs.get("bidirectional", False)),
                    batch_first,
                    freeze,
                    str(inputs.get("nonlinearity") or "tanh") if kind == "rnn" else None,
                )

            if self._layer is None or self._layer_cfg != cfg:
                if kind == "clockwork":
                    self._layer = ClockworkRNN(
                        input_size=in_f, hidden_size=hidden,
                        num_modules=cfg[3], batch_first=batch_first,
                    )
                elif kind == "indrnn":
                    self._layer = IndRNN(
                        input_size=in_f, hidden_size=hidden,
                        num_layers=cfg[3], dropout=cfg[4],
                        batch_first=batch_first,
                    )
                else:
                    kw = dict(input_size=in_f, hidden_size=cfg[2],
                              num_layers=cfg[3], dropout=cfg[4],
                              bidirectional=cfg[5], batch_first=cfg[6])
                    if kind == "lstm":
                        self._layer = nn.LSTM(**kw)
                    elif kind == "gru":
                        self._layer = nn.GRU(**kw)
                    else:  # rnn
                        self._layer = nn.RNN(nonlinearity=cfg[8], **kw)
                if freeze:
                    for p in self._layer.parameters():
                        p.requires_grad = False
                self._layer_cfg = cfg

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

        out_var = ov.get("output", "_out")
        hid_var = ov.get("hidden", "_hidden")

        # Variants: emit class definition inline so the script is self-contained.
        if kind in ("clockwork", "indrnn"):
            cls = ClockworkRNN if kind == "clockwork" else IndRNN
            src = inspect.getsource(cls)
            imports = ["import torch", "import torch.nn as nn",
                       "import torch.nn.functional as F", "import math"]
            lines = [src]
            if kind == "clockwork":
                lines.append(
                    f"{lv} = {cls.__name__}("
                    f"input_size={x}.shape[-1], "
                    f"hidden_size={self._val(iv, 'hidden_size')}, "
                    f"num_modules={self._val(iv, 'num_modules')}, "
                    f"batch_first={self._val(iv, 'batch_first')})"
                )
            else:
                lines.append(
                    f"{lv} = {cls.__name__}("
                    f"input_size={x}.shape[-1], "
                    f"hidden_size={self._val(iv, 'hidden_size')}, "
                    f"num_layers={self._val(iv, 'num_layers')}, "
                    f"dropout={self._val(iv, 'dropout')}, "
                    f"batch_first={self._val(iv, 'batch_first')})"
                )
            if self.inputs["freeze"].default_value:
                lines.append(f"for _p in {lv}.parameters(): _p.requires_grad = False")
            call = f"{lv}({x}, {h0})" if h0 else f"{lv}({x})"
            cell_var = ov.get("cell", "_cell")
            lines += [
                f"{out_var}, {hid_var} = {call}",
                f"{cell_var} = None",
                f"{ov.get('module', f'_{kind}_module')} = {lv}",
            ]
            return imports, lines

        # Built-in nn.* path
        cls_name = {"rnn": "RNN", "lstm": "LSTM", "gru": "GRU"}[kind]
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

        lines = [f"{lv} = nn.{cls_name}("] + ctor_args + [")"]
        if self.inputs["freeze"].default_value:
            lines.append(f"for _p in {lv}.parameters(): _p.requires_grad = False")

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
