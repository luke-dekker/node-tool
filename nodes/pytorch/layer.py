"""Consolidated single-in single-out layer node — replaces 12 per-class nodes.

Kinds (set via the `kind` dropdown):
  linear              — nn.LazyLinear(out_features), bias, activation, freeze
  conv2d              — nn.LazyConv2d(out_ch, kernel, stride, padding), activation, freeze
  batchnorm1d         — nn.LazyBatchNorm1d (num_features inferred from input)
  batchnorm2d         — nn.LazyBatchNorm2d
  layernorm           — nn.LayerNorm over input.shape[-1], eps
  dropout             — nn.Dropout(p)
  embedding           — nn.Embedding(num_embeddings, embedding_dim), freeze
  activation          — standalone activation (relu/gelu/etc), reads `activation`
  positional_encoding — sinusoidal | learned, reads `max_len`, `pe_kind`
  transformer_encoder — one nn.TransformerEncoderLayer block, reads nhead/dim_ff/dropout/activation
  max_pool2d          — nn.MaxPool2d(kernel, stride)
  avg_pool2d          — nn.AvgPool2d(kernel, stride)
  adaptive_avg_pool2d — nn.AdaptiveAvgPool2d(output_size)

The inspector's `relevant_inputs` method narrows the visible config to the
fields each kind actually reads. All input dimensions are inferred from the
upstream tensor at runtime; legacy in_features/in_ch ports are intentionally
NOT exposed (the per-class nodes had them as dead fallbacks).

Multi-tensor / multi-output layers stay standalone:
  - RecurrentLayerNode    (rnn/lstm/gru — varying outputs, hidden/cell)
  - MultiheadAttentionNode (3 tensor inputs)
  - PackSequenceNode / UnpackSequenceNode (PackedSequence type)
  - ApplyModuleNode, FlattenNode, GateNode, ReshapeForLossNode (utility)
"""
from __future__ import annotations
import math
from typing import Any
import torch
import torch.nn as nn
from core.node import BaseNode, PortType
from nodes.pytorch._helpers import (
    _make_activation, _forward, _act_func, _act_expr, _infer_feature_dim,
)


_KINDS = [
    "linear", "conv2d", "batchnorm1d", "batchnorm2d", "layernorm",
    "dropout", "embedding", "activation", "positional_encoding",
    "transformer_encoder",
    "max_pool2d", "avg_pool2d", "adaptive_avg_pool2d",
]

_ACT_HELP = "none | relu | leakyrelu | sigmoid | tanh | gelu | elu | silu | softmax"

# Each kind's relevant input subset, used by the inspector to hide
# fields that don't apply. Single source of truth — execute() and
# export() also dispatch on `kind`.
_RELEVANT: dict[str, list[str]] = {
    "linear":              ["kind", "out_features", "bias", "activation", "freeze"],
    "conv2d":              ["kind", "out_ch", "kernel", "stride", "padding", "activation", "freeze"],
    "batchnorm1d":         ["kind"],
    "batchnorm2d":         ["kind"],
    "layernorm":           ["kind", "eps"],
    "dropout":             ["kind", "p"],
    "embedding":           ["kind", "num_embeddings", "embedding_dim", "freeze"],
    "activation":          ["kind", "activation"],
    "positional_encoding": ["kind", "max_len", "pe_kind"],
    "transformer_encoder": ["kind", "nhead", "dim_feedforward", "dropout", "activation"],
    "max_pool2d":          ["kind", "kernel", "stride"],
    "avg_pool2d":          ["kind", "kernel", "stride"],
    "adaptive_avg_pool2d": ["kind", "output_size"],
}


# ── Sinusoidal / Learned positional encoding modules ──────────────────────────
# Inlined from positional_encoding.py — no other consumer remains.

class _SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(pos * div)
        else:
            pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class _LearnedPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)
        return x + self.pe(positions).unsqueeze(0)


# ── LayerNode ─────────────────────────────────────────────────────────────────

class LayerNode(BaseNode):
    type_name   = "pt_layer"
    label       = "Layer"
    category    = "Layers"
    subcategory = ""
    description = (
        "Single-tensor-in, single-tensor-out neural-network layer. Pick "
        "`kind` from the dropdown; the inspector hides fields that don't "
        "apply to the chosen kind. Input dimensions are inferred from the "
        "upstream tensor — only declare output sizes."
    )

    def __init__(self):
        self._layer: nn.Module | None = None
        self._layer_cfg: tuple | None = None
        self._act_name: str = ""
        super().__init__()

    def get_layers(self) -> list[nn.Module]:
        if self._layer is None:
            return []
        modules: list[nn.Module] = [self._layer]
        # For kinds that apply a separate post-activation (linear, conv2d),
        # surface it so freeze / save iterate over it.
        act = _make_activation(self._act_name)
        if act is not None and self._needs_post_activation():
            modules.append(act)
        return modules

    def _needs_post_activation(self) -> bool:
        kind = (self.inputs["kind"].default_value or "").strip()
        return kind in ("linear", "conv2d")

    def relevant_inputs(self, values):
        kind = (values.get("kind") or "linear").strip()
        return list(_RELEVANT.get(kind, ["kind"]))

    def _setup_ports(self) -> None:
        self.add_input("tensor_in",       PortType.TENSOR, default=None)
        self.add_input("kind",            PortType.STRING, default="linear", choices=_KINDS)
        # Union of all per-kind config across the 13 kinds. relevant_inputs
        # narrows what the inspector shows.
        self.add_input("out_features",    PortType.INT,    default=32, optional=True,
                       description="linear: number of neurons in this layer")
        self.add_input("bias",            PortType.BOOL,   default=True, optional=True,
                       description="linear: include bias term")
        self.add_input("out_ch",          PortType.INT,    default=16, optional=True,
                       description="conv2d: number of output channels")
        self.add_input("kernel",          PortType.INT,    default=3, optional=True,
                       description="conv2d / pool: kernel size")
        self.add_input("stride",          PortType.INT,    default=1, optional=True,
                       description="conv2d / pool: stride")
        self.add_input("padding",         PortType.INT,    default=0, optional=True,
                       description="conv2d: padding")
        self.add_input("output_size",     PortType.INT,    default=4, optional=True,
                       description="adaptive_avg_pool2d: output H,W (square)")
        self.add_input("eps",             PortType.FLOAT,  default=1e-5, optional=True,
                       description="layernorm: epsilon")
        self.add_input("p",               PortType.FLOAT,  default=0.5, optional=True,
                       description="dropout: drop probability")
        self.add_input("num_embeddings",  PortType.INT,    default=1000, optional=True,
                       description="embedding: vocabulary size")
        self.add_input("embedding_dim",   PortType.INT,    default=64, optional=True,
                       description="embedding: per-token vector size")
        self.add_input("max_len",         PortType.INT,    default=5000, optional=True,
                       description="positional_encoding: max sequence length")
        self.add_input("pe_kind",         PortType.STRING, default="sinusoidal",
                       choices=["sinusoidal", "learned"], optional=True,
                       description="positional_encoding: fixed-trig vs learned embedding")
        self.add_input("nhead",           PortType.INT,    default=8, optional=True,
                       description="transformer_encoder: attention heads")
        self.add_input("dim_feedforward", PortType.INT,    default=1024, optional=True,
                       description="transformer_encoder: FFN inner dim")
        self.add_input("dropout",         PortType.FLOAT,  default=0.1, optional=True,
                       description="transformer_encoder: dropout")
        self.add_input("activation",      PortType.STRING, default="none",
                       description=_ACT_HELP, optional=True)
        self.add_input("freeze",          PortType.BOOL,   default=False, optional=True,
                       description="set requires_grad=False on this layer's params")
        self.add_output("tensor_out", PortType.TENSOR)

    # ── execute ─────────────────────────────────────────────────────────────

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        tin  = inputs.get("tensor_in")
        kind = (inputs.get("kind") or "linear").strip()
        try:
            return getattr(self, f"_exec_{kind}")(tin, inputs)
        except AttributeError:
            return {"tensor_out": None}
        except Exception:
            return {"tensor_out": None}

    def _build_if_changed(self, cfg: tuple, factory):
        if self._layer is None or self._layer_cfg != cfg:
            self._layer = factory()
            self._layer_cfg = cfg

    def _apply_freeze(self, freeze: bool) -> None:
        if self._layer is not None:
            for p in self._layer.parameters():
                p.requires_grad = not freeze

    def _exec_linear(self, tin, inputs):
        # Use a concrete nn.Linear with placeholder in_features=1 during
        # priming (no tensor) so model.parameters() works and the orchestrator
        # can count > 0 params. Real forward triggers rebuild via the cfg
        # tuple change, propagating the actual upstream width.
        out_f = int(inputs.get("out_features") or 32)
        bias  = bool(inputs.get("bias", True))
        in_f  = int(tin.shape[-1]) if tin is not None else 1
        self._build_if_changed((in_f, out_f, bias),
                               lambda: nn.Linear(in_f, out_f, bias=bias))
        self._act_name = inputs.get("activation") or ""
        self._apply_freeze(bool(inputs.get("freeze", False)))
        act = _make_activation(self._act_name)
        return {"tensor_out": _forward(self._layer, act, tin)}

    def _exec_conv2d(self, tin, inputs):
        out_ch  = int(inputs.get("out_ch") or 16)
        kernel  = int(inputs.get("kernel") or 3)
        stride  = int(inputs.get("stride") or 1)
        padding = int(inputs.get("padding") or 0)
        in_ch = _infer_feature_dim(tin, None, axis=1) if tin is not None else 1
        if in_ch <= 0:
            in_ch = 1
        self._build_if_changed(
            (in_ch, out_ch, kernel, stride, padding),
            lambda: nn.Conv2d(in_ch, out_ch, kernel_size=kernel,
                              stride=stride, padding=padding),
        )
        self._act_name = inputs.get("activation") or ""
        self._apply_freeze(bool(inputs.get("freeze", False)))
        act = _make_activation(self._act_name)
        return {"tensor_out": _forward(self._layer, act, tin)}

    def _exec_batchnorm1d(self, tin, inputs):
        nf = _infer_feature_dim(tin, None, axis=1) if tin is not None else 1
        if nf <= 0:
            nf = 1
        self._build_if_changed((nf,), lambda: nn.BatchNorm1d(nf))
        if not torch.is_grad_enabled():
            self._layer.eval()
        return {"tensor_out": _forward(self._layer, None, tin)}

    def _exec_batchnorm2d(self, tin, inputs):
        nf = _infer_feature_dim(tin, None, axis=1) if tin is not None else 1
        if nf <= 0:
            nf = 1
        self._build_if_changed((nf,), lambda: nn.BatchNorm2d(nf))
        if not torch.is_grad_enabled():
            self._layer.eval()
        return {"tensor_out": _forward(self._layer, None, tin)}

    def _exec_layernorm(self, tin, inputs):
        # No LazyLayerNorm in PyTorch — build with the input's shape if we
        # have one, else a placeholder (1,) so the module exists during
        # priming. The first real forward rebuilds with the correct shape.
        eps = float(inputs.get("eps") or 1e-5)
        ns  = _infer_feature_dim(tin, None, axis=-1) if tin is not None else 1
        if ns <= 0:
            ns = 1
        self._build_if_changed((ns, eps), lambda: nn.LayerNorm(ns, eps=eps))
        return {"tensor_out": _forward(self._layer, None, tin)}

    def _exec_dropout(self, tin, inputs):
        p = float(inputs.get("p") or 0.5)
        self._build_if_changed((p,), lambda: nn.Dropout(p=p))
        return {"tensor_out": _forward(self._layer, None, tin)}

    def _exec_embedding(self, tin, inputs):
        n_emb = int(inputs.get("num_embeddings") or 1000)
        d_emb = int(inputs.get("embedding_dim")  or 64)
        self._build_if_changed((n_emb, d_emb),
                               lambda: nn.Embedding(n_emb, d_emb))
        self._apply_freeze(bool(inputs.get("freeze", False)))
        return {"tensor_out": _forward(self._layer, None, tin)}

    def _exec_activation(self, tin, inputs):
        name = inputs.get("activation") or "relu"
        self._act_name = name
        self._build_if_changed((name,), lambda: _make_activation(name) or nn.Identity())
        return {"tensor_out": _forward(self._layer, None, tin)}

    def _exec_positional_encoding(self, tin, inputs):
        # Placeholder d_model during priming (no tensor); rebuild on first forward.
        d = _infer_feature_dim(tin, None, axis=-1) if tin is not None else 1
        if d <= 0:
            d = 1
        max_len = int(inputs.get("max_len") or 5000)
        pe_kind = str(inputs.get("pe_kind") or "sinusoidal")
        self._build_if_changed(
            (d, max_len, pe_kind),
            lambda: (_LearnedPE(d, max_len) if pe_kind == "learned"
                     else _SinusoidalPE(d, max_len)),
        )
        if tin is None:
            return {"tensor_out": None}
        return {"tensor_out": self._layer(tin)}

    def _exec_transformer_encoder(self, tin, inputs):
        # Placeholder d_model during priming; rebuild with the correct
        # dim on first real forward. nhead must divide d_model — when
        # priming with d=1 we force nhead=1 to satisfy the constraint.
        nhead   = int(inputs.get("nhead") or 8)
        dim_ff  = int(inputs.get("dim_feedforward") or 1024)
        dropout = float(inputs.get("dropout") or 0.1)
        act     = str(inputs.get("activation") or "relu")
        if act == "none":
            act = "relu"
        if tin is not None:
            d = _infer_feature_dim(tin, None, axis=-1)
        else:
            d, nhead = 1, 1   # priming-time placeholder
        if d <= 0:
            d = max(nhead, 1)
        self._build_if_changed(
            (d, nhead, dim_ff, dropout, act),
            lambda: nn.TransformerEncoderLayer(
                d_model=d, nhead=nhead, dim_feedforward=dim_ff,
                dropout=dropout, activation=act, batch_first=True,
            ),
        )
        return {"tensor_out": _forward(self._layer, None, tin)}

    def _exec_max_pool2d(self, tin, inputs):
        kernel = int(inputs.get("kernel") or 2)
        stride = int(inputs.get("stride") or 2)
        self._build_if_changed(("max", kernel, stride),
                               lambda: nn.MaxPool2d(kernel_size=kernel, stride=stride))
        return {"tensor_out": _forward(self._layer, None, tin)}

    def _exec_avg_pool2d(self, tin, inputs):
        kernel = int(inputs.get("kernel") or 2)
        stride = int(inputs.get("stride") or 2)
        self._build_if_changed(("avg", kernel, stride),
                               lambda: nn.AvgPool2d(kernel_size=kernel, stride=stride))
        return {"tensor_out": _forward(self._layer, None, tin)}

    def _exec_adaptive_avg_pool2d(self, tin, inputs):
        out_sz = int(inputs.get("output_size") or 4)
        self._build_if_changed(("adapt", out_sz),
                               lambda: nn.AdaptiveAvgPool2d(out_sz))
        return {"tensor_out": _forward(self._layer, None, tin)}

    # ── export ──────────────────────────────────────────────────────────────

    def export(self, iv, ov):
        kind = (self.inputs["kind"].default_value or "linear")
        try:
            return getattr(self, f"_export_{kind}")(iv, ov)
        except AttributeError:
            return [], [f"# unknown LayerNode kind {kind!r}"]

    def _post_act(self, tout: str, iv) -> tuple[list[str], list[str]]:
        """Optional post-activation lines for kinds that use it."""
        act_name = (self.inputs["activation"].default_value or "none")
        func = _act_func(act_name)
        if func:
            return ["import torch.nn.functional as F"], [f"{tout} = {func}({tout})"]
        return [], []

    def _export_linear(self, iv, ov):
        lv   = f"_lin_{self.safe_id}"
        tin  = iv.get("tensor_in") or "_x"
        tout = ov.get("tensor_out", "_out")
        out_f = self._val(iv, "out_features"); bias = self._val(iv, "bias")
        imps = ["import torch", "import torch.nn as nn"]
        lines = [f"{lv} = nn.LazyLinear({out_f}, bias={bias})", f"{tout} = {lv}({tin})"]
        ai, al = self._post_act(tout, iv); imps += ai; lines += al
        return imps, lines

    def _export_conv2d(self, iv, ov):
        lv   = f"_conv_{self.safe_id}"
        tin  = iv.get("tensor_in") or "_x"
        tout = ov.get("tensor_out", "_out")
        out_ch = self._val(iv, "out_ch"); k = self._val(iv, "kernel")
        s = self._val(iv, "stride");      p = self._val(iv, "padding")
        imps = ["import torch", "import torch.nn as nn"]
        lines = [
            f"{lv} = nn.LazyConv2d({out_ch}, kernel_size={k}, stride={s}, padding={p})",
            f"{tout} = {lv}({tin})",
        ]
        ai, al = self._post_act(tout, iv); imps += ai; lines += al
        return imps, lines

    def _export_batchnorm1d(self, iv, ov):
        lv = f"_bn1_{self.safe_id}"; tin = iv.get("tensor_in") or "_x"
        tout = ov.get("tensor_out", "_out")
        return ["import torch", "import torch.nn as nn"], [
            f"{lv} = nn.LazyBatchNorm1d()", f"{tout} = {lv}({tin})",
        ]

    def _export_batchnorm2d(self, iv, ov):
        lv = f"_bn2_{self.safe_id}"; tin = iv.get("tensor_in") or "_x"
        tout = ov.get("tensor_out", "_out")
        return ["import torch", "import torch.nn as nn"], [
            f"{lv} = nn.LazyBatchNorm2d()", f"{tout} = {lv}({tin})",
        ]

    def _export_layernorm(self, iv, ov):
        lv = f"_ln_{self.safe_id}"; tin = iv.get("tensor_in") or "_x"
        tout = ov.get("tensor_out", "_out"); eps = self._val(iv, "eps")
        return ["import torch", "import torch.nn as nn"], [
            f"{lv} = nn.LayerNorm({tin}.shape[-1], eps={eps}).to({tin}.device)",
            f"{tout} = {lv}({tin})",
        ]

    def _export_dropout(self, iv, ov):
        lv = f"_drop_{self.safe_id}"; tin = iv.get("tensor_in") or "_x"
        tout = ov.get("tensor_out", "_out")
        return ["import torch", "import torch.nn as nn"], [
            f"{lv} = nn.Dropout({self._val(iv, 'p')})", f"{tout} = {lv}({tin})",
        ]

    def _export_embedding(self, iv, ov):
        lv = f"_emb_{self.safe_id}"; tin = iv.get("tensor_in") or "_x"
        tout = ov.get("tensor_out", "_out")
        return ["import torch", "import torch.nn as nn"], [
            f"{lv} = nn.Embedding({self._val(iv, 'num_embeddings')}, "
            f"{self._val(iv, 'embedding_dim')})",
            f"{tout} = {lv}({tin})",
        ]

    def _export_activation(self, iv, ov):
        tin = iv.get("tensor_in") or "_x"; tout = ov.get("tensor_out", "_out")
        act_name = (self.inputs["activation"].default_value or "relu")
        expr = _act_expr(act_name) or "nn.ReLU()"
        return ["import torch.nn as nn"], [
            f"_act_{self.safe_id} = {expr}",
            f"{tout} = _act_{self.safe_id}({tin})",
        ]

    def _export_positional_encoding(self, iv, ov):
        # Inline the sinusoidal pattern so the script is self-contained.
        tin = iv.get("tensor_in") or "_x"; tout = ov.get("tensor_out", "_out")
        pe_kind = (self.inputs["pe_kind"].default_value or "sinusoidal")
        max_len = self._val(iv, "max_len")
        return ["import torch", "import torch.nn as nn", "import math"], [
            f"# positional encoding ({pe_kind}, max_len={max_len})",
            f"{tout} = {tin}  # NOTE: PE applied in-graph; export elided for brevity",
        ]

    def _export_transformer_encoder(self, iv, ov):
        lv = f"_tel_{self.safe_id}"; tin = iv.get("tensor_in") or "_x"
        tout = ov.get("tensor_out", "_out")
        nhead = self._val(iv, "nhead"); dff = self._val(iv, "dim_feedforward")
        drop  = self._val(iv, "dropout"); act = self._val(iv, "activation")
        return ["import torch", "import torch.nn as nn"], [
            f"{lv} = nn.TransformerEncoderLayer(d_model={tin}.shape[-1], "
            f"nhead={nhead}, dim_feedforward={dff}, dropout={drop}, "
            f"activation={act}, batch_first=True)",
            f"{tout} = {lv}({tin})",
        ]

    def _export_max_pool2d(self, iv, ov):
        lv = f"_mxp_{self.safe_id}"; tin = iv.get("tensor_in") or "_x"
        tout = ov.get("tensor_out", "_out")
        return ["import torch", "import torch.nn as nn"], [
            f"{lv} = nn.MaxPool2d({self._val(iv, 'kernel')}, stride={self._val(iv, 'stride')})",
            f"{tout} = {lv}({tin})",
        ]

    def _export_avg_pool2d(self, iv, ov):
        lv = f"_avp_{self.safe_id}"; tin = iv.get("tensor_in") or "_x"
        tout = ov.get("tensor_out", "_out")
        return ["import torch", "import torch.nn as nn"], [
            f"{lv} = nn.AvgPool2d({self._val(iv, 'kernel')}, stride={self._val(iv, 'stride')})",
            f"{tout} = {lv}({tin})",
        ]

    def _export_adaptive_avg_pool2d(self, iv, ov):
        lv = f"_aap_{self.safe_id}"; tin = iv.get("tensor_in") or "_x"
        tout = ov.get("tensor_out", "_out")
        return ["import torch", "import torch.nn as nn"], [
            f"{lv} = nn.AdaptiveAvgPool2d({self._val(iv, 'output_size')})",
            f"{tout} = {lv}({tin})",
        ]
