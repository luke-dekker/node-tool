"""nn.MultiheadAttention layer node — also covers memory-attention over an
RNN's hidden history when wired to one source.

The "memory attention" / "RNN with attention over its past" pattern Luke is
exploring is just self-attention with `causal=True` and (optionally) a
sliding `window`: at step t, the QK-softmax over [max(0, t-window+1), t]
*is* the small selector model that picks which past timestep to mix in.

Wire RNN.output → `query`, leave `key`/`value` unwired (defaults to query),
turn on `causal`, optionally cap `window`, and turn on `residual` to get a
drop-in attention-augmented RNN block. The `attention_weights` output
exposes the selector's distribution for inspection.
"""
from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
from core.node import BaseNode, PortType


class MultiheadAttentionNode(BaseNode):
    type_name   = "pt_multihead_attention"
    label       = "Multihead Attention"
    category    = "Layers"
    subcategory = "Attention"
    description = (
        "nn.MultiheadAttention(embed_dim, num_heads). Wire only `query` for "
        "self-attention; wire `key`/`value` too for cross-attention. Inputs "
        "(B, T, embed_dim).\n"
        "  • causal=True       — mask out future positions (decoder-style)\n"
        "  • window>0          — only attend to the last `window` positions\n"
        "                         (sliding-window self-attention)\n"
        "  • residual=True     — output = query + attn(query, key, value),\n"
        "                         drops in between an RNN and the next layer\n"
        "  • `attention_weights` output exposes the post-softmax weights\n"
        "    (B, num_heads, T, S) — the selector pattern over past steps."
    )

    def __init__(self):
        self._layer: nn.MultiheadAttention | None = None
        self._layer_cfg: tuple | None = None
        # Cache window masks keyed by (T, window, device) so we don't rebuild
        # them every forward (training over a fixed-T batch hits this hot).
        self._mask_cache: dict[tuple, torch.Tensor] = {}
        super().__init__()

    def _get_layer(self, embed_dim, num_heads, dropout):
        cfg = (embed_dim, num_heads, dropout)
        if self._layer is None or self._layer_cfg != cfg:
            self._layer = nn.MultiheadAttention(
                embed_dim=embed_dim, num_heads=num_heads,
                dropout=dropout, batch_first=True,
            )
            self._layer_cfg = cfg
        return self._layer

    def get_layers(self) -> list[nn.Module]:
        return [self._layer] if self._layer is not None else []

    def _setup_ports(self) -> None:
        self.add_input("query",      PortType.TENSOR, default=None,
                       description="(B, T, embed_dim) query")
        self.add_input("key",        PortType.TENSOR, default=None,
                       description="(B, S, embed_dim) key — defaults to query for self-attn")
        self.add_input("value",      PortType.TENSOR, default=None,
                       description="(B, S, embed_dim) value — defaults to key")
        self.add_input("num_heads",  PortType.INT, default=8)
        self.add_input("dropout",    PortType.FLOAT, default=0.0)
        self.add_input("causal",     PortType.BOOL, default=False,
                       description="Mask future positions (i can only attend to j ≤ i).")
        self.add_input("window",     PortType.INT, default=0,
                       description="Sliding-window size. 0 = no window. Combine with "
                                   "causal=True for memory-attention over the last "
                                   "`window` past hidden states.")
        self.add_input("residual",   PortType.BOOL, default=False,
                       description="Add query to the attention output (drop-in block).")
        self.add_output("tensor_out",         PortType.TENSOR,
                        description="(B, T, embed_dim) attended output (+ query if residual).")
        self.add_output("attention_weights",  PortType.TENSOR,
                        description="(B, num_heads, T, S) post-softmax attention weights, "
                                    "or None when needs_weights wasn't requested.")

    def _build_mask(self, T_q: int, T_k: int, causal: bool, window: int,
                    device, dtype) -> torch.Tensor | None:
        """Build a (T_q, T_k) boolean mask. True = don't attend.

        Causal alone: nn.MultiheadAttention's is_causal kwarg is faster
        (avoids materializing the mask), so callers should use that path
        when window == 0. This is for the windowed cases.
        """
        if not causal and window <= 0:
            return None
        key = (T_q, T_k, causal, window, str(device))
        cached = self._mask_cache.get(key)
        if cached is not None:
            return cached
        mask = torch.zeros(T_q, T_k, dtype=torch.bool, device=device)
        if causal:
            # Standard causal: position i can't attend to j > i. Self-attn:
            # T_q == T_k. Cross-attn: align by trailing index (rare here).
            offset = T_k - T_q
            for i in range(T_q):
                mask[i, i + offset + 1:] = True
        if window > 0:
            # Position i can only attend to j in [i - window + 1, i].
            offset = T_k - T_q
            for i in range(T_q):
                lo = max(0, i + offset - window + 1)
                mask[i, :lo] = True
        self._mask_cache[key] = mask
        return mask

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        from nodes.pytorch._helpers import _infer_feature_dim
        q = inputs.get("query")
        embed_dim = _infer_feature_dim(q, None, axis=-1)
        if q is None or embed_dim <= 0:
            return {"tensor_out": None, "attention_weights": None}
        layer = self._get_layer(
            embed_dim,
            int(inputs.get("num_heads") or 8),
            float(inputs.get("dropout") or 0.0),
        )
        k = inputs.get("key")   if inputs.get("key")   is not None else q
        v = inputs.get("value") if inputs.get("value") is not None else k
        causal   = bool(inputs.get("causal", False))
        window   = int(inputs.get("window") or 0)
        residual = bool(inputs.get("residual", False))

        try:
            T_q = q.shape[-2]
            T_k = k.shape[-2]
            # Fast path: no window + plain or causal — let nn.MHA's is_causal do it.
            if window <= 0:
                out, attn = layer(
                    q, k, v,
                    need_weights=True, average_attn_weights=False,
                    is_causal=causal,
                    attn_mask=(nn.Transformer.generate_square_subsequent_mask(T_q,
                                                                              device=q.device,
                                                                              dtype=q.dtype)
                               if (causal and T_q == T_k) else None),
                )
            else:
                # Windowed (and possibly causal): build explicit boolean mask.
                mask = self._build_mask(T_q, T_k, causal, window, q.device, q.dtype)
                out, attn = layer(
                    q, k, v,
                    need_weights=True, average_attn_weights=False,
                    attn_mask=mask,
                )
            if residual:
                if out.shape == q.shape:
                    out = out + q
            return {"tensor_out": out, "attention_weights": attn}
        except Exception:
            return {"tensor_out": None, "attention_weights": None}

    def export(self, iv, ov):
        lv = f"_mha_{self.safe_id}"
        q  = iv.get("query") or "_x"
        k  = iv.get("key")   or q
        v  = iv.get("value") or k
        out  = ov.get("tensor_out",        "_out")
        attn = ov.get("attention_weights", "_attn")
        causal   = bool(self.inputs["causal"].default_value)
        window   = int(self.inputs["window"].default_value or 0)
        residual = bool(self.inputs["residual"].default_value)
        nh       = self._val(iv, "num_heads")
        dp       = self._val(iv, "dropout")

        lines = [
            f"{lv} = nn.MultiheadAttention(embed_dim={q}.shape[-1], "
            f"num_heads={nh}, dropout={dp}, batch_first=True)",
        ]
        # Build attn_mask if needed.
        if window > 0 or causal:
            lines += [
                f"_T_q = {q}.shape[-2]; _T_k = {k}.shape[-2]",
                f"_mask = torch.zeros(_T_q, _T_k, dtype=torch.bool, device={q}.device)",
            ]
            if causal:
                lines += [
                    f"_off = _T_k - _T_q",
                    f"for _i in range(_T_q): _mask[_i, _i + _off + 1:] = True",
                ]
            if window > 0:
                lines += [
                    f"_off = _T_k - _T_q",
                    f"for _i in range(_T_q):",
                    f"    _lo = max(0, _i + _off - {window} + 1)",
                    f"    _mask[_i, :_lo] = True",
                ]
            lines.append(
                f"{out}, {attn} = {lv}({q}, {k}, {v}, "
                f"need_weights=True, average_attn_weights=False, attn_mask=_mask)"
            )
        else:
            lines.append(
                f"{out}, {attn} = {lv}({q}, {k}, {v}, "
                f"need_weights=True, average_attn_weights=False)"
            )
        if residual:
            lines.append(f"if {out}.shape == {q}.shape: {out} = {out} + {q}")
        return ["import torch", "import torch.nn as nn"], lines
