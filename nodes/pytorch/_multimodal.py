"""MultimodalFusion — fuses pre-encoded modality features with optional missing handling.

In the GraphAsModule architecture, modality encoders are ordinary layer nodes in the
graph. This module is fusion-only: it takes a dict of already-encoded features and
combines them. Missing modalities are handled via zeros / noise / learnable tokens.
"""
from __future__ import annotations
import torch
import torch.nn as nn


_FUSION_OPS = ("concat", "sum", "mean", "attention", "gated")
_MISSING_OPS = ("zeros", "gaussian", "uniform", "learnable_token", "skip")


def _flatten_features(t: torch.Tensor) -> torch.Tensor:
    """Flatten anything to (B, F). Encoders may emit conv feature maps, sequences, etc."""
    if t.dim() == 1:
        return t.unsqueeze(0)
    if t.dim() == 2:
        return t
    return t.flatten(start_dim=1)


class MultimodalFusion(nn.Module):
    """Fuses pre-encoded modality features.

    Args:
        modalities:       list of modality names (order matters — defines fusion order)
        fusion:           'concat' | 'sum' | 'mean' | 'attention' | 'gated'
        fusion_dim:       projection dim per modality before fusion
        missing_strategy: how to fill a None modality
        use_active_flag:  append a binary [B, M] mask of present modalities
    """

    def __init__(
        self,
        modalities: list[str],
        fusion: str = "concat",
        fusion_dim: int = 512,
        missing_strategy: str = "zeros",
        use_active_flag: bool = False,
    ):
        super().__init__()
        if fusion not in _FUSION_OPS:
            raise ValueError(f"fusion must be one of {_FUSION_OPS}")
        if missing_strategy not in _MISSING_OPS:
            raise ValueError(f"missing_strategy must be one of {_MISSING_OPS}")

        self.modalities       = list(modalities)
        self.fusion           = fusion
        self.fusion_dim       = int(fusion_dim)
        self.missing_strategy = missing_strategy
        self.use_active_flag  = bool(use_active_flag)

        # Lazy per-modality projection (built on first real feature for that modality)
        self.projections: nn.ModuleDict = nn.ModuleDict()

        # Learnable mask token per modality
        self.mask_tokens = nn.ParameterDict({
            m: nn.Parameter(torch.zeros(self.fusion_dim))
            for m in self.modalities
        })

        # Attention fusion
        if fusion == "attention":
            self.attn_query = nn.Parameter(torch.randn(1, 1, self.fusion_dim) * 0.02)
            self.attn = nn.MultiheadAttention(self.fusion_dim, num_heads=4, batch_first=True)

        # Gated fusion
        if fusion == "gated":
            in_dim = self.fusion_dim * len(self.modalities)
            self.gate_net = nn.Sequential(
                nn.Linear(in_dim, in_dim // 2), nn.ReLU(),
                nn.Linear(in_dim // 2, len(self.modalities)), nn.Sigmoid(),
            )

    # -------------------------------------------------------- helpers

    def _project(self, modality: str, x: torch.Tensor) -> torch.Tensor:
        x = _flatten_features(x)
        in_dim = x.shape[-1]
        if modality not in self.projections:
            self.projections[modality] = nn.Linear(in_dim, self.fusion_dim).to(x.device)
        return self.projections[modality](x)

    def _fill_missing(self, modality: str, batch_size: int, device, dtype) -> torch.Tensor:
        if self.missing_strategy == "zeros":
            return torch.zeros(batch_size, self.fusion_dim, device=device, dtype=dtype)
        if self.missing_strategy == "gaussian":
            return torch.randn(batch_size, self.fusion_dim, device=device, dtype=dtype) * 0.1
        if self.missing_strategy == "uniform":
            return (torch.rand(batch_size, self.fusion_dim, device=device, dtype=dtype) - 0.5) * 0.2
        if self.missing_strategy == "learnable_token":
            return self.mask_tokens[modality].unsqueeze(0).expand(batch_size, -1).to(device=device, dtype=dtype)
        return torch.zeros(batch_size, self.fusion_dim, device=device, dtype=dtype)

    # -------------------------------------------------------- forward

    def forward(self, features: dict[str, torch.Tensor | None]) -> torch.Tensor:
        """features: {modality_name: pre-encoded tensor or None}"""
        # Reference tensor to infer batch size/device
        ref = next((v for v in features.values() if v is not None), None)
        if ref is None:
            raise ValueError("MultimodalFusion.forward() called with no present modalities")
        B, device, dtype = ref.shape[0] if ref.dim() > 0 else 1, ref.device, ref.dtype

        feats: list[torch.Tensor] = []
        present: list[bool] = []
        for m in self.modalities:
            x = features.get(m)
            if x is not None:
                feats.append(self._project(m, x))
                present.append(True)
            else:
                if self.missing_strategy == "skip":
                    feats.append(torch.zeros(B, self.fusion_dim, device=device, dtype=dtype))
                    present.append(False)
                else:
                    feats.append(self._fill_missing(m, B, device, dtype))
                    present.append(True)

        # Fuse
        if self.fusion == "concat":
            fused = torch.cat(feats, dim=-1)
        elif self.fusion == "sum":
            fused = torch.stack(feats, dim=1).sum(dim=1)
        elif self.fusion == "mean":
            stk = torch.stack(feats, dim=1)
            denom = max(1, sum(present))
            fused = stk.sum(dim=1) / denom
        elif self.fusion == "attention":
            stk = torch.stack(feats, dim=1)
            q = self.attn_query.expand(B, -1, -1)
            key_pad = torch.tensor([[not p for p in present]] * B, device=device)
            out, _ = self.attn(q, stk, stk, key_padding_mask=key_pad if not all(present) else None)
            fused = out.squeeze(1)
        elif self.fusion == "gated":
            stk = torch.stack(feats, dim=1)
            flat = stk.flatten(start_dim=1)
            gates = self.gate_net(flat).unsqueeze(-1)
            fused = (stk * gates).sum(dim=1)
        else:
            fused = torch.cat(feats, dim=-1)

        if self.use_active_flag:
            mask = torch.tensor([[1.0 if p else 0.0 for p in present]] * B,
                                device=device, dtype=dtype)
            fused = torch.cat([fused, mask], dim=-1)

        return fused
