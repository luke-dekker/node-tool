"""ACT Policy node — Action Chunking with Transformers for imitation learning.

Wraps LeRobot's ACT policy for both training and inference. In training mode,
takes (observation, action) pairs from a LeRobot dataset and outputs a loss
tensor for the training loop. In inference mode, takes a live observation
and outputs predicted actions for the robot.

ACT is the default imitation learning algorithm in LeRobot. It predicts
a CHUNK of future actions (not just the next one), which gives smoother
motion and better temporal coherence.
"""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


class ACTPolicyNode(BaseNode):
    """ACT Policy — train from demonstrations or predict actions for deployment."""
    type_name   = "lr_act_policy"
    label       = "ACT Policy"
    category    = "LeRobot"
    description = (
        "Action Chunking with Transformers. In training: takes observation + "
        "action from a dataset, outputs loss. In inference: takes live "
        "observation, outputs predicted joint targets."
    )

    def __init__(self):
        self._policy = None
        self._layer = None  # for GraphAsModule parameter discovery
        super().__init__()

    def _setup_ports(self) -> None:
        # Configuration
        self.add_input("observation_dim", PortType.INT, 6,
                       description="Observation space dimension (6 for SO-101 joints)")
        self.add_input("action_dim",      PortType.INT, 6,
                       description="Action space dimension (6 for SO-101 joints)")
        self.add_input("chunk_size",      PortType.INT, 100,
                       description="Number of future actions predicted at once")
        self.add_input("hidden_dim",      PortType.INT, 512)
        self.add_input("n_heads",         PortType.INT, 8)
        self.add_input("n_layers",        PortType.INT, 4)
        # Data inputs
        self.add_input("observation",     PortType.TENSOR, None,
                       description="Joint positions (B, obs_dim) or (B, T, obs_dim)")
        self.add_input("action",          PortType.TENSOR, None,
                       description="Target actions for training (B, chunk_size, act_dim)")
        self.add_input("mode",            PortType.STRING, "train",
                       choices=["train", "inference"],
                       description="train: output loss. inference: output predicted actions.")
        # Outputs
        self.add_output("predicted_action", PortType.TENSOR,
                        description="Predicted action chunk (B, chunk_size, act_dim)")
        self.add_output("loss",            PortType.TENSOR,
                        description="Training loss (only in train mode)")
        self.add_output("model",           PortType.MODULE,
                        description="The ACT model for parameter discovery")
        self.add_output("info",            PortType.STRING)

    def _ensure_policy(self, obs_dim: int, act_dim: int, chunk: int,
                       hidden: int, n_heads: int, n_layers: int):
        """Lazily build the ACT policy."""
        cfg = (obs_dim, act_dim, chunk, hidden, n_heads, n_layers)
        if self._policy is not None:
            return

        try:
            # Try to use LeRobot's ACT implementation
            from lerobot.common.policies.act.modeling_act import ACTPolicy
            from lerobot.common.policies.act.configuration_act import ACTConfig
            config = ACTConfig(
                input_shapes={"observation.state": [obs_dim]},
                output_shapes={"action": [act_dim]},
                chunk_size=chunk,
                dim_model=hidden,
                n_heads=n_heads,
                n_encoder_layers=n_layers,
            )
            self._policy = ACTPolicy(config)
            self._layer = self._policy
        except (ImportError, Exception) as exc:
            # Fallback: build a simple transformer-based policy ourselves
            import torch
            import torch.nn as nn

            class SimpleACT(nn.Module):
                """Minimal ACT-like policy for when lerobot isn't installed."""
                def __init__(self, obs_dim, act_dim, chunk, hidden, n_heads, n_layers):
                    super().__init__()
                    self.obs_encoder = nn.Linear(obs_dim, hidden)
                    encoder_layer = nn.TransformerEncoderLayer(
                        d_model=hidden, nhead=n_heads, batch_first=True)
                    self.transformer = nn.TransformerEncoder(
                        encoder_layer, num_layers=n_layers)
                    self.action_head = nn.Linear(hidden, act_dim * chunk)
                    self.chunk_size = chunk
                    self.act_dim = act_dim

                def forward(self, obs):
                    if obs.dim() == 2:
                        obs = obs.unsqueeze(1)  # (B, 1, obs_dim)
                    h = self.obs_encoder(obs)
                    h = self.transformer(h)
                    h = h[:, -1, :]  # take last token
                    actions = self.action_head(h)
                    return actions.reshape(-1, self.chunk_size, self.act_dim)

            self._policy = SimpleACT(obs_dim, act_dim, chunk, hidden, n_heads, n_layers)
            self._layer = self._policy

    def get_layers(self) -> list:
        return [self._layer] if self._layer is not None else []

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        obs_dim  = int(inputs.get("observation_dim") or 6)
        act_dim  = int(inputs.get("action_dim") or 6)
        chunk    = int(inputs.get("chunk_size") or 100)
        hidden   = int(inputs.get("hidden_dim") or 512)
        n_heads  = int(inputs.get("n_heads") or 8)
        n_layers = int(inputs.get("n_layers") or 4)
        mode     = str(inputs.get("mode") or "train").lower()
        obs      = inputs.get("observation")
        action   = inputs.get("action")

        empty = {"predicted_action": None, "loss": None, "model": None, "info": ""}

        self._ensure_policy(obs_dim, act_dim, chunk, hidden, n_heads, n_layers)

        if self._policy is None:
            return {**empty, "info": "Failed to build ACT policy"}

        empty["model"] = self._policy

        if obs is None:
            n_params = sum(p.numel() for p in self._policy.parameters())
            return {**empty, "info": f"ACT policy: {n_params:,} params. Connect observation."}

        import torch
        try:
            predicted = self._policy(obs)
            n_params = sum(p.numel() for p in self._policy.parameters())

            if mode == "train" and action is not None:
                # Compute MSE loss between predicted and target actions
                import torch.nn.functional as F
                loss = F.mse_loss(predicted, action)
                return {
                    "predicted_action": predicted,
                    "loss": loss,
                    "model": self._policy,
                    "info": f"ACT training: loss={loss.item():.6f} ({n_params:,} params)",
                }
            else:
                return {
                    "predicted_action": predicted,
                    "loss": None,
                    "model": self._policy,
                    "info": f"ACT inference: chunk={chunk}x{act_dim} ({n_params:,} params)",
                }
        except Exception as exc:
            return {**empty, "info": f"Forward failed: {exc}"}

    def export(self, iv, ov):
        return ["import torch", "import torch.nn as nn"], [
            f"# ACT Policy — see lerobot for the full implementation",
            f"# pip install lerobot",
        ]
