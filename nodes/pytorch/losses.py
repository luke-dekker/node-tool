"""Re-export shim — LossFnNode is the consolidated loss factory."""
from nodes.pytorch.mse_loss import LossFnNode

__all__ = ["LossFnNode"]
