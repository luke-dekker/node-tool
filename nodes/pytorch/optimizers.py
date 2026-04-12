"""Re-export shim — OptimizerNode is the consolidated optimizer factory."""
from nodes.pytorch.adam import OptimizerNode

__all__ = ["OptimizerNode"]
