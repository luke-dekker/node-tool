"""Re-export shim — autoencoder building blocks (per-layer composition).

VAENode and AutoencoderNode were hard-deleted: those monolithic nodes
violated "graph IS the model" by hiding nn.Sequential and an inline class
inside a single node. Build VAEs and AEs from explicit per-layer node
chains in templates instead. The pieces below are the per-layer primitives
the templates compose: reparameterization, KL, sampler, etc.
"""
from nodes.pytorch.reparameterize import ReparameterizeNode
from nodes.pytorch.kl_divergence import KLDivergenceNode
from nodes.pytorch.vae_loss import VAELossNode
from nodes.pytorch.latent_sampler import LatentSamplerNode
from nodes.pytorch.gaussian_noise import GaussianNoiseNode

__all__ = [
    "ReparameterizeNode", "KLDivergenceNode", "VAELossNode",
    "LatentSamplerNode", "GaussianNoiseNode",
]
