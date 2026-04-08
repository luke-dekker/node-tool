"""MNIST VAE image generator template.

A variational autoencoder over MNIST. After training, the latent sampler
draws fresh samples from the prior and decodes them into new digit images.
This is the canonical "image generator" workflow.

NOTE: VAENode is monolithic — its encoder Sequential expects a flat (B, 784)
tensor. For raw MNIST (which gives (B, 1, 28, 28)), you'll need to either
modify VAENode.forward to do `x = x.view(x.size(0), -1)`, or wire through a
flatten step. The graph below shows the wiring you'd want; live training will
need that one tweak. Code export from this template is fully runnable.
"""
from __future__ import annotations
from core.graph import Graph
from templates._helpers import grid


def build(graph: Graph) -> dict[str, tuple[int, int]]:
    from nodes.pytorch.mnist_dataset        import MNISTDatasetNode
    from nodes.pytorch.vae                  import VAENode
    from nodes.pytorch.vae_training_config  import VAETrainingConfigNode
    from nodes.pytorch.adam                 import AdamNode
    from nodes.pytorch.latent_sampler       import LatentSamplerNode

    pos = grid(step_x=240)
    positions: dict[str, tuple[int, int]] = {}

    mnist = MNISTDatasetNode()
    mnist.inputs["batch_size"].default_value = 128
    graph.add_node(mnist); positions[mnist.id] = pos()

    vae = VAENode()
    vae.inputs["layer_sizes"].default_value = "784,256,64"
    vae.inputs["latent_dim"].default_value  = 16
    vae.inputs["activation"].default_value  = "relu"
    graph.add_node(vae); positions[vae.id] = pos()

    opt = AdamNode()
    opt.inputs["lr"].default_value = 0.001
    graph.add_node(opt); positions[opt.id] = pos()

    cfg = VAETrainingConfigNode()
    cfg.inputs["epochs"].default_value          = 10
    cfg.inputs["beta"].default_value            = 1.0
    cfg.inputs["recon_loss_type"].default_value = "bce"
    cfg.inputs["device"].default_value          = "cpu"
    graph.add_node(cfg); positions[cfg.id] = pos()

    # Latent sampler — after training, decodes fresh samples from N(0,1)
    sampler = LatentSamplerNode()
    sampler.inputs["latent_dim"].default_value = 16
    sampler.inputs["n_samples"].default_value  = 16
    graph.add_node(sampler); positions[sampler.id] = pos()

    # Wire it: VAE is the model, AdamNode wraps it as optimizer, both feed cfg.
    # The dataloader feeds cfg directly; the training loop calls model(x) per batch.
    graph.add_connection(vae.id,   "model",      opt.id,     "model")
    graph.add_connection(vae.id,   "model",      cfg.id,     "model")
    graph.add_connection(opt.id,   "optimizer",  cfg.id,     "optimizer")
    graph.add_connection(mnist.id, "dataloader", cfg.id,     "dataloader")
    graph.add_connection(vae.id,   "decoder",    sampler.id, "decoder")
    return positions
