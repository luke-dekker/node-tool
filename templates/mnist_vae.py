"""MNIST Variational Autoencoder — built from per-layer nodes.

Demonstrates:

  - Building a VAE entirely from explicit graph nodes (no monolithic VAENode)
  - The reparameterization trick as a real graph node
  - Computing loss INSIDE the graph via LossComputeNode + KLDivergenceNode +
    VAELossNode → fed into TrainingConfig with loss_is_output=True
  - Multi-output models: encoder splits into mu and log_var Linear heads,
    both visible in the canvas

Architecture (top to bottom):

    MNIST → BatchInput
              └─> x ─> Flatten ─> Linear+ReLU ─> Linear+ReLU
                                                    ├─> Linear (mu head)
                                                    └─> Linear (log_var head)
              mu, log_var ─> Reparameterize ─> z
                          ↓
                    z ─> Linear+ReLU ─> Linear+ReLU ─> Linear+Sigmoid ─> recon

    LossCompute(recon, original_x, type='bce')   ─> recon_loss
    KLDivergence(mu, log_var)                    ─> kl_loss
    VAELoss(recon_loss, kl_loss, beta=1.0)       ─> total_loss (scalar)

    total_loss ─> TrainingConfig(loss_is_output=True).tensor_in

The graph IS the model AND the loss. Training just calls
`loss = model(batch); loss.backward()` — no special VAE branch in the loop.
This pattern generalizes to multi-task: more LossCompute nodes summed via
math nodes or a future LossCombiner.
"""
from __future__ import annotations
from core.graph import Graph
from templates._helpers import grid

LABEL = "MNIST VAE (Image Generator)"
DESCRIPTION = (
    "Variational autoencoder built from per-layer nodes. Loss is computed "
    "in the graph via LossCompute + KLDivergence + VAELoss, then fed into "
    "TrainingConfig with loss_is_output=True."
)


def build(graph: Graph) -> dict[str, tuple[int, int]]:
    from nodes.pytorch.mnist_dataset    import MNISTDatasetNode
    from nodes.pytorch.batch_input      import BatchInputNode
    from nodes.pytorch.flatten          import FlattenNode
    from nodes.pytorch.linear           import LinearNode
    from nodes.pytorch.reparameterize   import ReparameterizeNode
    from nodes.pytorch.kl_divergence    import KLDivergenceNode
    from nodes.pytorch.loss_compute     import LossComputeNode
    from nodes.pytorch.vae_loss         import VAELossNode
    from nodes.pytorch.training_config  import TrainingConfigNode

    pos = grid(step_x=200)
    positions: dict[str, tuple[int, int]] = {}

    LATENT = 16

    # ── Data ────────────────────────────────────────────────────────────────
    mnist = MNISTDatasetNode()
    mnist.inputs["batch_size"].default_value = 128
    graph.add_node(mnist); positions[mnist.id] = pos(col=0, row=2)

    batch = BatchInputNode()
    graph.add_node(batch); positions[batch.id] = pos(col=1, row=2)

    flat = FlattenNode()
    flat.inputs["start_dim"].default_value = 1
    graph.add_node(flat); positions[flat.id] = pos(col=2, row=2)

    # ── Encoder body: 784 → 256 → 64 ────────────────────────────────────────
    enc1 = LinearNode()
    enc1.inputs["in_features"].default_value  = 784
    enc1.inputs["out_features"].default_value = 256
    enc1.inputs["activation"].default_value   = "relu"
    graph.add_node(enc1); positions[enc1.id] = pos(col=3, row=2)

    enc2 = LinearNode()
    enc2.inputs["in_features"].default_value  = 256
    enc2.inputs["out_features"].default_value = 64
    enc2.inputs["activation"].default_value   = "relu"
    graph.add_node(enc2); positions[enc2.id] = pos(col=4, row=2)

    # ── Two heads: mu and log_var ───────────────────────────────────────────
    mu_head = LinearNode()
    mu_head.inputs["in_features"].default_value  = 64
    mu_head.inputs["out_features"].default_value = LATENT
    mu_head.inputs["activation"].default_value   = "none"
    graph.add_node(mu_head); positions[mu_head.id] = pos(col=5, row=1)

    lv_head = LinearNode()
    lv_head.inputs["in_features"].default_value  = 64
    lv_head.inputs["out_features"].default_value = LATENT
    lv_head.inputs["activation"].default_value   = "none"
    graph.add_node(lv_head); positions[lv_head.id] = pos(col=5, row=3)

    # ── Reparameterize ──────────────────────────────────────────────────────
    repar = ReparameterizeNode()
    graph.add_node(repar); positions[repar.id] = pos(col=6, row=2)

    # ── Decoder: 16 → 64 → 256 → 784 ────────────────────────────────────────
    dec1 = LinearNode()
    dec1.inputs["in_features"].default_value  = LATENT
    dec1.inputs["out_features"].default_value = 64
    dec1.inputs["activation"].default_value   = "relu"
    graph.add_node(dec1); positions[dec1.id] = pos(col=7, row=2)

    dec2 = LinearNode()
    dec2.inputs["in_features"].default_value  = 64
    dec2.inputs["out_features"].default_value = 256
    dec2.inputs["activation"].default_value   = "relu"
    graph.add_node(dec2); positions[dec2.id] = pos(col=8, row=2)

    dec3 = LinearNode()
    dec3.inputs["in_features"].default_value  = 256
    dec3.inputs["out_features"].default_value = 784
    dec3.inputs["activation"].default_value   = "sigmoid"   # outputs in [0, 1]
    graph.add_node(dec3); positions[dec3.id] = pos(col=9, row=2)

    # ── Loss: BCE(recon, x) + beta * KL(mu, log_var) ────────────────────────
    recon_loss = LossComputeNode()
    recon_loss.inputs["loss_type"].default_value = "bce"
    recon_loss.inputs["weight"].default_value    = 1.0
    graph.add_node(recon_loss); positions[recon_loss.id] = pos(col=10, row=1)

    kl = KLDivergenceNode()
    graph.add_node(kl); positions[kl.id] = pos(col=10, row=3)

    vae_loss = VAELossNode()
    vae_loss.inputs["beta"].default_value = 1.0
    graph.add_node(vae_loss); positions[vae_loss.id] = pos(col=11, row=2)

    # ── Training config (loss-as-output mode) ───────────────────────────────
    cfg = TrainingConfigNode()
    cfg.inputs["epochs"].default_value         = 10
    cfg.inputs["lr"].default_value             = 0.001
    cfg.inputs["optimizer"].default_value      = "adam"
    cfg.inputs["loss_is_output"].default_value = True
    graph.add_node(cfg); positions[cfg.id] = pos(col=12, row=2)

    # ── Wire it ─────────────────────────────────────────────────────────────
    # Data path
    graph.add_connection(mnist.id, "dataloader", batch.id, "dataloader")
    graph.add_connection(mnist.id, "dataloader", cfg.id,   "dataloader")
    graph.add_connection(batch.id, "x",          flat.id,  "tensor_in")

    # Encoder body
    graph.add_connection(flat.id, "tensor_out", enc1.id, "tensor_in")
    graph.add_connection(enc1.id, "tensor_out", enc2.id, "tensor_in")

    # Heads (mu, log_var) both fed from enc2 output
    graph.add_connection(enc2.id, "tensor_out", mu_head.id, "tensor_in")
    graph.add_connection(enc2.id, "tensor_out", lv_head.id, "tensor_in")

    # Reparameterize
    graph.add_connection(mu_head.id, "tensor_out", repar.id, "mu")
    graph.add_connection(lv_head.id, "tensor_out", repar.id, "log_var")

    # Decoder
    graph.add_connection(repar.id, "z",          dec1.id, "tensor_in")
    graph.add_connection(dec1.id,  "tensor_out", dec2.id, "tensor_in")
    graph.add_connection(dec2.id,  "tensor_out", dec3.id, "tensor_in")

    # Loss assembly
    # Recon loss compares decoder output to the FLATTENED input (same as BCE expects)
    graph.add_connection(dec3.id,      "tensor_out", recon_loss.id, "pred")
    graph.add_connection(flat.id,      "tensor_out", recon_loss.id, "target")
    # KL on mu / log_var
    graph.add_connection(mu_head.id,   "tensor_out", kl.id,         "mu")
    graph.add_connection(lv_head.id,   "tensor_out", kl.id,         "log_var")
    # Combine
    graph.add_connection(recon_loss.id, "loss",     vae_loss.id, "recon_loss")
    graph.add_connection(kl.id,        "kl_loss",   vae_loss.id, "kl_loss")
    # Final scalar loss into the training config (loss_is_output=True)
    graph.add_connection(vae_loss.id,  "loss",      cfg.id,      "tensor_in")
    return positions
