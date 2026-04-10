"""Tests for autoencoder/VAE building blocks and freeze functionality.

VAENode and AutoencoderNode were hard-deleted in favor of building VAEs and
AEs from per-layer node chains in templates. This file tests the per-layer
primitives (Reparameterize, KL, VAELoss, LatentSampler, GaussianNoise) plus
the freeze controls and the new LossComputeNode.
"""
import pytest
import torch


# ── Freeze parameter on layer nodes ──────────────────────────────────────────

def test_linear_freeze():
    from nodes.pytorch.layers import LinearNode
    node = LinearNode()
    node.execute({"tensor_in": None, "in_features": 4, "out_features": 2,
                  "bias": True, "activation": "none", "freeze": True})
    assert all(not p.requires_grad for p in node._layer.parameters())

def test_linear_no_freeze():
    from nodes.pytorch.layers import LinearNode
    node = LinearNode()
    node.execute({"tensor_in": None, "in_features": 4, "out_features": 2,
                  "bias": True, "activation": "none", "freeze": False})
    assert all(p.requires_grad for p in node._layer.parameters())

def test_conv2d_freeze():
    from nodes.pytorch.layers import Conv2dNode
    node = Conv2dNode()
    node.execute({"tensor_in": None, "in_ch": 1, "out_ch": 16, "kernel": 3,
                  "stride": 1, "padding": 0, "activation": "none", "freeze": True})
    assert all(not p.requires_grad for p in node._layer.parameters())

def test_rnn_freeze():
    from nodes.pytorch.recurrent import RNNLayerNode
    result = RNNLayerNode().execute({"input_size": 8, "hidden_size": 16, "num_layers": 1,
                                      "nonlinearity": "tanh", "dropout": 0.0,
                                      "bidirectional": False, "batch_first": True, "freeze": True})
    assert all(not p.requires_grad for p in result["module"].parameters())

def test_gru_freeze():
    from nodes.pytorch.recurrent import GRULayerNode
    result = GRULayerNode().execute({"input_size": 8, "hidden_size": 16, "num_layers": 1,
                                      "dropout": 0.0, "bidirectional": False,
                                      "batch_first": True, "freeze": True})
    assert all(not p.requires_grad for p in result["module"].parameters())

def test_lstm_freeze():
    from nodes.pytorch.recurrent import LSTMLayerNode
    result = LSTMLayerNode().execute({"input_size": 8, "hidden_size": 16, "num_layers": 1,
                                       "dropout": 0.0, "bidirectional": False,
                                       "batch_first": True, "freeze": True})
    assert all(not p.requires_grad for p in result["module"].parameters())

def test_freeze_named_layers():
    from nodes.pytorch.backbones import ResNet18Node, FreezeNamedLayersNode
    model = ResNet18Node().execute({"pretrained": False, "num_classes": 10})["model"]
    result = FreezeNamedLayersNode().execute({"model": model, "names": "layer1,layer2", "freeze": True})
    assert result["model"] is not None
    assert "Frozen" in result["info"]

def test_freeze_named_layers_none_guard():
    from nodes.pytorch.backbones import FreezeNamedLayersNode
    result = FreezeNamedLayersNode().execute({"model": None, "names": "encoder", "freeze": True})
    assert result["model"] is None


# ── Reparameterize ────────────────────────────────────────────────────────────

def test_reparameterize():
    from nodes.pytorch.autoencoder import ReparameterizeNode
    mu      = torch.zeros(4, 8)
    log_var = torch.zeros(4, 8)
    result  = ReparameterizeNode().execute({"mu": mu, "log_var": log_var})
    z = result["z"]
    assert z is not None
    assert z.shape == (4, 8)
    # z should NOT be all zeros (sampling adds noise)
    assert not torch.all(z == 0)

def test_reparameterize_none_guard():
    from nodes.pytorch.autoencoder import ReparameterizeNode
    result = ReparameterizeNode().execute({"mu": None, "log_var": None})
    assert result["z"] is None


# ── KL Divergence ─────────────────────────────────────────────────────────────

def test_kl_divergence():
    from nodes.pytorch.autoencoder import KLDivergenceNode
    mu      = torch.zeros(4, 8)
    log_var = torch.zeros(4, 8)
    result  = KLDivergenceNode().execute({"mu": mu, "log_var": log_var})
    assert result["kl_loss"] is not None
    # KL(N(0,1) || N(0,1)) = 0
    assert abs(result["kl_loss"].item()) < 1e-5

def test_kl_divergence_nonzero():
    from nodes.pytorch.autoencoder import KLDivergenceNode
    mu      = torch.ones(4, 8) * 2.0
    log_var = torch.ones(4, 8) * 0.5
    result  = KLDivergenceNode().execute({"mu": mu, "log_var": log_var})
    assert result["kl_loss"].item() > 0

def test_kl_none_guard():
    from nodes.pytorch.autoencoder import KLDivergenceNode
    result = KLDivergenceNode().execute({"mu": None, "log_var": None})
    assert result["kl_loss"] is None


# ── VAE Loss combiner ────────────────────────────────────────────────────────

def test_vae_loss_combines():
    from nodes.pytorch.autoencoder import VAELossNode
    recon = torch.tensor(0.5)
    kl    = torch.tensor(0.1)
    result = VAELossNode().execute({"recon_loss": recon, "kl_loss": kl, "beta": 1.0})
    assert abs(result["loss"].item() - 0.6) < 1e-5

def test_vae_loss_beta():
    from nodes.pytorch.autoencoder import VAELossNode
    recon = torch.tensor(0.5)
    kl    = torch.tensor(0.1)
    result = VAELossNode().execute({"recon_loss": recon, "kl_loss": kl, "beta": 4.0})
    assert abs(result["loss"].item() - 0.9) < 1e-5

def test_vae_loss_none_guard():
    from nodes.pytorch.autoencoder import VAELossNode
    result = VAELossNode().execute({"recon_loss": None, "kl_loss": None, "beta": 1.0})
    assert result["loss"] is None


# ── Latent Sampler & Gaussian Noise ──────────────────────────────────────────

def test_latent_sampler():
    """Use a simple Sequential as the decoder — not the deleted VAENode."""
    import torch.nn as nn
    from nodes.pytorch.autoencoder import LatentSamplerNode
    decoder = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 32))
    result = LatentSamplerNode().execute({
        "decoder": decoder, "latent_dim": 4, "n_samples": 6, "device": "cpu"
    })
    assert result["samples"] is not None
    assert result["samples"].shape[0] == 6
    assert result["samples"].shape[1] == 32

def test_latent_sampler_none_guard():
    from nodes.pytorch.autoencoder import LatentSamplerNode
    result = LatentSamplerNode().execute({"decoder": None, "latent_dim": 8, "n_samples": 4, "device": "cpu"})
    assert result["samples"] is None

def test_gaussian_noise():
    from nodes.pytorch.autoencoder import GaussianNoiseNode
    t = torch.ones(4, 8) * 0.5
    result = GaussianNoiseNode().execute({"tensor": t, "std": 0.1, "clip": True})
    noisy = result["tensor"]
    assert noisy is not None
    assert noisy.shape == t.shape
    assert not torch.all(noisy == t)  # noise was added
    assert noisy.min() >= 0.0 and noisy.max() <= 1.0  # clipped

def test_gaussian_noise_none_guard():
    from nodes.pytorch.autoencoder import GaussianNoiseNode
    result = GaussianNoiseNode().execute({"tensor": None, "std": 0.1, "clip": True})
    assert result["tensor"] is None


# ── LossComputeNode (the new generic in-graph loss node) ─────────────────────

def test_loss_compute_mse():
    from nodes.pytorch.loss_compute import LossComputeNode
    pred   = torch.zeros(4, 8)
    target = torch.ones(4, 8)
    result = LossComputeNode().execute({
        "pred": pred, "target": target, "loss_type": "mse", "weight": 1.0,
    })
    # MSE between zeros and ones is 1.0
    assert abs(result["loss"].item() - 1.0) < 1e-5

def test_loss_compute_weight():
    from nodes.pytorch.loss_compute import LossComputeNode
    pred   = torch.zeros(4, 8)
    target = torch.ones(4, 8)
    result = LossComputeNode().execute({
        "pred": pred, "target": target, "loss_type": "mse", "weight": 0.5,
    })
    # 0.5 * 1.0 = 0.5
    assert abs(result["loss"].item() - 0.5) < 1e-5

def test_loss_compute_cross_entropy():
    from nodes.pytorch.loss_compute import LossComputeNode
    # 4 samples, 10 classes — uniform logits should give loss ≈ ln(10) ≈ 2.30
    pred   = torch.zeros(4, 10)
    target = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    result = LossComputeNode().execute({
        "pred": pred, "target": target, "loss_type": "cross_entropy", "weight": 1.0,
    })
    import math
    assert abs(result["loss"].item() - math.log(10)) < 1e-3

def test_loss_compute_bce():
    from nodes.pytorch.loss_compute import LossComputeNode
    pred   = torch.full((4, 8), 0.5)
    target = torch.full((4, 8), 0.5)
    result = LossComputeNode().execute({
        "pred": pred, "target": target, "loss_type": "bce", "weight": 1.0,
    })
    # BCE(0.5, 0.5) = -log(0.5) = ln(2)
    import math
    assert abs(result["loss"].item() - math.log(2)) < 1e-3

def test_loss_compute_none_guard():
    from nodes.pytorch.loss_compute import LossComputeNode
    result = LossComputeNode().execute({
        "pred": None, "target": None, "loss_type": "mse", "weight": 1.0,
    })
    assert result["loss"] is None

def test_loss_compute_in_registry():
    from nodes import NODE_REGISTRY
    assert "pt_loss_compute" in NODE_REGISTRY


# ── Registry ──────────────────────────────────────────────────────────────────

def test_registry_has_per_layer_blocks():
    """The per-layer building blocks should still be registered after the
    monolithic VAENode/AutoencoderNode/VAETrainingConfigNode hard-delete."""
    from nodes import NODE_REGISTRY
    expected = ["pt_reparameterize", "pt_kl_divergence", "pt_vae_loss",
                "pt_latent_sampler", "pt_freeze_named_layers",
                "pt_loss_compute", "pt_gate", "pt_train_output", "pt_dataset"]
    for tn in expected:
        assert tn in NODE_REGISTRY, f"Missing: {tn}"

def test_registry_does_not_have_deleted_nodes():
    """Obsolete nodes should not appear in the registry."""
    from nodes import NODE_REGISTRY
    obsolete = ("pt_vae", "pt_autoencoder", "pt_vae_training_config",
                "batch_input", "pt_multi_dataset", "pt_multimodal_model",
                "pt_gaussian_noise", "pt_training_config",
                "pt_multimodal_training_config")
    for tn in obsolete:
        assert tn not in NODE_REGISTRY, f"Should be removed: {tn}"
