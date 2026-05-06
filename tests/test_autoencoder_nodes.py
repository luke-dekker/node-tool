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
    from nodes.pytorch.layer import LayerNode
    import torch
    node = LayerNode()
    node.execute({"tensor_in": torch.randn(2, 4), "kind": "linear",
                  "out_features": 2, "bias": True, "activation": "none", "freeze": True})
    assert all(not p.requires_grad for p in node._layer.parameters())

def test_linear_no_freeze():
    from nodes.pytorch.layer import LayerNode
    import torch
    node = LayerNode()
    node.execute({"tensor_in": torch.randn(2, 4), "kind": "linear",
                  "out_features": 2, "bias": True, "activation": "none", "freeze": False})
    assert all(p.requires_grad for p in node._layer.parameters())

def test_conv2d_freeze():
    from nodes.pytorch.layer import LayerNode
    import torch
    node = LayerNode()
    node.execute({"tensor_in": torch.randn(1, 1, 8, 8), "kind": "conv2d",
                  "out_ch": 16, "kernel": 3, "stride": 1, "padding": 0,
                  "activation": "none", "freeze": True})
    assert all(not p.requires_grad for p in node._layer.parameters())

def test_rnn_freeze():
    from nodes.pytorch.recurrent_layer import RecurrentLayerNode
    import torch
    result = RecurrentLayerNode().execute({
        "kind": "rnn", "input_seq": torch.randn(1, 1, 8),
        "hidden_size": 16, "num_layers": 1, "nonlinearity": "tanh",
        "dropout": 0.0, "bidirectional": False, "batch_first": True, "freeze": True,
    })
    assert all(not p.requires_grad for p in result["module"].parameters())

def test_gru_freeze():
    from nodes.pytorch.recurrent_layer import RecurrentLayerNode
    import torch
    result = RecurrentLayerNode().execute({
        "kind": "gru", "input_seq": torch.randn(1, 1, 8),
        "hidden_size": 16, "num_layers": 1, "dropout": 0.0,
        "bidirectional": False, "batch_first": True, "freeze": True,
    })
    assert all(not p.requires_grad for p in result["module"].parameters())

def test_lstm_freeze():
    from nodes.pytorch.recurrent_layer import RecurrentLayerNode
    import torch
    result = RecurrentLayerNode().execute({
        "kind": "lstm", "input_seq": torch.randn(1, 1, 8),
        "hidden_size": 16, "num_layers": 1, "dropout": 0.0,
        "bidirectional": False, "batch_first": True, "freeze": True,
    })
    assert all(not p.requires_grad for p in result["module"].parameters())

def test_freeze_named_layers():
    from nodes.pytorch.backbones import ResNet18Node
    from nodes.pytorch.freeze_backbone import FreezeLayersNode
    model = ResNet18Node().execute({"pretrained": False, "num_classes": 10})["model"]
    result = FreezeLayersNode().execute({
        "model": model, "mode": "by_name", "names": "layer1,layer2", "freeze": True,
    })
    assert result["model"] is not None
    assert "frozen" in result["info"].lower()

def test_freeze_named_layers_none_guard():
    from nodes.pytorch.freeze_backbone import FreezeLayersNode
    result = FreezeLayersNode().execute({
        "model": None, "mode": "by_name", "names": "encoder", "freeze": True,
    })
    assert result["model"] is None


# ── Reparameterize ────────────────────────────────────────────────────────────

def test_reparameterize():
    from nodes.pytorch.latent import LatentNode
    mu      = torch.zeros(4, 8); log_var = torch.zeros(4, 8)
    result  = LatentNode().execute({"mode": "reparameterize", "mean": mu, "log_variance": log_var})
    z = result["z"]
    assert z is not None and z.shape == (4, 8)
    assert not torch.all(z == 0)

def test_reparameterize_none_guard():
    from nodes.pytorch.latent import LatentNode
    result = LatentNode().execute({"mode": "reparameterize", "mean": None, "log_variance": None})
    assert result["z"] is None


# ── KL Divergence (now mode="kl" on VAELossNode) ───────────────────────────

def test_kl_divergence():
    from nodes.pytorch.vae_loss import VAELossNode
    mu = torch.zeros(4, 8); log_var = torch.zeros(4, 8)
    result = VAELossNode().execute({"mode": "kl", "mean": mu, "log_variance": log_var})
    assert result["kl_loss"] is not None
    assert abs(result["kl_loss"].item()) < 1e-5  # KL(N(0,1)||N(0,1)) = 0

def test_kl_divergence_nonzero():
    from nodes.pytorch.vae_loss import VAELossNode
    mu = torch.ones(4, 8) * 2.0; log_var = torch.ones(4, 8) * 0.5
    result = VAELossNode().execute({"mode": "kl", "mean": mu, "log_variance": log_var})
    assert result["kl_loss"].item() > 0

def test_kl_none_guard():
    from nodes.pytorch.vae_loss import VAELossNode
    result = VAELossNode().execute({"mode": "kl", "mean": None, "log_variance": None})
    assert result["kl_loss"] is None


# ── VAE Loss combiner ────────────────────────────────────────────────────────

def test_vae_loss_combines():
    from nodes.pytorch.vae_loss import VAELossNode
    recon = torch.tensor(0.5); kl = torch.tensor(0.1)
    result = VAELossNode().execute({"mode": "combine", "recon_loss": recon, "kl_loss": kl, "beta": 1.0})
    assert abs(result["loss"].item() - 0.6) < 1e-5

def test_vae_loss_beta():
    from nodes.pytorch.vae_loss import VAELossNode
    recon = torch.tensor(0.5); kl = torch.tensor(0.1)
    result = VAELossNode().execute({"mode": "combine", "recon_loss": recon, "kl_loss": kl, "beta": 4.0})
    assert abs(result["loss"].item() - 0.9) < 1e-5

def test_vae_loss_none_guard():
    from nodes.pytorch.vae_loss import VAELossNode
    result = VAELossNode().execute({"mode": "combine", "recon_loss": None, "kl_loss": None, "beta": 1.0})
    assert result["loss"] is None


# ── Latent Sampler & Gaussian Noise ──────────────────────────────────────────

def test_latent_sampler():
    import torch.nn as nn
    from nodes.pytorch.latent import LatentNode
    decoder = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 32))
    result = LatentNode().execute({
        "mode": "sample", "decoder": decoder, "latent_dim": 4, "n_samples": 6, "device": "cpu",
    })
    assert result["samples"] is not None
    assert result["samples"].shape == (6, 32)

def test_latent_sampler_none_guard():
    from nodes.pytorch.latent import LatentNode
    result = LatentNode().execute({"mode": "sample", "decoder": None,
                                    "latent_dim": 8, "n_samples": 4, "device": "cpu"})
    assert result["samples"] is None

def test_gate_noise():
    from nodes.pytorch.gate import GateNode
    t = torch.ones(4, 8) * 0.5
    result = GateNode().execute({"tensor_in": t, "mode": "noise", "noise_std": 0.1})
    noisy = result["tensor_out"]
    assert noisy is not None
    assert noisy.shape == t.shape
    assert not torch.all(noisy == t)

def test_gate_none_guard():
    from nodes.pytorch.gate import GateNode
    result = GateNode().execute({"tensor_in": None, "mode": "pass"})
    assert result["tensor_out"] is None


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
    # Reparameterize + LatentSampler collapsed into pt_latent (LatentNode).
    # KLDivergence collapsed into pt_vae_loss (mode="kl").
    # FreezeNamedLayers collapsed into pt_freeze_backbone (mode="by_name").
    expected = ["pt_latent", "pt_vae_loss",
                "pt_freeze_backbone",
                "pt_loss_compute", "pt_gate",
                "pt_input_marker", "pt_train_marker"]
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
