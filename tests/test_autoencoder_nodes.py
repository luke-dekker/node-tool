"""Tests for autoencoder/VAE nodes and freeze functionality."""
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


# ── Autoencoder ───────────────────────────────────────────────────────────────

def test_autoencoder_creates_model():
    from nodes.pytorch.autoencoder import AutoencoderNode
    result = AutoencoderNode().execute({"layer_sizes": "64,32,16", "activation": "relu"})
    assert result["model"] is not None
    assert result["latent_dim"] == 16
    assert result["encoder"] is not None
    assert result["decoder"] is not None
    assert "params=" in result["info"]

def test_autoencoder_forward():
    from nodes.pytorch.autoencoder import AutoencoderNode
    result = AutoencoderNode().execute({"layer_sizes": "64,32,16", "activation": "relu"})
    model = result["model"]
    x = torch.randn(4, 64)
    out = model(x)
    assert out.shape == (4, 64)  # reconstruction same shape as input

def test_autoencoder_different_activations():
    from nodes.pytorch.autoencoder import AutoencoderNode
    for act in ["relu", "tanh", "sigmoid"]:
        result = AutoencoderNode().execute({"layer_sizes": "32,16,8", "activation": act})
        assert result["model"] is not None

def test_autoencoder_none_guard():
    from nodes.pytorch.autoencoder import AutoencoderNode
    result = AutoencoderNode().execute({"layer_sizes": "not,valid", "activation": "relu"})
    assert result["model"] is None


# ── VAE ───────────────────────────────────────────────────────────────────────

def test_vae_creates_model():
    from nodes.pytorch.autoencoder import VAENode
    result = VAENode().execute({"layer_sizes": "64,32", "latent_dim": 8, "activation": "relu"})
    assert result["model"] is not None
    assert "params=" in result["info"]

def test_vae_forward_returns_triple():
    from nodes.pytorch.autoencoder import VAENode
    result = VAENode().execute({"layer_sizes": "64,32", "latent_dim": 8, "activation": "relu"})
    model = result["model"]
    x = torch.randn(4, 64)
    out = model(x)
    assert isinstance(out, tuple) and len(out) == 3
    recon, mu, log_var = out
    assert recon.shape == (4, 64)
    assert mu.shape == (4, 8)
    assert log_var.shape == (4, 8)

def test_vae_encode_decode():
    from nodes.pytorch.autoencoder import VAENode
    result = VAENode().execute({"layer_sizes": "64,32", "latent_dim": 8, "activation": "relu"})
    model = result["model"]
    x = torch.randn(2, 64)
    mu, log_var = model.encode(x)
    assert mu.shape == (2, 8)
    z = torch.randn(2, 8)
    recon = model.decode(z)
    assert recon.shape == (2, 64)

def test_vae_sample():
    from nodes.pytorch.autoencoder import VAENode
    result = VAENode().execute({"layer_sizes": "64,32", "latent_dim": 8, "activation": "relu"})
    model = result["model"]
    samples = model.sample(5)
    assert samples.shape == (5, 64)

def test_vae_none_guard():
    from nodes.pytorch.autoencoder import VAENode
    result = VAENode().execute({"layer_sizes": "bad", "latent_dim": 8, "activation": "relu"})
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


# ── VAE Loss ──────────────────────────────────────────────────────────────────

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


# ── VAE Training Config ───────────────────────────────────────────────────────

def test_vae_training_config():
    from nodes.pytorch.autoencoder import VAETrainingConfigNode
    result = VAETrainingConfigNode().execute({
        "model": None, "optimizer": None, "dataloader": None,
        "val_dataloader": None, "scheduler": None,
        "epochs": 10, "beta": 2.0, "recon_loss_type": "mse", "device": "cpu"
    })
    cfg = result["config"]
    assert cfg is not None
    assert cfg["vae"] is True
    assert cfg["beta"] == 2.0
    assert cfg["epochs"] == 10


# ── Latent Sampler & Gaussian Noise ──────────────────────────────────────────

def test_latent_sampler():
    from nodes.pytorch.autoencoder import VAENode, LatentSamplerNode
    model = VAENode().execute({"layer_sizes": "32,16", "latent_dim": 4, "activation": "relu"})["model"]
    result = LatentSamplerNode().execute({
        "decoder": model.decoder, "latent_dim": 4, "n_samples": 6, "device": "cpu"
    })
    assert result["samples"] is not None
    assert result["samples"].shape[0] == 6

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


# ── Registry ──────────────────────────────────────────────────────────────────

def test_registry_has_autoencoder_nodes():
    from nodes import NODE_REGISTRY
    expected = ["pt_autoencoder", "pt_vae", "pt_reparameterize", "pt_kl_divergence",
                "pt_vae_loss", "pt_vae_training_config", "pt_latent_sampler",
                "pt_gaussian_noise", "pt_freeze_named_layers"]
    for tn in expected:
        assert tn in NODE_REGISTRY, f"Missing: {tn}"

def test_generative_subcategory():
    from nodes import NODE_REGISTRY
    for tn in ["pt_autoencoder", "pt_vae", "pt_reparameterize"]:
        assert NODE_REGISTRY[tn].subcategory == "Autoencoder", f"{tn} wrong subcategory"
