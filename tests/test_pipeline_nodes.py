"""Tests for layer nodes - tensor forward-pass and persistent state."""
import pytest
torch = pytest.importorskip("torch")
import torch.nn as nn

from nodes.pytorch.layers import (
    FlattenNode, LinearNode, Conv2dNode,
    BatchNorm1dNode, BatchNorm2dNode,
    DropoutNode, MaxPool2dNode, AvgPool2dNode,
    ActivationNode, EmbeddingNode,
)


def _exec(node, **kw):
    return node.execute(kw)


# ── Tensor forward-pass ───────────────────────────────────────────────────────

def test_flatten_output_shape():
    flat = FlattenNode()
    out = _exec(flat, tensor_in=torch.randn(1, 1, 28, 28), start_dim=1)["tensor_out"]
    assert out.shape == (1, 784)

def test_flatten_none_input_returns_none():
    out = _exec(FlattenNode(), tensor_in=None, start_dim=1)["tensor_out"]
    assert out is None

def test_linear_output_shape():
    lin = LinearNode()
    out = _exec(lin, tensor_in=torch.randn(1, 784),
                in_features=784, out_features=256, bias=True,
                activation="none", freeze=False)["tensor_out"]
    assert out.shape == (1, 256)

def test_linear_relu_activation():
    lin = LinearNode()
    out = _exec(lin, tensor_in=torch.randn(2, 4),
                in_features=4, out_features=8, bias=True,
                activation="relu", freeze=False)["tensor_out"]
    assert out.shape == (2, 8)
    assert (out >= 0).all()

def test_linear_sigmoid_activation():
    lin = LinearNode()
    out = _exec(lin, tensor_in=torch.randn(2, 4),
                in_features=4, out_features=4, bias=False,
                activation="sigmoid", freeze=False)["tensor_out"]
    assert (out > 0).all() and (out < 1).all()

def test_linear_none_input_returns_none():
    out = _exec(LinearNode(), tensor_in=None,
                in_features=4, out_features=8, bias=True,
                activation="none", freeze=False)["tensor_out"]
    assert out is None

def test_full_tensor_pipeline():
    flat = FlattenNode()
    lin1 = LinearNode()
    lin2 = LinearNode()
    x = torch.randn(1, 1, 28, 28)
    t1 = _exec(flat, tensor_in=x, start_dim=1)["tensor_out"]
    assert t1.shape == (1, 784)
    t2 = _exec(lin1, tensor_in=t1, in_features=784, out_features=256,
               bias=True, activation="relu", freeze=False)["tensor_out"]
    assert t2.shape == (1, 256)
    t3 = _exec(lin2, tensor_in=t2, in_features=256, out_features=10,
               bias=True, activation="none", freeze=False)["tensor_out"]
    assert t3.shape == (1, 10)

def test_conv2d_output_shape():
    out = _exec(Conv2dNode(), tensor_in=torch.randn(1, 1, 8, 8),
                in_ch=1, out_ch=4, kernel=3, stride=1, padding=0,
                activation="none", freeze=False)["tensor_out"]
    assert out.shape == (1, 4, 6, 6)

def test_maxpool_output_shape():
    out = _exec(MaxPool2dNode(), tensor_in=torch.randn(1, 1, 8, 8),
                kernel=2, stride=2)["tensor_out"]
    assert out.shape == (1, 1, 4, 4)

def test_avgpool_output_shape():
    out = _exec(AvgPool2dNode(), tensor_in=torch.randn(1, 1, 8, 8),
                kernel=2, stride=2)["tensor_out"]
    assert out.shape == (1, 1, 4, 4)

def test_activation_relu():
    out = _exec(ActivationNode(), tensor_in=torch.randn(2, 4),
                activation="relu")["tensor_out"]
    assert (out >= 0).all()

def test_activation_none_passthrough():
    x = torch.randn(2, 4)
    out = _exec(ActivationNode(), tensor_in=x, activation="none")["tensor_out"]
    assert out is x

def test_dropout_shape_preserved():
    out = _exec(DropoutNode(), tensor_in=torch.ones(2, 8), p=0.5)["tensor_out"]
    assert out.shape == (2, 8)


# ── Persistent layer state ────────────────────────────────────────────────────

def test_layer_same_object_on_reexecute():
    lin = LinearNode()
    _exec(lin, tensor_in=None, in_features=4, out_features=8,
          bias=True, activation="none", freeze=False)
    layer_a = lin._layer
    _exec(lin, tensor_in=None, in_features=4, out_features=8,
          bias=True, activation="none", freeze=False)
    assert lin._layer is layer_a

def test_layer_recreated_on_size_change():
    lin = LinearNode()
    _exec(lin, tensor_in=None, in_features=4, out_features=8,
          bias=True, activation="none", freeze=False)
    layer_a = lin._layer
    _exec(lin, tensor_in=None, in_features=16, out_features=8,
          bias=True, activation="none", freeze=False)
    assert lin._layer is not layer_a

def test_weight_edit_survives_reexecute():
    lin = LinearNode()
    _exec(lin, tensor_in=None, in_features=4, out_features=4,
          bias=True, activation="none", freeze=False)
    with torch.no_grad():
        lin._layer.weight.fill_(1.0)
    _exec(lin, tensor_in=None, in_features=4, out_features=4,
          bias=True, activation="none", freeze=False)
    assert lin._layer.weight.mean().item() == pytest.approx(1.0)

def test_training_update_reflected_in_tensor_out():
    lin = LinearNode()
    x = torch.randn(1, 4)
    out1 = _exec(lin, tensor_in=x, in_features=4, out_features=4,
                 bias=True, activation="none", freeze=False)["tensor_out"].clone()
    with torch.no_grad():
        lin._layer.weight.fill_(0.5)
        lin._layer.bias.fill_(0.0)
    out2 = _exec(lin, tensor_in=x, in_features=4, out_features=4,
                 bias=True, activation="none", freeze=False)["tensor_out"]
    assert not torch.allclose(out1, out2)


# ── get_layers() for model assembly ──────────────────────────────────────────

def test_flatten_get_layers():
    flat = FlattenNode()
    _exec(flat, tensor_in=None, start_dim=1)
    layers = flat.get_layers()
    assert len(layers) == 1
    assert isinstance(layers[0], nn.Flatten)

def test_linear_get_layers_no_activation():
    lin = LinearNode()
    _exec(lin, tensor_in=None, in_features=4, out_features=8,
          bias=True, activation="none", freeze=False)
    layers = lin.get_layers()
    assert len(layers) == 1
    assert isinstance(layers[0], nn.Linear)

def test_linear_get_layers_with_activation():
    lin = LinearNode()
    _exec(lin, tensor_in=None, in_features=4, out_features=8,
          bias=True, activation="relu", freeze=False)
    layers = lin.get_layers()
    assert len(layers) == 2
    assert isinstance(layers[0], nn.Linear)
    assert isinstance(layers[1], nn.ReLU)

def test_get_layers_returns_persistent_layer():
    lin = LinearNode()
    _exec(lin, tensor_in=None, in_features=4, out_features=8,
          bias=True, activation="none", freeze=False)
    assert lin.get_layers()[0] is lin._layer

def test_assembled_sequential_trains_same_weights():
    """Layers collected by get_layers() share objects with the nodes."""
    lin = LinearNode()
    _exec(lin, tensor_in=None, in_features=4, out_features=4,
          bias=True, activation="none", freeze=False)
    modules = lin.get_layers()
    model = nn.Sequential(*modules)
    # Train model for one step
    import torch.optim as optim
    opt = optim.SGD(model.parameters(), lr=0.1)
    x = torch.randn(2, 4)
    loss = model(x).sum()
    loss.backward()
    w_before = lin._layer.weight.data.clone()
    opt.step()
    # lin._layer.weight should have changed
    assert not torch.allclose(lin._layer.weight.data, w_before)
