"""MNIST VAE — per-layer VAE with loss-as-output. Every layer visible."""
from __future__ import annotations
from core.graph import Graph
from templates._helpers import grid

LABEL = "MNIST VAE (Image Generator)"
DESCRIPTION = "Per-layer VAE with loss computed in the graph via LossCompute + KL + VAELoss."

def build(graph: Graph) -> dict[str, tuple[int, int]]:
    from nodes.pytorch.dataset          import DatasetNode
    from nodes.pytorch.flatten          import FlattenNode
    from nodes.pytorch.linear           import LinearNode
    from nodes.pytorch.reparameterize   import ReparameterizeNode
    from nodes.pytorch.kl_divergence    import KLDivergenceNode
    from nodes.pytorch.loss_compute     import LossComputeNode
    from nodes.pytorch.vae_loss         import VAELossNode
    from nodes.pytorch.train_output     import TrainOutputNode

    LATENT = 16
    pos = grid(step_x=200); positions = {}

    mnist = DatasetNode(); mnist.inputs["path"].default_value = "mnist"; mnist.inputs["batch_size"].default_value = 128
    graph.add_node(mnist); positions[mnist.id] = pos(col=0, row=2)

    flat = FlattenNode(); graph.add_node(flat); positions[flat.id] = pos(col=1, row=2)

    enc1 = LinearNode(); enc1.inputs["in_features"].default_value=784; enc1.inputs["out_features"].default_value=256; enc1.inputs["activation"].default_value="relu"
    graph.add_node(enc1); positions[enc1.id] = pos(col=2, row=2)
    enc2 = LinearNode(); enc2.inputs["in_features"].default_value=256; enc2.inputs["out_features"].default_value=64; enc2.inputs["activation"].default_value="relu"
    graph.add_node(enc2); positions[enc2.id] = pos(col=3, row=2)

    mu = LinearNode(); mu.inputs["in_features"].default_value=64; mu.inputs["out_features"].default_value=LATENT
    graph.add_node(mu); positions[mu.id] = pos(col=4, row=1)
    lv = LinearNode(); lv.inputs["in_features"].default_value=64; lv.inputs["out_features"].default_value=LATENT
    graph.add_node(lv); positions[lv.id] = pos(col=4, row=3)

    repar = ReparameterizeNode(); graph.add_node(repar); positions[repar.id] = pos(col=5, row=2)

    dec1 = LinearNode(); dec1.inputs["in_features"].default_value=LATENT; dec1.inputs["out_features"].default_value=64; dec1.inputs["activation"].default_value="relu"
    graph.add_node(dec1); positions[dec1.id] = pos(col=6, row=2)
    dec2 = LinearNode(); dec2.inputs["in_features"].default_value=64; dec2.inputs["out_features"].default_value=256; dec2.inputs["activation"].default_value="relu"
    graph.add_node(dec2); positions[dec2.id] = pos(col=7, row=2)
    dec3 = LinearNode(); dec3.inputs["in_features"].default_value=256; dec3.inputs["out_features"].default_value=784; dec3.inputs["activation"].default_value="sigmoid"
    graph.add_node(dec3); positions[dec3.id] = pos(col=8, row=2)

    rl = LossComputeNode(); rl.inputs["loss_type"].default_value="bce"
    graph.add_node(rl); positions[rl.id] = pos(col=9, row=1)
    kl = KLDivergenceNode(); graph.add_node(kl); positions[kl.id] = pos(col=9, row=3)
    vl = VAELossNode(); graph.add_node(vl); positions[vl.id] = pos(col=10, row=2)
    target = TrainOutputNode(); target.inputs["loss_is_output"].default_value=True
    graph.add_node(target); positions[target.id] = pos(col=11, row=2)

    graph.add_connection(mnist.id,"x",flat.id,"tensor_in")
    graph.add_connection(flat.id,"tensor_out",enc1.id,"tensor_in"); graph.add_connection(enc1.id,"tensor_out",enc2.id,"tensor_in")
    graph.add_connection(enc2.id,"tensor_out",mu.id,"tensor_in"); graph.add_connection(enc2.id,"tensor_out",lv.id,"tensor_in")
    graph.add_connection(mu.id,"tensor_out",repar.id,"mu"); graph.add_connection(lv.id,"tensor_out",repar.id,"log_var")
    graph.add_connection(repar.id,"z",dec1.id,"tensor_in"); graph.add_connection(dec1.id,"tensor_out",dec2.id,"tensor_in"); graph.add_connection(dec2.id,"tensor_out",dec3.id,"tensor_in")
    graph.add_connection(dec3.id,"tensor_out",rl.id,"pred"); graph.add_connection(flat.id,"tensor_out",rl.id,"target")
    graph.add_connection(mu.id,"tensor_out",kl.id,"mu"); graph.add_connection(lv.id,"tensor_out",kl.id,"log_var")
    graph.add_connection(rl.id,"loss",vl.id,"recon_loss"); graph.add_connection(kl.id,"kl_loss",vl.id,"kl_loss")
    graph.add_connection(vl.id,"loss",target.id,"tensor_in")
    return positions
