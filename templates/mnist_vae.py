"""MNIST VAE — per-layer VAE with loss-as-output. Every layer visible.

Marker-based architecture: the graph contains zero data-loading nodes. An A
marker injects the image batch at training time; the VAE encodes, samples, and
decodes; reconstruction loss is computed against the flattened input (not an
external label); and a B marker marks the combined VAE loss as the training
output. All dataset config lives in the Training Panel.

    Data In (A:x) → Flatten → Encoder → μ/logvar → Reparameterize → Decoder
                                                                          ↓
                                              VAELoss ← KL ← μ/logvar
                                              VAELoss ← BCE(recon, flat_input)
                                                  ↓
                                           Data Out (B:loss)
"""
from __future__ import annotations
from core.graph import Graph
from templates._helpers import grid

LABEL = "MNIST VAE (Image Generator)"
DESCRIPTION = "Per-layer VAE with loss computed in the graph via LossCompute + KL + VAELoss. Marker-based — dataset lives in the panel."

def build(graph: Graph) -> dict[str, tuple[int, int]]:
    from nodes.pytorch.input_marker      import InputMarkerNode
    from nodes.pytorch.flatten           import FlattenNode
    from nodes.pytorch.layer             import LayerNode
    from nodes.pytorch.latent            import LatentNode
    from nodes.pytorch.loss_compute      import LossComputeNode
    from nodes.pytorch.vae_loss          import VAELossNode
    from nodes.pytorch.train_marker      import TrainMarkerNode

    LATENT = 16
    pos = grid(step_x=200); positions = {}

    data_in = InputMarkerNode(); data_in.inputs["modality"].default_value = "x"
    graph.add_node(data_in); positions[data_in.id] = pos(col=0, row=2)

    flat = FlattenNode(); graph.add_node(flat); positions[flat.id] = pos(col=1, row=2)

    def _lin(out_f: int, act: str = "none") -> LayerNode:
        n = LayerNode()
        n.inputs["kind"].default_value         = "linear"
        n.inputs["out_features"].default_value = out_f
        n.inputs["activation"].default_value   = act
        return n

    enc1 = _lin(256, "relu"); graph.add_node(enc1); positions[enc1.id] = pos(col=2, row=2)
    enc2 = _lin(64,  "relu"); graph.add_node(enc2); positions[enc2.id] = pos(col=3, row=2)
    mu   = _lin(LATENT);      graph.add_node(mu);   positions[mu.id]   = pos(col=4, row=1)
    lv   = _lin(LATENT);      graph.add_node(lv);   positions[lv.id]   = pos(col=4, row=3)

    repar = LatentNode()
    repar.inputs["mode"].default_value = "reparameterize"
    graph.add_node(repar); positions[repar.id] = pos(col=5, row=2)

    dec1 = _lin(64,  "relu");    graph.add_node(dec1); positions[dec1.id] = pos(col=6, row=2)
    dec2 = _lin(256, "relu");    graph.add_node(dec2); positions[dec2.id] = pos(col=7, row=2)
    dec3 = _lin(784, "sigmoid"); graph.add_node(dec3); positions[dec3.id] = pos(col=8, row=2)

    rl = LossComputeNode(); rl.inputs["loss_type"].default_value="bce"
    graph.add_node(rl); positions[rl.id] = pos(col=9, row=1)
    kl = VAELossNode(); kl.inputs["mode"].default_value = "kl"
    graph.add_node(kl); positions[kl.id] = pos(col=9, row=3)
    vl = VAELossNode(); vl.inputs["mode"].default_value = "combine"
    graph.add_node(vl); positions[vl.id] = pos(col=10, row=2)
    data_out = TrainMarkerNode(); data_out.inputs["kind"].default_value="loss"
    graph.add_node(data_out); positions[data_out.id] = pos(col=11, row=2)

    graph.add_connection(data_in.id,"tensor",flat.id,"tensor_in")
    graph.add_connection(flat.id,"tensor_out",enc1.id,"tensor_in"); graph.add_connection(enc1.id,"tensor_out",enc2.id,"tensor_in")
    graph.add_connection(enc2.id,"tensor_out",mu.id,"tensor_in"); graph.add_connection(enc2.id,"tensor_out",lv.id,"tensor_in")
    graph.add_connection(mu.id,"tensor_out",repar.id,"mean"); graph.add_connection(lv.id,"tensor_out",repar.id,"log_variance")
    graph.add_connection(repar.id,"z",dec1.id,"tensor_in"); graph.add_connection(dec1.id,"tensor_out",dec2.id,"tensor_in"); graph.add_connection(dec2.id,"tensor_out",dec3.id,"tensor_in")
    graph.add_connection(dec3.id,"tensor_out",rl.id,"pred"); graph.add_connection(flat.id,"tensor_out",rl.id,"target")
    graph.add_connection(mu.id,"tensor_out",kl.id,"mean"); graph.add_connection(lv.id,"tensor_out",kl.id,"log_variance")
    graph.add_connection(rl.id,"loss",vl.id,"recon_loss"); graph.add_connection(kl.id,"kl_loss",vl.id,"kl_loss")
    graph.add_connection(vl.id,"loss",data_out.id,"tensor_in")
    return positions
