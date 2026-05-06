"""Char-level LSTM language model — Karpathy minRNN style.

Marker-based architecture: no data-loading node in the graph. Two A markers
inject the token indices (x) and shifted labels (label) at training time.
The pipeline embeds, runs an LSTM, projects back to vocab, reshapes for
cross-entropy, and a B marker marks the loss as the training target.

    Data In (A:x) → Embedding → LSTM → Linear → ReshapeForLoss ──► LossCompute(CE) → Data Out (B:loss)
    Data In (A:label) ──────────────────────────────────────────►/
"""
from __future__ import annotations
from core.graph import Graph
from templates._helpers import grid

LABEL = "Char-Level LSTM Language Model"
DESCRIPTION = "Char-level LSTM LM. Marker-based — dataset lives in the panel."


def build(graph: Graph) -> dict[str, tuple[int, int]]:
    from nodes.pytorch.input_marker       import InputMarkerNode
    from nodes.pytorch.layer              import LayerNode
    from nodes.pytorch.recurrent_layer    import RecurrentLayerNode
    from nodes.pytorch.reshape_for_loss   import ReshapeForLossNode
    from nodes.pytorch.loss_compute       import LossComputeNode
    from nodes.pytorch.train_marker       import TrainMarkerNode

    pos = grid(step_x=220); positions = {}
    SEQ_LEN, EMBED, HIDDEN = 64, 64, 128

    # ── Input markers ───────────────────────────────────────────────────────
    x_in = InputMarkerNode()
    x_in.inputs["modality"].default_value = "x"
    graph.add_node(x_in); positions[x_in.id] = pos(col=0, row=1)

    label_in = InputMarkerNode()
    label_in.inputs["modality"].default_value = "label"
    graph.add_node(label_in); positions[label_in.id] = pos(col=0, row=2)

    # ── Embedding ───────────────────────────────────────────────────────────
    # vocab_size hardcoded to 256 (fallback corpus default)
    emb = LayerNode()
    emb.inputs["kind"].default_value           = "embedding"
    emb.inputs["num_embeddings"].default_value = 256
    emb.inputs["embedding_dim"].default_value  = EMBED
    graph.add_node(emb); positions[emb.id] = pos(col=1, row=1)

    # ── LSTM ─────────────────────────────────────────────────────────────────
    lstm = RecurrentLayerNode()
    lstm.inputs["kind"].default_value        = "lstm"
    lstm.inputs["hidden_size"].default_value = HIDDEN
    lstm.inputs["batch_first"].default_value = True
    graph.add_node(lstm); positions[lstm.id] = pos(col=2, row=1)

    # ── Linear head ──────────────────────────────────────────────────────────
    # out_features hardcoded to 256 (fallback corpus default)
    head = LayerNode()
    head.inputs["kind"].default_value         = "linear"
    head.inputs["out_features"].default_value = 256
    graph.add_node(head); positions[head.id] = pos(col=3, row=1)

    # ── Reshape + loss ───────────────────────────────────────────────────────
    reshape = ReshapeForLossNode()
    graph.add_node(reshape); positions[reshape.id] = pos(col=4, row=1)

    loss = LossComputeNode()
    loss.inputs["loss_type"].default_value = "cross_entropy"
    graph.add_node(loss); positions[loss.id] = pos(col=5, row=1)

    # ── Train marker ─────────────────────────────────────────────────────────
    data_out = TrainMarkerNode()
    data_out.inputs["kind"].default_value = "loss"
    graph.add_node(data_out); positions[data_out.id] = pos(col=6, row=1)

    # ── Connections ──────────────────────────────────────────────────────────
    graph.add_connection(x_in.id,     "tensor",     emb.id,      "tensor_in")
    graph.add_connection(emb.id,      "tensor_out",  lstm.id,     "input_seq")
    graph.add_connection(lstm.id,     "output",      head.id,     "tensor_in")
    graph.add_connection(head.id,     "tensor_out",  reshape.id,  "logits")
    graph.add_connection(label_in.id, "tensor",      reshape.id,  "labels")
    graph.add_connection(reshape.id,  "logits_flat", loss.id,     "pred")
    graph.add_connection(reshape.id,  "labels_flat", loss.id,     "target")
    graph.add_connection(loss.id,     "loss",        data_out.id, "tensor_in")
    return positions
