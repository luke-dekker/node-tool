"""Char-level LSTM language model template.

The classic Karpathy minRNN setup, fully wired in the visual editor:

  TextDataset (built-in corpus or your own .txt)
    └─> dataloader ─> BatchInput
                       └─> x (int seq) ─> Embedding
                                            └─> LSTM Forward (over LSTM Layer)
                                                  └─> output ─> Linear (vocab head)
                                                                  └─> ReshapeForLoss
                                                                       (logits flatten)
                       └─> label  ─> ReshapeForLoss (labels flatten)
                                       └─> labels_flat
  ReshapeForLoss.logits_flat ─> TrainingConfig.tensor_in
  TrainingConfig (loss=crossentropy)

This template demonstrates:
  - Sequence model training with token-level loss
  - The TextDataset / Embedding / LSTM Layer / LSTM Forward / ReshapeForLoss
    quartet that any char-LM, word-LM, or sequence-labeling model needs
  - The shape contract: (B, T) int → (B, T, embed) → (B, T, hidden) →
    (B, T, V) → (B*T, V) for loss

NOTE: BatchInput's `label` port is the per-window target — TextDataset emits
batches as (input, target) tuples and BatchInput unpacks them with x → its
`x` output and the second tuple element → its `label` output.
"""
from __future__ import annotations
from core.graph import Graph
from templates._helpers import grid


def build(graph: Graph) -> dict[str, tuple[int, int]]:
    from nodes.pytorch.text_dataset      import TextDatasetNode
    from nodes.pytorch.batch_input       import BatchInputNode
    from nodes.pytorch.embedding         import EmbeddingNode
    from nodes.pytorch.lstm_layer        import LSTMLayerNode
    from nodes.pytorch.lstm_forward      import LSTMForwardNode
    from nodes.pytorch.linear            import LinearNode
    from nodes.pytorch.reshape_for_loss  import ReshapeForLossNode
    from nodes.pytorch.training_config   import TrainingConfigNode

    pos = grid(step_x=240)
    positions: dict[str, tuple[int, int]] = {}

    # Configurable knobs (visible to the user as node inputs)
    SEQ_LEN    = 64
    EMBED_DIM  = 64
    HIDDEN     = 128
    # Embedding.num_embeddings and Linear.out_features are WIRED to the
    # TextDataset's vocab_size output below — the layer nodes rebuild on the
    # first execute with the actual vocab from the loaded text. The defaults
    # below only matter as placeholders before the first run.
    VOCAB_PLACEHOLDER = 256

    ds = TextDatasetNode()
    ds.inputs["seq_len"].default_value    = SEQ_LEN
    ds.inputs["batch_size"].default_value = 32
    graph.add_node(ds); positions[ds.id] = pos(col=0, row=1)

    batch = BatchInputNode()
    graph.add_node(batch); positions[batch.id] = pos(col=1, row=1)

    emb = EmbeddingNode()
    emb.inputs["num_embeddings"].default_value = VOCAB_PLACEHOLDER
    emb.inputs["embedding_dim"].default_value  = EMBED_DIM
    graph.add_node(emb); positions[emb.id] = pos(col=2, row=1)

    lstm_layer = LSTMLayerNode()
    lstm_layer.inputs["input_size"].default_value  = EMBED_DIM
    lstm_layer.inputs["hidden_size"].default_value = HIDDEN
    lstm_layer.inputs["num_layers"].default_value  = 1
    lstm_layer.inputs["batch_first"].default_value = True
    graph.add_node(lstm_layer); positions[lstm_layer.id] = pos(col=3, row=0)

    lstm_fwd = LSTMForwardNode()
    graph.add_node(lstm_fwd); positions[lstm_fwd.id] = pos(col=3, row=1)

    head = LinearNode()
    head.inputs["in_features"].default_value  = HIDDEN
    head.inputs["out_features"].default_value = VOCAB_PLACEHOLDER
    head.inputs["activation"].default_value   = "none"
    graph.add_node(head); positions[head.id] = pos(col=4, row=1)

    reshape = ReshapeForLossNode()
    graph.add_node(reshape); positions[reshape.id] = pos(col=5, row=1)

    cfg = TrainingConfigNode()
    cfg.inputs["epochs"].default_value    = 5
    cfg.inputs["lr"].default_value        = 0.005
    cfg.inputs["loss"].default_value      = "crossentropy"
    cfg.inputs["optimizer"].default_value = "adam"
    graph.add_node(cfg); positions[cfg.id] = pos(col=6, row=1)

    # Wire it
    graph.add_connection(ds.id,         "dataloader", batch.id,      "dataloader")
    graph.add_connection(ds.id,         "dataloader", cfg.id,        "dataloader")
    # Vocab size flows from the dataset into Embedding and Linear so the layer
    # sizes match the actual vocab of the loaded text — no hardcoding needed
    graph.add_connection(ds.id,         "vocab_size", emb.id,        "num_embeddings")
    graph.add_connection(ds.id,         "vocab_size", head.id,       "out_features")
    graph.add_connection(batch.id,      "x",          emb.id,        "tensor_in")
    graph.add_connection(emb.id,        "tensor_out", lstm_fwd.id,   "x")
    graph.add_connection(lstm_layer.id, "module",     lstm_fwd.id,   "module")
    graph.add_connection(lstm_fwd.id,   "output",     head.id,       "tensor_in")
    graph.add_connection(head.id,       "tensor_out", reshape.id,    "logits")
    graph.add_connection(batch.id,      "label",      reshape.id,    "labels")
    graph.add_connection(reshape.id,    "logits_flat", cfg.id,       "tensor_in")
    return positions
