"""Char-level LSTM language model — Karpathy minRNN style."""
from __future__ import annotations
from core.graph import Graph
from templates._helpers import grid

LABEL = "Char-Level LSTM Language Model"
DESCRIPTION = "TextDataset -> Embedding -> LSTM -> Linear -> ReshapeForLoss -> TrainOutput."

def build(graph: Graph) -> dict[str, tuple[int, int]]:
    from nodes.pytorch.dataset           import DatasetNode
    from nodes.pytorch.embedding         import EmbeddingNode
    from nodes.pytorch.lstm_layer        import LSTMLayerNode
    from nodes.pytorch.lstm_forward      import LSTMForwardNode
    from nodes.pytorch.linear            import LinearNode
    from nodes.pytorch.reshape_for_loss  import ReshapeForLossNode
    from nodes.pytorch.train_output      import TrainOutputNode

    pos = grid(step_x=220); positions = {}
    SEQ_LEN, EMBED, HIDDEN = 64, 64, 128

    ds = DatasetNode(); ds.inputs["path"].default_value="data/text.txt"; ds.inputs["seq_len"].default_value=SEQ_LEN; ds.inputs["batch_size"].default_value=32
    graph.add_node(ds); positions[ds.id] = pos(col=0, row=1)

    emb = EmbeddingNode(); emb.inputs["num_embeddings"].default_value=256; emb.inputs["embedding_dim"].default_value=EMBED
    graph.add_node(emb); positions[emb.id] = pos(col=1, row=1)

    lstm = LSTMLayerNode(); lstm.inputs["input_size"].default_value=EMBED; lstm.inputs["hidden_size"].default_value=HIDDEN; lstm.inputs["batch_first"].default_value=True
    graph.add_node(lstm); positions[lstm.id] = pos(col=2, row=0)
    fwd = LSTMForwardNode(); graph.add_node(fwd); positions[fwd.id] = pos(col=2, row=1)

    head = LinearNode(); head.inputs["in_features"].default_value=HIDDEN; head.inputs["out_features"].default_value=256
    graph.add_node(head); positions[head.id] = pos(col=3, row=1)

    reshape = ReshapeForLossNode(); graph.add_node(reshape); positions[reshape.id] = pos(col=4, row=1)
    target = TrainOutputNode(); graph.add_node(target); positions[target.id] = pos(col=5, row=1)

    # Wire vocab_size into embedding + head
    graph.add_connection(ds.id,"vocab_size",emb.id,"num_embeddings")
    graph.add_connection(ds.id,"vocab_size",head.id,"out_features")
    graph.add_connection(ds.id,"x",emb.id,"tensor_in")
    graph.add_connection(emb.id,"tensor_out",fwd.id,"x"); graph.add_connection(lstm.id,"module",fwd.id,"module")
    graph.add_connection(fwd.id,"output",head.id,"tensor_in")
    graph.add_connection(head.id,"tensor_out",reshape.id,"logits"); graph.add_connection(ds.id,"label",reshape.id,"labels")
    graph.add_connection(reshape.id,"logits_flat",target.id,"tensor_in")
    return positions
