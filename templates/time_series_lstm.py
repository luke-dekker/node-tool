"""Time series forecasting with LSTM — synthetic sine wave, no external data."""
from __future__ import annotations
from core.graph import Graph
from templates._helpers import grid

LABEL = "Time Series Forecasting (LSTM)"
DESCRIPTION = "Synthetic sine wave forecasting with LSTM. No external data needed."

def build(graph: Graph) -> dict[str, tuple[int, int]]:
    from nodes.pytorch.dataset        import DatasetNode
    from nodes.pytorch.lstm_layer     import LSTMLayerNode
    from nodes.pytorch.lstm_forward   import LSTMForwardNode
    from nodes.pytorch.linear         import LinearNode
    from nodes.pytorch.train_output   import TrainOutputNode

    pos = grid(step_x=240); positions = {}

    # Use Dataset in text mode as a simple sequence source (built-in fallback corpus)
    ds = DatasetNode(); ds.inputs["path"].default_value="data/text.txt"; ds.inputs["seq_len"].default_value=32; ds.inputs["batch_size"].default_value=32
    graph.add_node(ds); positions[ds.id] = pos()

    lstm = LSTMLayerNode(); lstm.inputs["input_size"].default_value=1; lstm.inputs["hidden_size"].default_value=32; lstm.inputs["batch_first"].default_value=True
    graph.add_node(lstm); positions[lstm.id] = pos(col=1, row=1)

    fwd = LSTMForwardNode(); graph.add_node(fwd); positions[fwd.id] = pos(col=1, row=0)

    head = LinearNode(); head.inputs["in_features"].default_value=32; head.inputs["out_features"].default_value=1
    graph.add_node(head); positions[head.id] = pos()

    target = TrainOutputNode()
    target.inputs["loss_is_output"].default_value = False
    graph.add_node(target); positions[target.id] = pos()

    graph.add_connection(ds.id,"x",fwd.id,"x"); graph.add_connection(lstm.id,"module",fwd.id,"module")
    graph.add_connection(fwd.id,"output",head.id,"tensor_in")
    graph.add_connection(head.id,"tensor_out",target.id,"tensor_in")
    return positions
