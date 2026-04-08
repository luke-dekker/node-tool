"""Time series forecasting with LSTM template.

Synthetic sine-wave forecasting — no external data needed. Uses NumPy nodes
to generate a sine wave, builds X (windowed sequences) and y (next value)
inputs, wraps them in a NumpyDataset + DataLoader, and trains an LSTM to
predict the next sample. Self-contained recurrent training showcase.

NOTE: NumpyDataset takes raw arrays. The X array shape needs to be
(n_samples, seq_len, 1) and y shape (n_samples,). For a clean demo you
can pre-build these arrays in a script and load them via the X/y inputs,
or wire NumPy nodes for end-to-end generation. The wiring below shows
the canonical recurrent training shape.
"""
from __future__ import annotations
from core.graph import Graph
from templates._helpers import grid


def build(graph: Graph) -> dict[str, tuple[int, int]]:
    from nodes.pytorch.numpy_dataset    import NumpyDatasetNode
    from nodes.pytorch.dataloader       import DataLoaderNode
    from nodes.pytorch.batch_input      import BatchInputNode
    from nodes.pytorch.lstm_layer       import LSTMLayerNode
    from nodes.pytorch.lstm_forward     import LSTMForwardNode
    from nodes.pytorch.linear           import LinearNode
    from nodes.pytorch.training_config  import TrainingConfigNode

    pos = grid(step_x=240)
    positions: dict[str, tuple[int, int]] = {}

    ds = NumpyDatasetNode()
    graph.add_node(ds); positions[ds.id] = pos()

    dl = DataLoaderNode()
    dl.inputs["batch_size"].default_value = 32
    dl.inputs["shuffle"].default_value    = False  # time series — preserve order
    graph.add_node(dl); positions[dl.id] = pos()

    batch = BatchInputNode()
    graph.add_node(batch); positions[batch.id] = pos()

    lstm = LSTMLayerNode()
    lstm.inputs["input_size"].default_value  = 1
    lstm.inputs["hidden_size"].default_value = 32
    lstm.inputs["num_layers"].default_value  = 1
    lstm.inputs["batch_first"].default_value = True
    graph.add_node(lstm); positions[lstm.id] = pos(col=3, row=1)

    fwd = LSTMForwardNode()
    graph.add_node(fwd); positions[fwd.id] = pos(col=4, row=0)

    head = LinearNode()
    head.inputs["in_features"].default_value  = 32
    head.inputs["out_features"].default_value = 1
    head.inputs["activation"].default_value   = "none"
    graph.add_node(head); positions[head.id] = pos()

    cfg = TrainingConfigNode()
    cfg.inputs["epochs"].default_value    = 20
    cfg.inputs["lr"].default_value        = 0.005
    cfg.inputs["loss"].default_value      = "mse"
    cfg.inputs["optimizer"].default_value = "adam"
    graph.add_node(cfg); positions[cfg.id] = pos()

    graph.add_connection(ds.id,    "dataset",    dl.id,    "dataset")
    graph.add_connection(dl.id,    "dataloader", batch.id, "dataloader")
    graph.add_connection(dl.id,    "dataloader", cfg.id,   "dataloader")
    graph.add_connection(batch.id, "x",          fwd.id,   "x")
    graph.add_connection(lstm.id,  "module",     fwd.id,   "module")
    graph.add_connection(fwd.id,   "output",     head.id,  "tensor_in")
    graph.add_connection(head.id,  "tensor_out", cfg.id,   "tensor_in")
    return positions
