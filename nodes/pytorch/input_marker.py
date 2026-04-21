"""InputMarkerNode — "A" marker for training graphs.

Half of the A/B marker pair that replaces DatasetNode + TrainOutputNode. A
marker nodes are dumb sockets — they carry no data-loading logic themselves.
The training panel owns the dataset config, fetches a batch, and feeds the
right tensor into each marker based on `modality`. Multiple A markers with
the same `group` form a multi-modal input for one task; markers with
different `group` values belong to different tasks in multi-task training.

During a live (non-training) graph execute, the marker returns whatever
tensor was most recently stashed in `_probe_tensor` by the panel's probe
run. If nothing has been stashed, the marker returns None — downstream
nodes will short-circuit and the graph stays quiet until a probe runs.

No `_layer`, no `_cached_loader`, no dynamic ports. The whole node body is
a setter and a getter.
"""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType, MarkerRole


class InputMarkerNode(BaseNode):
    type_name   = "pt_input_marker"
    label       = "Data In (A)"
    category    = "Training"
    subcategory = "Markers"
    marker_role = MarkerRole.INPUT
    description = (
        "Data input marker. Pairs with a Data Out (B) marker of the same "
        "group to form a trainable graph section. The Training Panel feeds "
        "a batch tensor into this marker at training time, keyed by the "
        "modality field (e.g. 'x', 'observation.state', 'audio'). No data "
        "loading happens inside the graph — all dataset config lives in the "
        "panel."
    )

    def __init__(self):
        # Tensor stashed by the panel/test before graph.execute(). Read by
        # execute() and emitted on the "tensor" output port. Instance-level
        # state, NOT a port, so it doesn't pollute serialization.
        self._probe_tensor: Any = None
        super().__init__()

    def _setup_ports(self) -> None:
        self.add_input("group",    PortType.STRING, default="task_1",
                       description="Pairs this marker with a Data Out (B) of the same group")
        self.add_input("modality", PortType.STRING, default="x",
                       description="Batch column name this marker consumes "
                                   "(e.g. 'x', 'label', 'observation.state')")
        # ── Dataset config (per-group). The training panel reads/writes
        # these defaults; the autoresearch agent can wire `control` into
        # any of them. The first A marker in a group is authoritative for
        # dataset config; sibling A markers in the same group share these
        # values (one dataset, multiple modalities).
        self.add_input("path",         PortType.STRING, default="",
                       description="Dataset path (mnist, cifar10, ./csv, lerobot/...)")
        self.add_input("batch_size",   PortType.INT,    default=128,
                       description="Batch size for this group's dataloader")
        self.add_input("split",        PortType.STRING, default="train",
                       choices=["train", "test", "val"],
                       description="Source split loaded from the dataset itself")
        self.add_input("val_fraction", PortType.FLOAT,  default=0.1,
                       description="0.0 = no held-out val; otherwise this "
                                   "fraction of the loaded dataset is split out")
        self.add_input("seq_len",      PortType.INT,    default=0,
                       description="Sequence length (text/timeseries datasets only)")
        self.add_input("chunk_size",   PortType.INT,    default=1,
                       description="Chunking factor (text/timeseries datasets only)")
        self.add_output("tensor",  PortType.TENSOR,
                        description="Injected batch tensor for this modality")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"tensor": self._probe_tensor}

    def export(self, iv, ov):
        modality = self.inputs["modality"].default_value or "x"
        out = ov.get("tensor", "_data_in")
        return [], [
            f"# Data In marker: modality={modality!r}",
            f"{out} = batch[{modality!r}]  # fed by the training loop",
        ]
