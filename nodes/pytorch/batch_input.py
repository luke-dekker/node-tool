"""BatchInputNode — marks where batch tensors enter the model graph.

This node is the literal entry point of the model. Its output ports are named
after modalities (audio, text, image, etc.) plus a generic 'x' for single-input
models. At training time, GraphAsModule injects the current batch's tensors
into these outputs before running the graph forward.

At live-preview time (Run Graph button), if a dataloader is connected, it pulls
one sample so the downstream layer chains have something to flow through.
Otherwise it yields None and the downstream nodes won't execute.
"""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


# Must match MODALITY_PORTS in multimodal_model.py so a direct wire is type-compatible
_BATCH_PORTS = ["x", "audio", "text", "image", "video", "sensor", "custom"]


class BatchInputNode(BaseNode):
    type_name   = "batch_input"
    label       = "Batch Input"
    category    = "Datasets"
    subcategory = "Loader"
    description = (
        "Entry point for training batches. Wire the dataloader in, then wire each "
        "output port into the matching encoder chain. At training time the current "
        "batch's tensors are injected here before the graph runs forward."
    )

    def _setup_ports(self) -> None:
        # Optional: wire a dataloader for live-preview sampling
        self.add_input("dataloader", PortType.DATALOADER, default=None,
                       description="Optional — used only for live preview sampling")
        # One output per modality plus 'x' for single-input pipelines
        for name in _BATCH_PORTS:
            self.add_output(name, PortType.TENSOR,
                            description=f"Batch tensor for '{name}' (None if absent)")
        self.add_output("label", PortType.TENSOR,
                        description="Batch labels")

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Live preview: if a dataloader is wired, pull one sample to visualize.
        Otherwise return None for all ports. Training mode is handled by
        GraphAsModule, which overrides these outputs directly before
        calling graph.execute()."""
        loader = inputs.get("dataloader")
        empty = {p: None for p in _BATCH_PORTS}
        empty["label"] = None
        if loader is None:
            return empty
        try:
            iterator = iter(loader)
            batch = next(iterator)
        except Exception:
            return empty

        # Handle the multimodal collate format: {data, mask, label, present}
        if isinstance(batch, dict) and "data" in batch:
            out = {}
            for p in _BATCH_PORTS:
                out[p] = batch["data"].get(p) if p != "x" else None
            out["label"] = batch.get("label")
            return out

        # Handle the standard (x, y) tuple format
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            return {**empty, "x": batch[0], "label": batch[1]}

        # Single tensor
        return {**empty, "x": batch}

    def export(self, iv, ov):
        """Emit code that pulls one batch from a connected dataloader and unpacks it
        into the per-modality tensor variables that downstream layers reference.

        The exported snippet matches the runtime semantics: it accepts the same three
        batch shapes (multimodal dict, (x, y) tuple, single tensor) and assigns each
        modality output that something downstream actually uses.
        """
        loader = iv.get("dataloader")
        lines: list[str] = []
        if loader is None:
            lines.append("# TODO: provide a torch.utils.data.DataLoader for `dataloader`")
            loader = "dataloader"

        # ov is {port_name: var_name} for every output port the exporter assigned
        # a variable to. Emit one unpack line per port — readers see exactly which
        # modalities the downstream graph consumes.
        lines += [
            f"_batch = next(iter({loader}))",
            "if isinstance(_batch, dict) and 'data' in _batch:",
            "    _data, _label_val = _batch['data'], _batch.get('label')",
            "elif isinstance(_batch, (list, tuple)) and len(_batch) >= 2:",
            "    _data, _label_val = {'x': _batch[0]}, _batch[1]",
            "else:",
            "    _data, _label_val = {'x': _batch}, None",
        ]
        for port_name, var_name in ov.items():
            if port_name == "label":
                lines.append(f"{var_name} = _label_val")
            else:
                lines.append(f"{var_name} = _data.get({port_name!r})")
        return ["import torch"], lines
