"""MultiDatasetNode — wrap 2-4 datasets into a single multimodal MultiLoader."""
from __future__ import annotations
from core.node import BaseNode, PortType


_STRATEGIES = ["round_robin", "weighted_mix", "alternate_epochs"]


class MultiDatasetNode(BaseNode):
    type_name   = "pt_multi_dataset"
    label       = "Multi Dataset"
    category    = "Datasets"
    subcategory = "Loader"
    description = (
        "Combine 2-4 multimodal datasets into one DataLoader. "
        "Strategies: round_robin (alternate per batch), weighted_mix (sample by weight), "
        "alternate_epochs (drain one fully then next)."
    )

    def _setup_ports(self):
        for i in range(1, 5):
            self.add_input(f"dataset_{i}", PortType.DATASET, default=None,
                           description=f"Multimodal dataset {i}")

        self.add_input("batch_size",  PortType.INT,    default=8)
        self.add_input("shuffle",     PortType.BOOL,   default=True)
        self.add_input("num_workers", PortType.INT,    default=0)
        self.add_input("strategy",    PortType.STRING, default="round_robin",
                       choices=_STRATEGIES,
                       description="How to alternate between datasets")
        self.add_input("weights",     PortType.STRING, default="1,1,1,1",
                       description="Comma-separated sampling weights for weighted_mix")
        self.add_input("epoch_steps", PortType.INT,    default=0,
                       description="Steps per epoch (0 = sum of all loaders, only used for weighted_mix)")

        self.add_output("dataloader", PortType.DATALOADER,
                        description="A MultiLoader instance compatible with MultimodalTrainingConfig")
        self.add_output("info",       PortType.STRING)

    def execute(self, inputs):
        try:
            from nodes.pytorch._multimodal_loader import MultiLoader
        except ImportError:
            return {"dataloader": None, "info": "import error"}

        datasets = [inputs.get(f"dataset_{i}") for i in range(1, 5)]
        datasets = [d for d in datasets if d is not None]
        if not datasets:
            return {"dataloader": None, "info": "no datasets connected"}

        try:
            weights_str = inputs.get("weights", "") or ""
            weights = [float(w) for w in weights_str.split(",") if w.strip()]
            if len(weights) < len(datasets):
                weights = weights + [1.0] * (len(datasets) - len(weights))
            weights = weights[:len(datasets)]

            epoch_steps = int(inputs.get("epoch_steps", 0) or 0) or None

            loader = MultiLoader(
                datasets    = datasets,
                batch_size  = int(inputs.get("batch_size", 8)),
                shuffle     = bool(inputs.get("shuffle", True)),
                num_workers = int(inputs.get("num_workers", 0)),
                strategy    = str(inputs.get("strategy", "round_robin")),
                weights     = weights,
                epoch_steps = epoch_steps,
            )
            info = (f"{len(datasets)} datasets, strategy={inputs.get('strategy')}, "
                    f"~{len(loader)} batches/epoch")
            return {"dataloader": loader, "info": info}
        except Exception as exc:
            return {"dataloader": None, "info": f"error: {exc}"}

    def export(self, iv, ov):
        return [], [
            f"# {self.label}: requires the MultiLoader helper class.",
            f"# Use the GUI training flow — code export of multimodal training is not yet supported.",
            f"{ov['dataloader']} = None",
            f"{ov['info']} = ''",
        ]
