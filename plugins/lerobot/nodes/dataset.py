"""LeRobot Dataset node — load demonstration data from HuggingFace Hub.

Wraps lerobot.datasets.lerobot_dataset.LeRobotDataset. Loads recorded
human demonstrations (joint positions + camera frames) and provides them
as a DataLoader for training ACT / Diffusion policies.

Usage:
    Set repo_id to a HuggingFace dataset like "lerobot/aloha_mobile_cabinet"
    or a local path to a recorded dataset.
"""
from __future__ import annotations
from typing import Any
from core.node import BaseNode, PortType


class LeRobotDatasetNode(BaseNode):
    type_name   = "lr_dataset"
    label       = "LeRobot Dataset"
    category    = "LeRobot"
    description = (
        "Load a LeRobot demonstration dataset from HuggingFace Hub or local path. "
        "Outputs a DataLoader of (observation, action) pairs for training "
        "imitation learning policies (ACT, Diffusion, etc.)."
    )

    def __init__(self):
        self._cached_loader = None
        self._cached_cfg: tuple = ()
        super().__init__()

    def _setup_ports(self) -> None:
        self.add_input("repo_id",     PortType.STRING, "lerobot/so100_test",
                       description="HuggingFace repo ID or local dataset path")
        self.add_input("batch_size",  PortType.INT, 32)
        self.add_input("shuffle",     PortType.BOOL, True)
        self.add_input("task_id",     PortType.STRING, "default",
                       description="Pairs with a Train Output of the same task_name")
        self.add_output("x",          PortType.TENSOR,
                        description="Observation batch (joint positions + image features)")
        self.add_output("label",      PortType.TENSOR,
                        description="Action batch (target joint positions)")
        self.add_output("dataloader", PortType.DATALOADER)
        self.add_output("info",       PortType.STRING)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        repo_id    = str(inputs.get("repo_id") or "")
        batch_size = max(1, int(inputs.get("batch_size") or 32))
        shuffle    = bool(inputs.get("shuffle", True))
        empty = {"x": None, "label": None, "dataloader": None, "info": ""}

        if not repo_id:
            return {**empty, "info": "Set repo_id to a HuggingFace dataset"}

        cfg = (repo_id, batch_size, shuffle)
        if self._cached_loader is not None and self._cached_cfg == cfg:
            loader = self._cached_loader
        else:
            try:
                from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
                from torch.utils.data import DataLoader

                dataset = LeRobotDataset(repo_id)
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
                self._cached_loader = loader
                self._cached_cfg = cfg
            except ImportError:
                return {**empty, "info": "lerobot not installed. Run: pip install lerobot"}
            except Exception as exc:
                return {**empty, "info": f"Load failed: {exc}"}

        # Sample one batch for preview
        x, label = None, None
        try:
            batch = next(iter(loader))
            # LeRobot batch is a dict with keys like 'observation.state',
            # 'observation.images.top', 'action', etc.
            if isinstance(batch, dict):
                # Extract joint state as x, action as label
                x = batch.get("observation.state", batch.get("observation", None))
                label = batch.get("action")
                if x is None:
                    # Fallback: try to find any tensor
                    for k, v in batch.items():
                        if hasattr(v, "shape") and x is None:
                            x = v
            elif isinstance(batch, (list, tuple)):
                x = batch[0]
                label = batch[1] if len(batch) > 1 else None
        except Exception:
            pass

        n_episodes = "?"
        try:
            n_episodes = len(loader.dataset)
        except Exception:
            pass

        info = f"{repo_id}: {n_episodes} samples, batch_size={batch_size}"
        return {"x": x, "label": label, "dataloader": loader, "info": info}

    def export(self, iv, ov):
        repo = self._val(iv, "repo_id")
        bs = self._val(iv, "batch_size")
        return [
            "from lerobot.common.datasets.lerobot_dataset import LeRobotDataset",
            "from torch.utils.data import DataLoader",
        ], [
            f"_lr_ds = LeRobotDataset({repo})",
            f"{ov.get('dataloader', '_lr_dl')} = DataLoader(_lr_ds, batch_size={bs})",
            f"_batch = next(iter({ov.get('dataloader', '_lr_dl')}))",
            f"{ov.get('x', '_lr_obs')} = _batch.get('observation.state', _batch.get('observation'))",
            f"{ov.get('label', '_lr_act')} = _batch.get('action')",
        ]
