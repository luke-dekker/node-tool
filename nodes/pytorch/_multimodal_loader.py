"""MultiLoader + multimodal collate — used by MultiDatasetNode and MultimodalTrainingConfig."""
from __future__ import annotations
from typing import Iterator, Any


def _to_tensor(v):
    """Best-effort: convert numpy / list / Tensor to torch tensor."""
    import torch
    if v is None:
        return None
    if isinstance(v, torch.Tensor):
        return v
    try:
        import numpy as np
        if isinstance(v, np.ndarray):
            return torch.from_numpy(v)
    except ImportError:
        pass
    if isinstance(v, (int, float)):
        return torch.tensor(v)
    if isinstance(v, list):
        try:
            return torch.tensor(v)
        except Exception:
            return None
    return None


def multimodal_collate(samples: list[dict]) -> dict:
    """Collate a list of {data, mask, label, id} samples into a batched dict.

    Output:
        {
            'data':  {modality: stacked_tensor or None},
            'mask':  [B, M] tensor,
            'label': [B] tensor (int) or list of strings,
            'present': list[str] of modalities present in ALL samples,
        }

    A modality is included in the batch only if EVERY sample in the batch has it
    (avoids ragged tensors). The mask reflects per-sample presence, but the
    'data' dict may contain None for modalities that aren't shared across the batch.
    """
    import torch

    if not samples:
        return {"data": {}, "mask": torch.zeros(0, 0), "label": [], "present": []}

    modalities = list(samples[0]["data"].keys())
    present_in_all = [m for m in modalities
                      if all(s["data"].get(m) is not None for s in samples)]

    data = {}
    for m in modalities:
        if m in present_in_all:
            tensors = [_to_tensor(s["data"][m]) for s in samples]
            tensors = [t for t in tensors if t is not None]
            if not tensors:
                data[m] = None
                continue
            try:
                # Stack if all same shape, else keep as list
                shapes = {tuple(t.shape) for t in tensors}
                if len(shapes) == 1:
                    data[m] = torch.stack(tensors, dim=0).float()
                else:
                    data[m] = tensors  # ragged — model needs to handle list
            except Exception:
                data[m] = None
        else:
            data[m] = None

    masks = torch.stack([s["mask"] for s in samples], dim=0) if "mask" in samples[0] else None
    labels = [s.get("label", "") for s in samples]
    # Try to convert labels to int tensor
    try:
        labels_t = torch.tensor([int(l) for l in labels])
    except (ValueError, TypeError):
        labels_t = labels  # keep as strings

    return {
        "data":    data,
        "mask":    masks,
        "label":   labels_t,
        "present": present_in_all,
    }


class MultiLoader:
    """Iterates over multiple DataLoaders with a chosen strategy.

    Strategies:
        round_robin     - one batch from each loader in order, repeat
        weighted_mix    - randomly pick a loader by weight each step
        alternate_epochs - drain loader 1 fully, then loader 2, etc.
    """

    def __init__(
        self,
        datasets: list,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        strategy: str = "round_robin",
        weights: list[float] | None = None,
        epoch_steps: int | None = None,
    ):
        from torch.utils.data import DataLoader
        self.datasets    = [d for d in datasets if d is not None]
        self.strategy    = strategy
        self.weights     = weights or [1.0] * len(self.datasets)
        self.epoch_steps = epoch_steps
        self.loaders = [
            DataLoader(d, batch_size=batch_size, shuffle=shuffle,
                       num_workers=num_workers, collate_fn=multimodal_collate)
            for d in self.datasets
        ]

    def __len__(self) -> int:
        if self.epoch_steps is not None:
            return int(self.epoch_steps)
        return sum(len(l) for l in self.loaders)

    def __iter__(self) -> Iterator[dict]:
        import random
        if not self.loaders:
            return

        if self.strategy == "alternate_epochs":
            for loader in self.loaders:
                yield from loader
            return

        if self.strategy == "weighted_mix":
            iters = [iter(l) for l in self.loaders]
            steps = self.epoch_steps or sum(len(l) for l in self.loaders)
            total_w = sum(self.weights)
            probs = [w / total_w for w in self.weights]
            for _ in range(steps):
                # Sample loader index by weight
                r = random.random()
                cum = 0.0
                idx = 0
                for i, p in enumerate(probs):
                    cum += p
                    if r <= cum:
                        idx = i
                        break
                try:
                    yield next(iters[idx])
                except StopIteration:
                    iters[idx] = iter(self.loaders[idx])
                    try:
                        yield next(iters[idx])
                    except StopIteration:
                        return
            return

        # default: round_robin
        iters = [iter(l) for l in self.loaders]
        active = list(range(len(iters)))
        while active:
            done = []
            for i in active:
                try:
                    yield next(iters[i])
                except StopIteration:
                    done.append(i)
            for i in done:
                active.remove(i)
