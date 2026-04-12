"""Shared temporal-chunking helper for dataset-like sources.

Used by both `DatasetNode` (offline parquet / LeRobot) and
`StreamingBufferNode` (live deque-backed source). Any dataset whose
`__getitem__(i)` returns a `dict[str, Tensor]` can be wrapped to yield
windowed samples for selected columns.
"""
from __future__ import annotations
from typing import Iterable


class ChunkedWrapper:
    """Wraps a dataset to return temporal windows for selected columns.

    For each index `i`, builds a window `[i, i+1, ..., i+chunk_size-1]`.
    For columns listed in `chunk_columns`, stacks the window along a new
    leading dim. Other columns return the value at `i` unchanged.

    If `episode_ids` is given, the window is clamped so it never crosses
    an episode boundary: indices past the end of episode[i] are replaced
    with the last valid index in that episode (standard action-chunking
    right-pad behavior).
    """

    def __init__(self, inner, chunk_size: int, chunk_columns: Iterable[str],
                 episode_ids=None):
        self.inner = inner
        self.chunk_size = max(1, int(chunk_size))
        self.chunk_columns = set(chunk_columns or [])
        self.episode_ids = episode_ids  # list-like or None

    def __len__(self):
        return len(self.inner)

    def _window_indices(self, i: int) -> list[int]:
        n = len(self.inner)
        if self.episode_ids is not None and i < len(self.episode_ids):
            ep = self.episode_ids[i]
            idxs: list[int] = []
            last_valid = i
            for k in range(self.chunk_size):
                j = i + k
                if j < n and j < len(self.episode_ids) and self.episode_ids[j] == ep:
                    last_valid = j
                    idxs.append(j)
                else:
                    idxs.append(last_valid)
            return idxs
        return [min(i + k, n - 1) for k in range(self.chunk_size)]

    def __getitem__(self, idx):
        import torch
        base = self.inner[idx]
        if self.chunk_size <= 1 or not self.chunk_columns:
            return base
        window = self._window_indices(idx)
        if all(j == idx for j in window):
            samples = [base] * self.chunk_size
        else:
            samples = [self.inner[j] for j in window]
        out = {}
        for k, v in base.items():
            if k in self.chunk_columns:
                pieces = [s.get(k) for s in samples]
                try:
                    out[k] = torch.stack([torch.as_tensor(p) for p in pieces])
                except Exception:
                    out[k] = v
            else:
                out[k] = v
        return out
