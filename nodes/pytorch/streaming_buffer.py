"""StreamingBufferNode — live deque-backed dataset with write + read faces.

Records samples from live sources (cameras, servos, teleop) into a rolling
buffer, then serves that buffer as a `torch.utils.data.DataLoader` to the
Training Panel. Same shape as `DatasetNode` on the read face so offline and
live training share the same downstream graph.

Key design:
- Dynamic observation ports declared via a string input `observation_names`
  (comma-separated, e.g. "image,state"), mirroring `DatasetNode.extra_outputs`.
- Write face: observation inputs + `action` + `record` (BOOL).
- Read face: one output per observation name + `action` + `dataloader`.
- Episode state lives on the node instance (`self._episode`), mutated by
  inspector-panel buttons, NOT by an input port. This keeps episode control
  off the canvas and fits the "general-purpose node + rich inspector" model.
- Temporal chunking reuses the shared `ChunkedWrapper`, so training on a live
  buffer supports action-chunking identically to offline datasets.
- Parquet save/load: basic round-trip with row-oriented layout. Tensor-valued
  columns are stored as Python lists (pyarrow handles nested list types).
"""
from __future__ import annotations
from collections import deque
from pathlib import Path
from typing import Any

from core.node import BaseNode, PortType
from nodes.pytorch._chunking import ChunkedWrapper


# Columns that are always present on every record, in addition to the
# dynamically-declared observation columns.
_FIXED_COLS = ("action", "episode_id", "t")


class _BufferDataset:
    """Adapter: exposes the deque as a map-style dataset for DataLoader."""

    def __init__(self, buffer, obs_names):
        self._buffer = buffer
        self._obs_names = tuple(obs_names)

    def __len__(self):
        return len(self._buffer)

    def __getitem__(self, idx):
        import torch
        rec = self._buffer[idx]
        out = {}
        for name in self._obs_names:
            v = rec.get(name)
            out[name] = v if v is not None else torch.tensor(0.0)
        act = rec.get("action")
        out["action"] = act if act is not None else torch.tensor(0.0)
        return out


def _collate(samples):
    """Stack per-key tensors. Mirrors dataset.py _collate_manifest."""
    import torch
    if not samples:
        return {}
    keys = list(samples[0].keys())
    batch = {}
    for k in keys:
        tensors = [s[k] for s in samples]
        try:
            batch[k] = torch.stack(tensors)
        except Exception:
            batch[k] = tensors
    return batch


class StreamingBufferNode(BaseNode):
    type_name   = "pt_streaming_buffer"
    label       = "Streaming Buffer"
    category    = "Data"
    subcategory = "Live"
    description = (
        "Rolling deque that records live (observation, action) samples and "
        "serves them as a DataLoader for continuous / batched training. Same "
        "read-face shape as Dataset, so the downstream graph is identical "
        "whether training is offline or live."
    )

    def __init__(self):
        self._buffer: deque = deque(maxlen=10000)
        self._maxlen = 10000
        self._obs_names: tuple[str, ...] = ()
        self._episode: int = 0
        self._t: int = 0
        self._last_load_path: str = ""
        self._declared = False
        super().__init__()

    # ── Port setup ──────────────────────────────────────────────────────

    def _setup_ports(self) -> None:
        self.add_input("observation_names", PortType.STRING, default="image,state",
                       description="Comma-separated observation column names. "
                                   "Creates matching input + output ports for "
                                   "each (e.g. 'image,state').")
        self.add_input("buffer_size", PortType.INT, default=10000,
                       description="Rolling FIFO capacity. Oldest samples drop "
                                   "when full.")
        self.add_input("action", PortType.TENSOR, default=None,
                       description="Action tensor to record this step.")
        self.add_input("record", PortType.BOOL, default=False,
                       description="When True, append (observations, action, "
                                   "episode_id) to the buffer on execute.")
        self.add_input("batch_size", PortType.INT, default=32)
        self.add_input("shuffle", PortType.BOOL, default=True)
        self.add_input("chunk_size", PortType.INT, default=1,
                       description="Temporal window length for chunk_columns. "
                                   "1 = no chunking. Clamped at episode "
                                   "boundaries.")
        self.add_input("chunk_columns", PortType.STRING, default="",
                       description="Comma-separated column names to window "
                                   "by chunk_size (e.g. 'action').")
        self.add_input("save_to", PortType.STRING, default="",
                       description="Parquet path. Manual save via the "
                                   "inspector 'Save Now' button.")
        self.add_input("load_from", PortType.STRING, default="",
                       description="Parquet path. Loads once when a new path "
                                   "is set.")

        # Always-present outputs
        self.add_output("dataloader", PortType.DATALOADER,
                        description="Training DataLoader over the current "
                                    "buffer contents.")
        self.add_output("action", PortType.TENSOR,
                        description="Passthrough of the action input (useful "
                                    "for wiring action into loss branches).")
        self.add_output("n_samples", PortType.INT,
                        description="Current buffer fill level.")
        self.add_output("episode_id", PortType.INT,
                        description="Current episode index.")
        self.add_output("info", PortType.STRING)

        # Build observation ports from the default name list so templates that
        # never touch the string input still get working ports.
        self._declare_observations(self.inputs["observation_names"].default_value)

    def _declare_observations(self, names: str | list[str]) -> None:
        """Create matching input + output ports for each observation name."""
        if isinstance(names, str):
            name_list = [c.strip() for c in names.split(",") if c.strip()]
        else:
            name_list = [str(c).strip() for c in names if str(c).strip()]
        self.inputs["observation_names"].default_value = ",".join(name_list)

        # Add missing ports. We never remove ports dynamically — stale ones
        # just become disconnected.
        for name in name_list:
            if name not in self.inputs:
                self.add_input(name, PortType.TENSOR, default=None,
                               description=f"Observation '{name}' (write face)")
            if name not in self.outputs:
                self.add_output(name, PortType.TENSOR,
                                description=f"Observation '{name}' (read face)")
        self._obs_names = tuple(name_list)
        self._declared = True

    # ── Execute ─────────────────────────────────────────────────────────

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import torch
        from torch.utils.data import DataLoader

        names_raw = str(inputs.get("observation_names") or "")
        if names_raw:
            self._declare_observations(names_raw)

        maxlen = max(1, int(inputs.get("buffer_size") or 10000))
        if maxlen != self._maxlen:
            # Rebuild deque with new capacity, preserving recent samples.
            self._buffer = deque(list(self._buffer)[-maxlen:], maxlen=maxlen)
            self._maxlen = maxlen

        load_path = str(inputs.get("load_from") or "").strip()
        if load_path and load_path != self._last_load_path:
            self._load_parquet(load_path)
            self._last_load_path = load_path

        record = bool(inputs.get("record", False))
        action = inputs.get("action")
        if record and action is not None:
            rec = {"action": _to_tensor(action),
                   "episode_id": self._episode,
                   "t": self._t}
            missing = []
            for name in self._obs_names:
                v = inputs.get(name)
                if v is None:
                    missing.append(name)
                    continue
                rec[name] = _to_tensor(v)
            if not missing:
                self._buffer.append(rec)
                self._t += 1

        batch_size = max(1, int(inputs.get("batch_size") or 32))
        shuffle    = bool(inputs.get("shuffle", True))
        chunk_size = max(1, int(inputs.get("chunk_size") or 1))
        chunk_cols_raw = str(inputs.get("chunk_columns") or "")
        chunk_cols = tuple(c.strip() for c in chunk_cols_raw.split(",") if c.strip())

        loader = None
        n = len(self._buffer)
        if n >= batch_size:
            inner = _BufferDataset(self._buffer, self._obs_names)
            if chunk_size > 1 and chunk_cols:
                episode_ids = [rec.get("episode_id") for rec in self._buffer]
                inner = ChunkedWrapper(inner, chunk_size=chunk_size,
                                       chunk_columns=chunk_cols,
                                       episode_ids=episode_ids)
            try:
                loader = DataLoader(
                    inner, batch_size=batch_size, shuffle=shuffle,
                    drop_last=True, collate_fn=_collate,
                )
            except Exception as exc:
                return self._empty_output(action, info=f"DataLoader error: {exc}")

        info = (f"buffer={n}/{self._maxlen}  episode={self._episode}  "
                f"t={self._t}  obs=[{','.join(self._obs_names)}]")

        out: dict[str, Any] = {
            "dataloader": loader,
            "action": action,
            "n_samples": n,
            "episode_id": self._episode,
            "info": info,
        }
        # Also surface the latest raw observations on the read-face outputs
        # so the same graph can feed them into either a training branch
        # (via the DataLoader) or a live-deploy branch (via direct wires).
        for name in self._obs_names:
            out[name] = inputs.get(name)
        return out

    def _empty_output(self, action, info: str) -> dict[str, Any]:
        out: dict[str, Any] = {
            "dataloader": None,
            "action": action,
            "n_samples": len(self._buffer),
            "episode_id": self._episode,
            "info": info,
        }
        for name in self._obs_names:
            out[name] = None
        return out

    # ── Inspector UI ────────────────────────────────────────────────────

    def inspector_ui(self, parent: str, app) -> None:
        import dearpygui.dearpygui as dpg

        dpg.add_separator(parent=parent)
        dpg.add_text("Buffer Controls", parent=parent, color=[88, 196, 245])

        # Live status. These are text widgets — the inspector rebuilds on
        # re-selection, so no per-frame refresh needed. Click a button and
        # re-select the node to see updated counts.
        n = len(self._buffer)
        dpg.add_text(f"Samples: {n} / {self._maxlen}", parent=parent)
        dpg.add_text(f"Episode: {self._episode}", parent=parent)
        dpg.add_text(f"Step (t): {self._t}", parent=parent)

        dpg.add_separator(parent=parent)
        with dpg.group(horizontal=True, parent=parent):
            dpg.add_button(label="Next Episode",
                           callback=lambda: self._on_next_episode(app))
            dpg.add_button(label="Reset t",
                           callback=lambda: self._on_reset_t(app))
        with dpg.group(horizontal=True, parent=parent):
            dpg.add_button(label="Clear Buffer",
                           callback=lambda: self._on_clear(app))
            dpg.add_button(label="Save Now",
                           callback=lambda: self._on_save_now(app))

    def _on_next_episode(self, app) -> None:
        self._episode += 1
        self._t = 0
        try:
            app._log(f"[StreamingBuffer] episode -> {self._episode}")
        except Exception:
            pass
        app._update_inspector(self.id)

    def _on_reset_t(self, app) -> None:
        self._t = 0
        app._update_inspector(self.id)

    def _on_clear(self, app) -> None:
        self._buffer.clear()
        self._t = 0
        try:
            app._log("[StreamingBuffer] cleared")
        except Exception:
            pass
        app._update_inspector(self.id)

    def _on_save_now(self, app) -> None:
        path = self.inputs["save_to"].default_value
        # Pick up edited value from the live widget if it exists
        widget_tag = getattr(app, "input_widgets", {}).get((self.id, "save_to"))
        if widget_tag is not None:
            try:
                import dearpygui.dearpygui as dpg
                path = dpg.get_value(widget_tag) or path
            except Exception:
                pass
        if not path:
            try:
                app._log("[StreamingBuffer] save_to is empty")
            except Exception:
                pass
            return
        try:
            self._save_parquet(str(path))
            app._log(f"[StreamingBuffer] saved {len(self._buffer)} samples to {path}")
        except Exception as exc:
            app._log(f"[StreamingBuffer] save failed: {exc}")

    # ── Persistence ─────────────────────────────────────────────────────

    def _save_parquet(self, path: str) -> None:
        """Flush the buffer to a parquet file. Tensor columns stored as lists."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        all_cols = list(self._obs_names) + list(_FIXED_COLS)
        data: dict[str, list] = {c: [] for c in all_cols}
        for rec in self._buffer:
            for c in all_cols:
                v = rec.get(c)
                if hasattr(v, "tolist"):
                    data[c].append(v.tolist())
                else:
                    data[c].append(v)
        table = pa.table(data)
        pq.write_table(table, path)

    def _load_parquet(self, path: str) -> None:
        import pyarrow.parquet as pq
        import torch

        if not Path(path).exists():
            return
        table = pq.read_table(path)
        cols = table.column_names
        n_rows = table.num_rows
        # Add any observation columns we didn't already know about
        obs_from_file = [c for c in cols if c not in _FIXED_COLS]
        if obs_from_file and tuple(obs_from_file) != self._obs_names:
            self._declare_observations(obs_from_file)
        pyd = table.to_pydict()
        for i in range(n_rows):
            rec: dict[str, Any] = {}
            for c in cols:
                v = pyd[c][i]
                if c in _FIXED_COLS and c != "action":
                    rec[c] = v
                else:
                    if isinstance(v, list):
                        rec[c] = torch.tensor(v, dtype=torch.float32)
                    elif v is None:
                        rec[c] = None
                    else:
                        rec[c] = torch.tensor(float(v), dtype=torch.float32)
            self._buffer.append(rec)
        # Resume episode/t from the last loaded record so further recording
        # continues cleanly instead of overwriting.
        if self._buffer:
            last = self._buffer[-1]
            self._episode = int(last.get("episode_id") or 0)
            self._t = int(last.get("t") or 0) + 1

    # ── Export ──────────────────────────────────────────────────────────

    def export(self, iv, ov):
        dl_var = ov.get("dataloader", "_buf_dl")
        return ["import torch", "from torch.utils.data import DataLoader"], [
            "# StreamingBufferNode — live-only, no offline export target",
            f"{dl_var} = None  # populated at runtime from a live producer",
        ]


def _to_tensor(v):
    """Coerce a value to a CPU float32 tensor for buffer storage."""
    import torch
    if isinstance(v, torch.Tensor):
        return v.detach().to("cpu")
    try:
        return torch.as_tensor(v, dtype=torch.float32)
    except Exception:
        return torch.tensor(0.0)
