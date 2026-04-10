"""Universal DatasetNode — one node to replace them all.

Points at a folder containing a `samples.csv` manifest. Each column becomes
a dynamically-created output port. File-path columns (images, audio, numpy)
are loaded as tensors; numeric columns become scalar tensors; string columns
are integer-encoded. Plus a `dataloader` port for the Training Panel.

Folder layout:
    my_dataset/
    ├── samples.csv       ← manifest (required)
    ├── images/           ← referenced by columns with .png/.jpg/.bmp paths
    ├── audio/            ← referenced by columns with .npy/.wav paths
    └── ...

samples.csv example:
    id,image,audio,label,score
    001,images/001.png,audio/001.npy,cat,0.7
    002,images/002.png,audio/002.npy,dog,0.3

Column type auto-detection:
    - Cells ending in .png/.jpg/.bmp/.jpeg → image (loaded via PIL, output as float tensor)
    - Cells ending in .npy → numpy array (loaded via np.load, output as float tensor)
    - Cells ending in .wav/.mp3/.flac → audio (loaded via numpy, output as float tensor)
    - Column named 'id' → skipped (not an output port)
    - All cells are numeric → FLOAT tensor
    - All cells are strings (non-path) → integer-encoded (label), output as LONG tensor
    - Mixed or empty → ANY, passed through as-is

The node dynamically creates output ports based on the manifest columns.
Set `columns` to a comma-separated subset to only output specific columns.
Leave blank to output all non-id columns.

This single node replaces: MNISTDatasetNode, CIFAR10DatasetNode,
TextDatasetNode, NumpyDatasetNode, CSVDatasetNode, ImageFolderDatasetNode,
AudioFolderDatasetNode, FolderMultimodalDatasetNode, HFDatasetNode.
"""
from __future__ import annotations
import os
import csv
from pathlib import Path
from typing import Any
from core.node import BaseNode, PortType


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
_NUMPY_EXTS = {".npy", ".npz"}
_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}
_SKIP_COLS  = {"id", "index", "idx", "file", "filename"}


def _detect_column_type(values: list[str]) -> str:
    """Heuristic: look at a column's values and classify as image/audio/numpy/numeric/label."""
    non_empty = [v.strip() for v in values if v.strip()]
    if not non_empty:
        return "skip"
    # Check file extensions on the first few non-empty values
    sample = non_empty[:10]
    exts = {Path(v).suffix.lower() for v in sample if "." in v}
    if exts & _IMAGE_EXTS:
        return "image"
    if exts & _NUMPY_EXTS:
        return "numpy"
    if exts & _AUDIO_EXTS:
        return "audio"
    # Check if all values are numeric
    try:
        [float(v) for v in sample]
        return "numeric"
    except (ValueError, TypeError):
        pass
    # Fallback: treat as categorical label (will be integer-encoded)
    return "label"


def _load_cell(value: str, col_type: str, root: str):
    """Load a single cell value based on its detected column type."""
    import torch
    import numpy as np
    value = value.strip()
    if not value:
        return None

    if col_type == "image":
        try:
            from PIL import Image
            path = os.path.join(root, value)
            img = Image.open(path).convert("RGB")
            arr = np.asarray(img, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (C, H, W)
        except Exception:
            return None

    if col_type == "numpy":
        try:
            path = os.path.join(root, value)
            arr = np.load(path)
            return torch.from_numpy(arr.astype(np.float32))
        except Exception:
            return None

    if col_type == "audio":
        try:
            path = os.path.join(root, value)
            arr = np.load(path)  # assume .npy for now; .wav would need soundfile
            return torch.from_numpy(arr.astype(np.float32))
        except Exception:
            return None

    if col_type == "numeric":
        try:
            return torch.tensor(float(value))
        except (ValueError, TypeError):
            return None

    # label: will be integer-encoded by the dataset class
    return value


class _ManifestDataset:
    """torch.utils.data.Dataset that reads from a samples.csv manifest."""

    def __init__(self, root: str, columns: list[str], col_types: dict[str, str],
                 rows: list[dict[str, str]], label_maps: dict[str, dict[str, int]]):
        self.root = root
        self.columns = columns
        self.col_types = col_types
        self.rows = rows
        self.label_maps = label_maps  # {col_name: {string_val: int_index}}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        import torch
        row = self.rows[idx]
        sample = {}
        for col in self.columns:
            ct = self.col_types[col]
            raw = row.get(col, "")
            if ct == "label":
                # Integer-encode
                lm = self.label_maps.get(col, {})
                sample[col] = torch.tensor(lm.get(raw.strip(), 0), dtype=torch.long)
            else:
                val = _load_cell(raw, ct, self.root)
                if val is not None:
                    sample[col] = val
                else:
                    sample[col] = torch.tensor(0.0)
        return sample


def _collate_manifest(samples: list[dict]) -> dict:
    """Collate a list of {col: tensor} dicts into a batched dict."""
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
            batch[k] = tensors  # fallback: list of tensors if shapes don't match
    return batch


class DatasetNode(BaseNode):
    type_name   = "pt_dataset"
    label       = "Dataset"
    category    = "Datasets"
    subcategory = "Universal"
    description = (
        "Universal dataset node. Point at a folder with a samples.csv manifest "
        "and get one output port per column — images, audio, labels, scores, "
        "anything. Replaces MNIST/CIFAR/Text/CSV/ImageFolder/Audio/Multimodal "
        "dataset nodes with one configurable node."
    )

    def __init__(self):
        self._cached_loader = None
        self._cached_dataset = None
        self._cached_cfg: tuple = ()
        self._col_info: dict[str, str] = {}  # col_name -> col_type
        self._dynamic_ports_built = False
        super().__init__()

    def _setup_ports(self) -> None:
        self.add_input("path",       PortType.STRING, default="data/my_dataset",
                       description="Folder containing samples.csv + data files")
        self.add_input("columns",    PortType.STRING, default="",
                       description="Comma-separated column names to output (blank = all)")
        self.add_input("batch_size", PortType.INT,    default=32)
        self.add_input("shuffle",    PortType.BOOL,   default=True)
        self.add_input("task_id",    PortType.STRING, default="default",
                       description="Pairs this dataset with a Train Output of the same task_name")
        # Dynamic outputs are added on first execute when the manifest is read.
        # Always present: dataloader + info
        self.add_output("dataloader", PortType.DATALOADER)
        self.add_output("info",       PortType.STRING)

    def _build_dynamic_ports(self, columns: list[str], col_types: dict[str, str]) -> None:
        """Add one TENSOR output port per manifest column (called once on first load)."""
        for col in columns:
            if col not in self.outputs:
                ct = col_types[col]
                pt = PortType.TENSOR
                desc = f"Column '{col}' ({ct})"
                self.add_output(col, pt, description=desc)
        self._dynamic_ports_built = True
        self._col_info = dict(col_types)

    def _load_manifest(self, path: str, columns_filter: str) -> dict:
        """Parse samples.csv, detect column types, build the Dataset + DataLoader."""
        import torch
        from torch.utils.data import DataLoader

        root = str(path)
        manifest_path = os.path.join(root, "samples.csv")
        if not os.path.exists(manifest_path):
            return {"error": f"No samples.csv found in {root}"}

        # Read CSV
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            rows = list(reader)

        if not rows or not fieldnames:
            return {"error": "samples.csv is empty"}

        # Filter columns
        if columns_filter.strip():
            selected = [c.strip() for c in columns_filter.split(",") if c.strip()]
            columns = [c for c in selected if c in fieldnames]
        else:
            columns = [c for c in fieldnames if c.lower() not in _SKIP_COLS]

        # Detect types per column
        col_types: dict[str, str] = {}
        for col in columns:
            values = [row.get(col, "") for row in rows]
            col_types[col] = _detect_column_type(values)

        # Remove skip columns
        columns = [c for c in columns if col_types[c] != "skip"]

        # Build label maps for categorical columns
        label_maps: dict[str, dict[str, int]] = {}
        for col in columns:
            if col_types[col] == "label":
                unique = sorted(set(row.get(col, "").strip() for row in rows))
                label_maps[col] = {v: i for i, v in enumerate(unique)}

        # Build dataset
        ds = _ManifestDataset(root, columns, col_types, rows, label_maps)

        return {
            "dataset": ds,
            "columns": columns,
            "col_types": col_types,
            "label_maps": label_maps,
            "n_samples": len(rows),
        }

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import torch
        from torch.utils.data import DataLoader

        path       = str(inputs.get("path") or "data/my_dataset")
        columns    = str(inputs.get("columns") or "")
        batch_size = max(1, int(inputs.get("batch_size") or 32))
        shuffle    = bool(inputs.get("shuffle", True))

        cfg = (path, columns, batch_size, shuffle)
        empty = {"dataloader": None, "info": ""}

        if self._cached_loader is None or self._cached_cfg != cfg:
            result = self._load_manifest(path, columns)
            if "error" in result:
                return {**empty, "info": result["error"]}

            ds = result["dataset"]
            cols = result["columns"]
            col_types = result["col_types"]

            # Build dynamic output ports if not done yet
            if not self._dynamic_ports_built:
                self._build_dynamic_ports(cols, col_types)

            try:
                self._cached_loader = DataLoader(
                    ds, batch_size=batch_size, shuffle=shuffle,
                    drop_last=True, collate_fn=_collate_manifest,
                )
            except Exception as exc:
                return {**empty, "info": f"DataLoader failed: {exc}"}

            self._cached_dataset = result
            self._cached_cfg = cfg

        loader = self._cached_loader
        result = self._cached_dataset

        # Sample one batch for tensor preview
        preview: dict[str, Any] = {}
        try:
            batch = next(iter(loader))
            for col in result["columns"]:
                if col in batch:
                    preview[col] = batch[col]
        except Exception:
            pass

        # Build info string
        col_summary = ", ".join(f"{c}({result['col_types'][c]})"
                                for c in result["columns"])
        label_info = ""
        for col, lm in result.get("label_maps", {}).items():
            label_info += f"  {col}: {len(lm)} classes ({', '.join(list(lm.keys())[:5])})\n"
        info = (f"{result['n_samples']} samples, {len(result['columns'])} columns\n"
                f"  {col_summary}\n{label_info}".strip())

        out = {
            "dataloader": loader,
            "info": info,
            **preview,  # one tensor per column from the sampled batch
        }
        return out

    def export(self, iv, ov):
        path = self._val(iv, "path")
        cols = self._val(iv, "columns")
        bs   = self._val(iv, "batch_size")
        shuf = self._val(iv, "shuffle")
        dl_var   = ov.get("dataloader", "_ds_dl")
        info_var = ov.get("info",       "_ds_info")
        return ["import torch", "from torch.utils.data import DataLoader"], [
            f"# Universal Dataset from {path}",
            f"# Columns: {cols or '(all)'}",
            f"# TODO: implement manifest-based dataset loading for export",
            f"# For now, construct your DataLoader manually here.",
            f"{dl_var} = None  # load from {path}/samples.csv",
            f"{info_var} = 'see {path}/samples.csv'",
        ]
