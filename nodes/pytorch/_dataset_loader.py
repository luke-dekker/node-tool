"""Universal DatasetNode — one node for any dataset source.

Point it at a folder, file, HuggingFace repo, or built-in name and it
figures out the format, creates dynamic output ports per column, and
provides a DataLoader for the Training Panel.

Supported sources (auto-detected from path):

  Local folder with clips/ + *.tsv  → Common Voice mode (Mozilla CV)
  Local folder with samples.csv     → CSV manifest mode
  Local folder with top-level *.tsv → TSV manifest mode (generic)
  Local folder with *.parquet       → Parquet mode (HF/LeRobot)
  Local folder with class subfolders of images → ImageFolder mode
  Local .txt file                   → Character-level text mode
  Local .csv / .tsv file            → Single manifest file
  Local .parquet file               → Single Parquet file mode
  Local .json / .jsonl file         → JSON manifest mode
  "mnist", "cifar10", etc.          → Torchvision built-in
  "org/repo" (not a local path)     → HuggingFace Hub download

Column type detection:
  .png/.jpg/.bmp paths → image (loaded as float tensor)
  .npy paths           → numpy array (loaded as float tensor)
  .wav/.mp3/.flac/.ogg → audio waveform via torchaudio (mono FloatTensor;
                          auto-resampled to 16 kHz unless overridden)
  all numeric          → FLOAT tensor
  all strings          → integer-encoded label (LONG tensor)
  long strings (>40c)  → text passthrough (e.g. CV `sentence`)
  "id"/"index"         → skipped

Common Voice quirks:
  - clips are 48 kHz mp3 → resampled to 16 kHz mono on load
  - splits live as `train.tsv` / `dev.tsv` / `test.tsv` / `validated.tsv`;
    the dataset node's `split` input picks which TSV to read
  - text column is `sentence`; audio column is `path` (relative to clips/)
"""
from __future__ import annotations
import os
import csv
import json
from pathlib import Path
from typing import Any
from core.node import BaseNode, PortType


# ── Constants ───────────────────────────────────────────────────────────

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
_NUMPY_EXTS = {".npy", ".npz"}
_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg"}
_SKIP_COLS  = {"id", "index", "idx", "file", "filename"}
_PARQUET_SKIP = {"timestamp", "frame_index", "episode_index", "index",
                 "episode_data_index_from", "episode_data_index_to",
                 "task_index", "next.done", "next.reward"}

_TORCHVISION_BUILTINS = {
    "mnist": ("MNIST", 1, 28),
    "fashion_mnist": ("FashionMNIST", 1, 28),
    "fashionmnist": ("FashionMNIST", 1, 28),
    "cifar10": ("CIFAR10", 3, 32),
    "cifar100": ("CIFAR100", 3, 32),
}

# Fallback corpus for text mode when path doesn't exist
_FALLBACK_CORPUS = (
    "the quick brown fox jumps over the lazy dog. " * 200
    + "pack my box with five dozen liquor jugs. " * 200
    + "how vexingly quick daft zebras jump! " * 200
)


# ── Format detection ────────────────────────────────────────────────────

def _detect_format(path: str) -> str:
    """Determine the dataset format from the path string.

    Returns one of: common_voice, csv_manifest, tsv_manifest, parquet_dir,
    imagefolder, text, csv_file, tsv_file, parquet_file, json_file,
    torchvision, huggingface, unknown.
    """
    low = path.strip().lower()

    # Built-in torchvision datasets
    if low in _TORCHVISION_BUILTINS:
        return "torchvision"

    # Check file extension first (works whether file exists or not)
    ext = Path(path).suffix.lower()
    if ext == ".txt":
        return "text"
    if ext == ".csv":
        return "csv_file"
    if ext == ".tsv":
        return "tsv_file"
    if ext == ".parquet":
        return "parquet_file"
    if ext in (".json", ".jsonl"):
        return "json_file"

    # Existing single file with unrecognized extension
    if os.path.isfile(path):
        return "unknown"

    # Local folder — detect from contents
    if os.path.isdir(path):
        # Common Voice: clips/ subdir + at least one canonical CV TSV
        clips = os.path.join(path, "clips")
        if os.path.isdir(clips):
            for cv_tsv in ("validated.tsv", "train.tsv", "dev.tsv", "test.tsv"):
                if os.path.exists(os.path.join(path, cv_tsv)):
                    return "common_voice"
        # HF / LeRobot format (meta/info.json + data/*.parquet)
        if os.path.exists(os.path.join(path, "meta", "info.json")):
            return "parquet_dir"
        # CSV manifest
        if os.path.exists(os.path.join(path, "samples.csv")):
            return "csv_manifest"
        # Generic TSV manifest (any .tsv at top level)
        try:
            if any(p.suffix.lower() == ".tsv"
                   for p in Path(path).iterdir() if p.is_file()):
                return "tsv_manifest"
        except (PermissionError, OSError):
            pass
        # Parquet files anywhere in the folder
        pq = list(Path(path).rglob("*.parquet"))
        if pq:
            return "parquet_dir"
        # JSON manifest
        for name in ("data.json", "data.jsonl", "manifest.json", "manifest.jsonl"):
            if os.path.exists(os.path.join(path, name)):
                return "json_file"
        # ImageFolder: subfolders containing image files
        try:
            for entry in os.scandir(path):
                if entry.is_dir():
                    sample = next(Path(entry.path).iterdir(), None)
                    if sample and sample.suffix.lower() in _IMAGE_EXTS:
                        return "imagefolder"
        except (PermissionError, OSError):
            pass
        return "unknown"

    # Not a local file or folder — could be an HF repo ID
    # HF repo IDs look like "org/dataset" — no drive letters, no backslashes
    stripped = path.strip()
    if ("/" in stripped
            and not stripped.startswith(("/", "\\", "~", "."))
            and ":" not in stripped
            and "\\" not in stripped):
        return "huggingface"

    return "unknown"


# ── Column type detection (for CSV / Parquet / JSON) ────────────────────

def _detect_column_type(values: list[str]) -> str:
    """Classify a column from sample values:
    image / numpy / audio / numeric / label / text / skip."""
    non_empty = [v.strip() for v in values if v.strip()]
    if not non_empty:
        return "skip"
    sample = non_empty[:10]
    exts = {Path(v).suffix.lower() for v in sample if "." in v}
    if exts & _IMAGE_EXTS:
        return "image"
    if exts & _NUMPY_EXTS:
        return "numpy"
    if exts & _AUDIO_EXTS:
        return "audio"
    try:
        [float(v) for v in sample]
        return "numeric"
    except (ValueError, TypeError):
        pass
    # Strings: distinguish short categorical labels from long-form text
    # (e.g. Common Voice `sentence` column). >40 chars or whitespace inside
    # a token = text passthrough rather than label-encode.
    if any(len(v) > 40 or " " in v for v in sample):
        return "text"
    return "label"


# Cache for torchaudio resamplers keyed by (orig_sr, target_sr) — building one
# is expensive, but the same dataset usually has a single source sample rate.
_RESAMPLERS: dict[tuple[int, int], Any] = {}


def _load_audio(abs_path: str, target_sr: int = 16000):
    """Load `abs_path` as a mono FloatTensor of shape (samples,) at `target_sr`.

    Tries torchaudio first (handles wav/mp3/flac/ogg via the configured
    backend; ffmpeg-on-PATH gets you mp3 on Windows). Falls back to
    soundfile for wav/flac/ogg if torchaudio is missing. Returns None on
    failure so the dataset call site can keep iterating.
    """
    import torch
    try:
        import torchaudio
    except ImportError:
        torchaudio = None
    if torchaudio is not None:
        try:
            wav, sr = torchaudio.load(abs_path)   # (channels, samples)
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            if sr != target_sr:
                key = (sr, target_sr)
                rs = _RESAMPLERS.get(key)
                if rs is None:
                    rs = torchaudio.transforms.Resample(sr, target_sr)
                    _RESAMPLERS[key] = rs
                wav = rs(wav)
            return wav.squeeze(0).contiguous()
        except Exception:
            pass
    # soundfile fallback (no mp3 support, but covers wav/flac/ogg)
    try:
        import soundfile as sf
        import numpy as np
        data, sr = sf.read(abs_path, dtype="float32", always_2d=True)
        mono = data.mean(axis=1)
        if sr != target_sr:
            # Cheap linear resample fallback — good enough for sanity tests
            n = int(round(len(mono) * (target_sr / sr)))
            xp = np.linspace(0, 1, len(mono), endpoint=False)
            x  = np.linspace(0, 1, n,           endpoint=False)
            mono = np.interp(x, xp, mono).astype(np.float32)
        return torch.from_numpy(mono)
    except Exception:
        return None


def _load_cell(value: str, col_type: str, root: str, target_sr: int = 16000):
    """Load a single cell value based on its detected column type."""
    import torch
    import numpy as np
    value = value.strip()
    if not value:
        return None
    if col_type == "image":
        try:
            from PIL import Image
            img = Image.open(os.path.join(root, value)).convert("RGB")
            arr = np.asarray(img, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        except Exception:
            return None
    if col_type == "numpy":
        try:
            arr = np.load(os.path.join(root, value))
            return torch.from_numpy(arr.astype(np.float32))
        except Exception:
            return None
    if col_type == "audio":
        # Audio path — load via torchaudio. NOTE: the previous implementation
        # incorrectly did `np.load(...)` here, which crashed on every real
        # .wav/.mp3 file. Now produces a mono FloatTensor at `target_sr`.
        return _load_audio(os.path.join(root, value), target_sr=target_sr)
    if col_type == "numeric":
        try:
            return torch.tensor(float(value))
        except (ValueError, TypeError):
            return None
    if col_type == "text":
        return value   # raw string passthrough
    return value      # label — integer-encoded by the dataset class


# ── Dataset wrappers ────────────────────────────────────────────────────

class _ManifestDataset:
    """Dataset from a row-oriented manifest (CSV, TSV, JSON, Parquet).

    `audio_root` overrides where audio paths are resolved (Common Voice clips
    live under `<root>/clips/`, not at the manifest root). `target_sr` is the
    sample rate to resample audio to on load (default 16 kHz).
    """

    def __init__(self, root, columns, col_types, rows, label_maps,
                 audio_root: str | None = None, target_sr: int = 16000):
        self.root        = root
        self.audio_root  = audio_root or root
        self.target_sr   = target_sr
        self.columns     = columns
        self.col_types   = col_types
        self.rows        = rows
        self.label_maps  = label_maps

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
                lm = self.label_maps.get(col, {})
                sample[col] = torch.tensor(lm.get(str(raw).strip(), 0), dtype=torch.long)
            elif ct == "audio":
                # Audio uses audio_root (Common Voice clips/ subdir, etc.).
                val = _load_cell(str(raw), ct, self.audio_root, target_sr=self.target_sr)
                sample[col] = val if val is not None else torch.tensor(0.0)
            else:
                val = _load_cell(str(raw), ct, self.root, target_sr=self.target_sr)
                sample[col] = val if val is not None else torch.tensor(0.0)
        return sample


def _collate_manifest(samples):
    """Collate list of {col: tensor|str} dicts into a batched dict.

    Variable-length tensors (audio waveforms, target strings) can't be stacked
    cleanly — we fall back to returning the per-sample list so the downstream
    graph can pad / pack as needed (see PackSequenceNode for CTC).
    """
    import torch
    if not samples:
        return {}
    keys = list(samples[0].keys())
    batch = {}
    for k in keys:
        vals = [s[k] for s in samples]
        # Pure string column → list passthrough (e.g. CV `sentence`).
        if all(isinstance(v, str) for v in vals):
            batch[k] = vals
            continue
        try:
            batch[k] = torch.stack(vals)
        except Exception:
            # Variable-length tensors / mixed shapes — keep as list.
            batch[k] = vals
    return batch


# ── Loaders (one per format) ────────────────────────────────────────────

def _load_csv_manifest(root, columns_filter, target_sr=16000):
    """Load from a folder containing samples.csv."""
    manifest_path = os.path.join(root, "samples.csv")
    if not os.path.exists(manifest_path):
        return None, "No samples.csv found"
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    if not rows:
        return None, "samples.csv is empty"
    return _build_manifest(root, fieldnames, rows, columns_filter,
                           target_sr=target_sr)


def _load_csv_file(path, columns_filter, target_sr=16000):
    """Load from a single .csv file."""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    if not rows:
        return None, "CSV file is empty"
    root = str(Path(path).parent)
    return _build_manifest(root, fieldnames, rows, columns_filter,
                           target_sr=target_sr)


def _read_tsv(path: str) -> tuple[list[str], list[dict]]:
    """Tiny TSV reader (Common Voice format). Returns (fieldnames, rows)."""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    return fieldnames, rows


def _load_tsv_file(path, columns_filter, target_sr=16000):
    """Load from a single .tsv file."""
    fieldnames, rows = _read_tsv(path)
    if not rows:
        return None, f"TSV file {path} is empty"
    root = str(Path(path).parent)
    return _build_manifest(root, fieldnames, rows, columns_filter,
                           target_sr=target_sr)


def _load_tsv_manifest(root, columns_filter, split, target_sr=16000):
    """Load from a folder containing one or more .tsv files. `split` picks
    which file (e.g. 'train' → train.tsv); falls back to the first .tsv."""
    candidates = [f"{split}.tsv", "train.tsv", "validated.tsv", "data.tsv"]
    chosen = next((c for c in candidates
                   if os.path.exists(os.path.join(root, c))), None)
    if chosen is None:
        # Pick any .tsv at top level
        for p in Path(root).iterdir():
            if p.is_file() and p.suffix.lower() == ".tsv":
                chosen = p.name
                break
    if chosen is None:
        return None, f"No .tsv files found in {root}"
    return _load_tsv_file(os.path.join(root, chosen), columns_filter,
                          target_sr=target_sr)


def _load_common_voice(root, columns_filter, split, target_sr=16000):
    """Load a Mozilla Common Voice extract.

    Layout: `<root>/clips/*.mp3` + `<root>/{train,dev,test,validated}.tsv`.
    The TSV `path` column is a clip filename relative to `clips/`. Standard
    columns: client_id, path, sentence_id, sentence, sentence_domain,
    up_votes, down_votes, age, gender, accents, variant, locale, segment.

    `split` picks the TSV. If the requested split doesn't exist (e.g. 'val'
    on a CV dump that only has dev), fall back to validated.tsv.
    """
    # Map common-but-non-CV split names
    split_aliases = {"val": "dev", "validation": "dev"}
    split_name = split_aliases.get(split, split)
    tsv_path = os.path.join(root, f"{split_name}.tsv")
    if not os.path.exists(tsv_path):
        # Fallbacks in priority order
        for fallback in ("validated.tsv", "train.tsv", "dev.tsv", "test.tsv"):
            cand = os.path.join(root, fallback)
            if os.path.exists(cand):
                tsv_path = cand
                break
        else:
            return None, f"No CV TSV found in {root}"

    fieldnames, rows = _read_tsv(tsv_path)
    if not rows:
        return None, f"Common Voice TSV {tsv_path} is empty"

    # Default to (path, sentence) when the user didn't filter — keeps the
    # output node's surface tight for ASR. Other columns (gender, accents…)
    # are available via columns_filter.
    if not columns_filter.strip():
        columns_filter = "path,sentence"

    audio_root = os.path.join(root, "clips")
    return _build_manifest(root, fieldnames, rows, columns_filter,
                           audio_root=audio_root,
                           audio_col="path", text_col="sentence",
                           target_sr=target_sr,
                           cv_split=Path(tsv_path).stem)


def _load_parquet_dir(root, columns_filter):
    """Load from a folder with Parquet files (plain or HF/LeRobot structure)."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None, "pyarrow not installed. Run: pip install pyarrow"

    # Find parquet files — prefer data/ subfolder if it exists
    data_dir = os.path.join(root, "data")
    search_dir = data_dir if os.path.isdir(data_dir) else root
    pq_files = sorted(Path(search_dir).rglob("*.parquet"))
    # Exclude metadata parquet files
    pq_files = [f for f in pq_files if "meta" not in str(f.relative_to(root)).split(os.sep)]
    if not pq_files:
        return None, "No parquet data files found"

    # Read all parquet files into one table
    tables = [pq.read_table(str(f)) for f in pq_files]
    import pyarrow as pa
    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]

    # Convert to rows
    df_dict = table.to_pydict()
    columns_all = list(df_dict.keys())
    n_rows = len(next(iter(df_dict.values())))

    # Filter out metadata columns
    columns_data = [c for c in columns_all if c.lower() not in _PARQUET_SKIP]

    # Build rows as list of dicts
    rows = []
    for i in range(n_rows):
        row = {col: df_dict[col][i] for col in columns_data}
        rows.append(row)

    # Detect column types from parquet data
    return _build_parquet_manifest(root, columns_data, rows, columns_filter, table.schema)


def _load_parquet_file(path, columns_filter):
    """Load from a single .parquet file."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None, "pyarrow not installed. Run: pip install pyarrow"
    table = pq.read_table(path)
    df_dict = table.to_pydict()
    columns_all = list(df_dict.keys())
    n_rows = len(next(iter(df_dict.values()))) if df_dict else 0
    columns_data = [c for c in columns_all if c.lower() not in _PARQUET_SKIP]
    rows = [{col: df_dict[col][i] for col in columns_data} for i in range(n_rows)]
    root = str(Path(path).parent)
    return _build_parquet_manifest(root, columns_data, rows, columns_filter, table.schema)


def _build_parquet_manifest(root, columns_data, rows, columns_filter, schema):
    """Build a manifest dataset from parquet-derived rows."""
    import torch

    if columns_filter.strip():
        selected = [c.strip() for c in columns_filter.split(",") if c.strip()]
        columns = [c for c in selected if c in columns_data]
    else:
        columns = columns_data

    if not columns or not rows:
        return None, "No usable columns/rows in parquet data"

    # Detect types from actual values (parquet values are already typed)
    col_types = {}
    label_maps = {}
    for col in columns:
        sample_vals = [rows[i][col] for i in range(min(10, len(rows)))]
        # Check if values are numeric (int, float, list of numbers)
        if all(isinstance(v, (int, float)) for v in sample_vals if v is not None):
            col_types[col] = "numeric"
        elif all(isinstance(v, (list, tuple)) for v in sample_vals if v is not None):
            col_types[col] = "tensor_list"
        elif all(isinstance(v, str) for v in sample_vals if v is not None):
            # Check if it looks like file paths
            exts = {Path(str(v)).suffix.lower() for v in sample_vals if v}
            if exts & _IMAGE_EXTS:
                col_types[col] = "image"
            elif exts & _NUMPY_EXTS:
                col_types[col] = "numpy"
            else:
                col_types[col] = "label"
                unique = sorted(set(str(rows[i][col]).strip() for i in range(len(rows))))
                label_maps[col] = {v: i for i, v in enumerate(unique)}
        else:
            col_types[col] = "numeric"  # fallback

    class _ParquetDataset:
        def __init__(self, rows, columns, col_types, label_maps, root):
            self.rows = rows
            self.columns = columns
            self.col_types = col_types
            self.label_maps = label_maps
            self.root = root

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            row = self.rows[idx]
            sample = {}
            for col in self.columns:
                ct = self.col_types[col]
                val = row.get(col)
                if ct == "tensor_list":
                    sample[col] = torch.tensor(val, dtype=torch.float32)
                elif ct == "numeric":
                    sample[col] = torch.tensor(float(val) if val is not None else 0.0,
                                               dtype=torch.float32)
                elif ct == "label":
                    lm = self.label_maps.get(col, {})
                    sample[col] = torch.tensor(lm.get(str(val).strip(), 0), dtype=torch.long)
                elif ct == "image":
                    loaded = _load_cell(str(val), "image", self.root)
                    sample[col] = loaded if loaded is not None else torch.tensor(0.0)
                elif ct == "numpy":
                    loaded = _load_cell(str(val), "numpy", self.root)
                    sample[col] = loaded if loaded is not None else torch.tensor(0.0)
                else:
                    sample[col] = torch.tensor(float(val) if val is not None else 0.0)
            return sample

    ds = _ParquetDataset(rows, columns, col_types, label_maps, root)
    n = len(rows)
    col_summary = ", ".join(f"{c}({col_types[c]})" for c in columns)
    info = f"{n} samples, {len(columns)} columns\n  {col_summary}"
    return (ds, columns, col_types, label_maps, info), None


def _build_manifest(root, fieldnames, rows, columns_filter,
                    audio_root: str | None = None,
                    audio_col: str | None = None,
                    text_col: str | None = None,
                    target_sr: int = 16000,
                    cv_split: str | None = None):
    """Build a ManifestDataset from CSV/TSV/JSON rows.

    Optional kwargs (used by Common Voice / TSV-with-known-schema):
      audio_root  — override root for audio files (e.g. <root>/clips/ for CV)
      audio_col   — force a column to type "audio" (e.g. CV `path`)
      text_col    — force a column to type "text" (e.g. CV `sentence`)
      target_sr   — sample rate to resample audio to
      cv_split    — CV-specific tag for the info string ('train.tsv', etc.)
    """
    if columns_filter.strip():
        selected = [c.strip() for c in columns_filter.split(",") if c.strip()]
        columns = [c for c in selected if c in fieldnames]
    else:
        columns = [c for c in fieldnames if c.lower() not in _SKIP_COLS]

    col_types = {}
    for col in columns:
        values = [row.get(col, "") for row in rows]
        col_types[col] = _detect_column_type(values)

    # Schema hints from the caller (Common Voice etc.) override auto-detection
    if audio_col and audio_col in col_types:
        col_types[audio_col] = "audio"
    if text_col and text_col in col_types:
        col_types[text_col] = "text"

    columns = [c for c in columns if col_types[c] != "skip"]

    label_maps = {}
    for col in columns:
        if col_types[col] == "label":
            unique = sorted(set(row.get(col, "").strip() for row in rows))
            label_maps[col] = {v: i for i, v in enumerate(unique)}

    ds = _ManifestDataset(root, columns, col_types, rows, label_maps,
                          audio_root=audio_root, target_sr=target_sr)
    n = len(rows)
    col_summary = ", ".join(f"{c}({col_types[c]})" for c in columns)
    label_info = ""
    for col, lm in label_maps.items():
        label_info += f"\n  {col}: {len(lm)} classes ({', '.join(list(lm.keys())[:5])})"
    cv_tag = f" [{cv_split}]" if cv_split else ""
    info = f"{n} samples{cv_tag}, {len(columns)} columns\n  {col_summary}{label_info}"
    return (ds, columns, col_types, label_maps, info), None


def _load_json_file(path, columns_filter):
    """Load from a .json or .jsonl file."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        else:
            data = json.load(f)
            if isinstance(data, list):
                rows = data
            elif isinstance(data, dict) and "data" in data:
                rows = data["data"]
            else:
                return None, "JSON must be an array or {data: [...]}"
    if not rows:
        return None, "JSON file has no rows"
    fieldnames = list(rows[0].keys())
    # Normalize all values to strings for the manifest pipeline
    str_rows = [{k: str(v) for k, v in row.items()} for row in rows]
    root = str(Path(path).parent)
    return _build_manifest(root, fieldnames, str_rows, columns_filter)


def _load_imagefolder(root, columns_filter):
    """Load from a folder where subfolders = class labels, containing images."""
    import torch
    import numpy as np

    classes = sorted([d.name for d in Path(root).iterdir()
                      if d.is_dir() and not d.name.startswith(".")])
    if not classes:
        return None, "No class subfolders found"

    class_to_idx = {c: i for i, c in enumerate(classes)}
    samples = []  # (path, class_idx)
    for cls_name in classes:
        cls_dir = Path(root) / cls_name
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() in _IMAGE_EXTS:
                samples.append((str(img_path), class_to_idx[cls_name]))

    if not samples:
        return None, "No images found in class subfolders"

    class _ImageFolderDataset:
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            from PIL import Image
            path, label = self.samples[idx]
            img = Image.open(path).convert("RGB")
            arr = np.asarray(img, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
            return {"x": tensor, "label": torch.tensor(label, dtype=torch.long)}

    ds = _ImageFolderDataset(samples)
    columns = ["x", "label"]
    col_types = {"x": "image", "label": "label"}
    label_maps = {"label": class_to_idx}
    info = (f"ImageFolder: {len(samples)} images, {len(classes)} classes "
            f"({', '.join(classes[:5])}{'...' if len(classes) > 5 else ''})")
    return (ds, columns, col_types, label_maps, info), None


def _load_text(path, seq_len):
    """Load a text file for character-level language modeling."""
    import torch

    p = Path(path)
    if p.exists() and p.is_file():
        try:
            text = p.read_text(encoding="utf-8")
            source = str(p)
        except Exception:
            text = _FALLBACK_CORPUS
            source = "built-in fallback"
    else:
        text = _FALLBACK_CORPUS
        source = "built-in fallback"

    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    vocab_size = len(chars)

    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n_windows = (len(data) - 1) // seq_len
    if n_windows < 1:
        return None, f"Text too short for seq_len={seq_len}"

    usable = n_windows * seq_len
    x = data[:usable].reshape(n_windows, seq_len)
    y = data[1:usable + 1].reshape(n_windows, seq_len)

    from torch.utils.data import TensorDataset

    class _TextDatasetWrapper:
        def __init__(self, x, y):
            self._x = x
            self._y = y

        def __len__(self):
            return len(self._x)

        def __getitem__(self, idx):
            return {"x": self._x[idx], "label": self._y[idx]}

    ds = _TextDatasetWrapper(x, y)
    columns = ["x", "label"]
    col_types = {"x": "numeric", "label": "numeric"}
    label_maps = {}
    info = (f"text={source}  chars={len(text):,}  vocab={vocab_size}  "
            f"windows={n_windows:,}  seq_len={seq_len}")
    extra = {"vocab_size": vocab_size}
    return (ds, columns, col_types, label_maps, info, extra), None


def _load_torchvision(name, train, batch_size, shuffle):
    """Load a built-in torchvision dataset by name."""
    import torch

    low = name.strip().lower()
    tv_name, channels, size = _TORCHVISION_BUILTINS[low]

    try:
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader

        cls = getattr(datasets, tv_name)
        dataset = cls(root="./data", train=train, download=True,
                      transform=transforms.ToTensor())

        class _TVWrapper:
            def __init__(self, ds):
                self.ds = ds

            def __len__(self):
                return len(self.ds)

            def __getitem__(self, idx):
                x, label = self.ds[idx]
                return {"x": x, "label": torch.tensor(label, dtype=torch.long)}

        ds = _TVWrapper(dataset)
        columns = ["x", "label"]
        col_types = {"x": "image", "label": "label"}
        label_maps = {}
        n_classes = len(dataset.classes) if hasattr(dataset, "classes") else "?"
        info = f"{tv_name}: {len(dataset)} samples, {n_classes} classes, {channels}x{size}x{size}"
        return (ds, columns, col_types, label_maps, info), None
    except ImportError:
        return None, f"torchvision not installed. Run: pip install torchvision"
    except Exception as exc:
        return None, f"Failed to load {tv_name}: {exc}"


def _load_huggingface(repo_id, split, columns_filter):
    """Load a dataset from HuggingFace Hub."""
    import torch

    # Try lerobot first (for robotics datasets)
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        dataset = LeRobotDataset(repo_id)

        class _LRWrapper:
            def __init__(self, ds):
                self.ds = ds

            def __len__(self):
                return len(self.ds)

            def __getitem__(self, idx):
                item = self.ds[idx]
                sample = {}
                for k, v in item.items():
                    if k.lower() in _PARQUET_SKIP:
                        continue
                    if isinstance(v, torch.Tensor):
                        sample[k] = v.float() if v.is_floating_point() else v
                    elif isinstance(v, (int, float)):
                        sample[k] = torch.tensor(float(v), dtype=torch.float32)
                return sample

        ds = _LRWrapper(dataset)
        # Detect columns from first sample
        sample = dataset[0]
        columns = [k for k in sample.keys() if k.lower() not in _PARQUET_SKIP]
        if columns_filter.strip():
            selected = [c.strip() for c in columns_filter.split(",") if c.strip()]
            columns = [c for c in selected if c in columns]
        col_types = {}
        for c in columns:
            v = sample[c]
            if isinstance(v, torch.Tensor):
                col_types[c] = "tensor_list" if v.dim() > 0 else "numeric"
            else:
                col_types[c] = "numeric"
        label_maps = {}
        info = f"HF LeRobot: {repo_id} ({len(dataset)} samples, {len(columns)} features)"
        return (ds, columns, col_types, label_maps, info), None
    except (ImportError, Exception):
        pass

    # Fall back to HuggingFace datasets library
    try:
        from datasets import load_dataset
        hf_ds = load_dataset(repo_id, split=split or "train")

        class _HFWrapper:
            def __init__(self, ds):
                self.ds = ds

            def __len__(self):
                return len(self.ds)

            def __getitem__(self, idx):
                item = self.ds[idx]
                sample = {}
                for k, v in item.items():
                    if isinstance(v, (int, float)):
                        sample[k] = torch.tensor(float(v), dtype=torch.float32)
                    elif isinstance(v, str):
                        sample[k] = v  # will be handled by collate
                    elif isinstance(v, (list, tuple)):
                        sample[k] = torch.tensor(v, dtype=torch.float32)
                    elif isinstance(v, torch.Tensor):
                        sample[k] = v
                    else:
                        sample[k] = torch.tensor(0.0)
                return sample

        ds = _HFWrapper(hf_ds)
        columns = list(hf_ds.column_names)
        if columns_filter.strip():
            selected = [c.strip() for c in columns_filter.split(",") if c.strip()]
            columns = [c for c in selected if c in columns]
        col_types = {c: "numeric" for c in columns}
        label_maps = {}
        info = f"HF Dataset: {repo_id}[{split or 'train'}] ({len(hf_ds)} samples)"
        return (ds, columns, col_types, label_maps, info), None
    except ImportError:
        return None, "Neither lerobot nor datasets installed. Run: pip install datasets"
    except Exception as exc:
        return None, f"HuggingFace load failed: {exc}"


# ── The one node ────────────────────────────────────────────────────────

class DatasetNode(BaseNode):
    type_name   = "pt_dataset"
    label       = "Dataset"
    category    = "Datasets"
    subcategory = "Universal"
    description = (
        "Universal dataset node. Point at a folder, file, HuggingFace repo, "
        "or built-in name (mnist, cifar10). Auto-detects the format and "
        "creates one output port per data column."
    )

    def __init__(self):
        self._cached_loader = None
        self._cached_dataset = None
        self._cached_cfg: tuple = ()
        self._col_info: dict[str, str] = {}
        self._dynamic_ports_built = False
        self._extra: dict = {}
        super().__init__()

    def _setup_ports(self) -> None:
        self.add_input("path",       PortType.STRING, default="data/my_dataset",
                       description="Folder, file, HF repo ID, or built-in (mnist, cifar10)")
        self.add_input("columns",    PortType.STRING, default="",
                       description="Comma-separated column subset (blank = all)")
        self.add_input("batch_size", PortType.INT,    default=32)
        self.add_input("shuffle",    PortType.BOOL,   default=True)
        self.add_input("split",      PortType.STRING, default="train",
                       description="train/test/val (for built-ins and HF datasets)")
        self.add_input("seq_len",    PortType.INT,    default=0,
                       description="Sequence length for text mode (0 = disabled)")
        self.add_input("target_sample_rate", PortType.INT, default=16000,
                       description="Resample audio to this rate on load. "
                                   "Common Voice clips are 48 kHz mp3 → 16 kHz by default.")
        self.add_input("task_id",    PortType.STRING, default="default",
                       description="Pairs this dataset with a Train Output of the same task_name")
        # Always-present outputs
        self.add_output("x",          PortType.TENSOR,
                        description="Primary data tensor (images, sequences, features)")
        self.add_output("label",      PortType.TENSOR,
                        description="Labels / targets")
        self.add_output("dataloader", PortType.DATALOADER)
        self.add_output("vocab_size", PortType.INT,
                        description="Character vocab size (text mode only, 0 otherwise)")
        self.add_output("info",       PortType.STRING)

    def _build_dynamic_ports(self, columns, col_types):
        """Add output ports for columns beyond the standard x/label."""
        for col in columns:
            if col not in self.outputs and col not in ("x", "label"):
                self.add_output(col, PortType.TENSOR,
                                description=f"Column '{col}' ({col_types.get(col, '?')})")
        self._dynamic_ports_built = True
        self._col_info = dict(col_types)

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import torch
        from torch.utils.data import DataLoader

        path       = str(inputs.get("path") or "data/my_dataset")
        columns_f  = str(inputs.get("columns") or "")
        batch_size = max(1, int(inputs.get("batch_size") or 32))
        shuffle    = bool(inputs.get("shuffle", True))
        split      = str(inputs.get("split") or "train")
        seq_len    = max(0, int(inputs.get("seq_len") or 0))
        target_sr  = max(1000, int(inputs.get("target_sample_rate") or 16000))

        cfg = (path, columns_f, batch_size, shuffle, split, seq_len, target_sr)
        empty = {"x": None, "label": None, "dataloader": None,
                 "vocab_size": 0, "info": ""}

        if self._cached_loader is not None and self._cached_cfg == cfg:
            # Use cache
            pass
        else:
            fmt = _detect_format(path)
            result = None
            error = None

            if fmt == "common_voice":
                result, error = _load_common_voice(path, columns_f, split, target_sr=target_sr)
            elif fmt == "csv_manifest":
                result, error = _load_csv_manifest(path, columns_f, target_sr=target_sr)
            elif fmt == "csv_file":
                result, error = _load_csv_file(path, columns_f, target_sr=target_sr)
            elif fmt == "tsv_file":
                result, error = _load_tsv_file(path, columns_f, target_sr=target_sr)
            elif fmt == "tsv_manifest":
                result, error = _load_tsv_manifest(path, columns_f, split, target_sr=target_sr)
            elif fmt == "parquet_dir":
                result, error = _load_parquet_dir(path, columns_f)
            elif fmt == "parquet_file":
                result, error = _load_parquet_file(path, columns_f)
            elif fmt == "json_file":
                # Resolve to the actual file path
                if os.path.isfile(path):
                    json_path = path
                else:
                    for name in ("data.json", "data.jsonl", "manifest.json", "manifest.jsonl"):
                        candidate = os.path.join(path, name)
                        if os.path.exists(candidate):
                            json_path = candidate
                            break
                    else:
                        json_path = path
                result, error = _load_json_file(json_path, columns_f)
            elif fmt == "imagefolder":
                result, error = _load_imagefolder(path, columns_f)
            elif fmt == "text":
                sl = seq_len if seq_len > 0 else 64
                result, error = _load_text(path, sl)
            elif fmt == "torchvision":
                train = split.lower() in ("train", "")
                result, error = _load_torchvision(path, train, batch_size, shuffle)
            elif fmt == "huggingface":
                result, error = _load_huggingface(path, split, columns_f)
            else:
                error = f"Could not detect dataset format for: {path}"

            if error:
                return {**empty, "info": error}

            # Unpack result
            if len(result) == 6:
                ds, columns, col_types, label_maps, info, extra = result
            else:
                ds, columns, col_types, label_maps, info = result
                extra = {}

            self._extra = extra

            # Build dynamic ports for non-standard columns
            if not self._dynamic_ports_built:
                self._build_dynamic_ports(columns, col_types)

            # Create DataLoader
            try:
                self._cached_loader = DataLoader(
                    ds, batch_size=batch_size, shuffle=shuffle,
                    drop_last=True, collate_fn=_collate_manifest,
                )
            except Exception as exc:
                return {**empty, "info": f"DataLoader failed: {exc}"}

            self._cached_dataset = {
                "columns": columns, "col_types": col_types,
                "label_maps": label_maps, "info": info, "n_samples": len(ds),
            }
            self._cached_cfg = cfg

        loader = self._cached_loader
        result_info = self._cached_dataset

        # Sample one batch for tensor preview
        preview: dict[str, Any] = {}
        try:
            batch = next(iter(loader))
            if isinstance(batch, dict):
                for col in result_info["columns"]:
                    if col in batch:
                        preview[col] = batch[col]
            elif isinstance(batch, (list, tuple)):
                if len(batch) >= 1:
                    preview["x"] = batch[0]
                if len(batch) >= 2:
                    preview["label"] = batch[1]
        except Exception:
            pass

        # Map columns to standard x/label if they exist under other names
        x = preview.get("x")
        label = preview.get("label")
        # For parquet datasets: map observation.state → x, action → label
        if x is None:
            for key in ("observation.state", "observation", "features", "input"):
                if key in preview:
                    x = preview[key]
                    break
        if label is None:
            for key in ("action", "target", "targets"):
                if key in preview:
                    label = preview[key]
                    break

        # If still no x, use the first tensor column
        if x is None:
            for col in result_info["columns"]:
                if col in preview and col != "label":
                    x = preview[col]
                    break

        vocab_size = self._extra.get("vocab_size", 0)
        fmt_tag = _detect_format(str(inputs.get("path") or ""))

        out = {
            "x": x,
            "label": label,
            "dataloader": loader,
            "vocab_size": vocab_size,
            "info": f"[{fmt_tag}] {result_info['info']}",
            **{k: v for k, v in preview.items() if k not in ("x", "label")},
        }
        return out

    def export(self, iv, ov):
        path = self._val(iv, "path")
        bs   = self._val(iv, "batch_size")
        dl_var = ov.get("dataloader", "_ds_dl")
        return ["import torch", "from torch.utils.data import DataLoader"], [
            f"# Dataset from {path}",
            f"# Auto-detected format — implement loading for your export target",
            f"{dl_var} = None  # TODO: load from {path}",
        ]
