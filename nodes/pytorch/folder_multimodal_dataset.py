"""FolderMultimodalDataset — load a multi-modal dataset from a folder layout.

Expected layout:
    root/
      samples.csv          (id, label, audio, text, image, sensor, ...)
      audio/   (.wav, .mp3, .npy)
      text/    (.txt, .json)
      images/  (.png, .jpg)
      sensor/  (.npy, .csv)

samples.csv has one row per sample. Empty cells = that modality is absent for
that sample. If samples.csv is missing, the dataset auto-discovers files by
matching filenames across modality folders (sample_001.* groups together).
"""
from __future__ import annotations
from core.node import BaseNode, PortType


_MODALITY_DIRS = ("audio", "text", "image", "images", "video", "sensor")


def _load_audio(path: str):
    import numpy as np
    if path.endswith(".npy"):
        return np.load(path)
    try:
        import torchaudio
        wav, sr = torchaudio.load(path)
        return wav  # (channels, samples) torch.Tensor
    except Exception:
        return None


def _load_text(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


def _load_image(path: str):
    try:
        from PIL import Image
        import numpy as np
        return np.array(Image.open(path).convert("RGB"))
    except Exception:
        return None


def _load_sensor(path: str):
    import numpy as np
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith(".csv"):
        return np.loadtxt(path, delimiter=",")
    return None


_LOADERS = {
    "audio":  _load_audio,
    "text":   _load_text,
    "image":  _load_image,
    "images": _load_image,
    "video":  _load_image,   # placeholder — treat as image batch for now
    "sensor": _load_sensor,
}


class _MultimodalFolderDS:
    """The actual torch Dataset returned by the node."""

    def __init__(self, root: str, modalities: list[str], drop_missing: bool = False):
        import os, csv
        self.root        = root
        self.modalities  = modalities
        self.drop_missing = drop_missing
        self.samples: list[dict] = []   # each: {"id", "label", modality_name: filepath_or_None}

        manifest = os.path.join(root, "samples.csv")
        if os.path.isfile(manifest):
            self._load_from_manifest(manifest)
        else:
            self._discover_from_folders()

        if drop_missing:
            self.samples = [s for s in self.samples
                            if all(s.get(m) is not None for m in modalities)]

    def _load_from_manifest(self, path: str):
        import os, csv
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample = {"id": row.get("id", ""), "label": row.get("label", "")}
                for m in self.modalities:
                    fname = (row.get(m) or "").strip()
                    if not fname:
                        sample[m] = None
                    else:
                        sub = m if os.path.isdir(os.path.join(self.root, m)) else (m + "s")
                        sample[m] = os.path.join(self.root, sub, fname)
                self.samples.append(sample)

    def _discover_from_folders(self):
        """No manifest — discover by matching filenames across modality folders."""
        import os, glob
        # Build per-modality {basename_without_ext: full_path}
        per_mod = {}
        for m in self.modalities:
            sub = m if os.path.isdir(os.path.join(self.root, m)) else (m + "s")
            d = os.path.join(self.root, sub)
            if not os.path.isdir(d):
                per_mod[m] = {}
                continue
            files = {}
            for fp in sorted(os.listdir(d)):
                stem = os.path.splitext(fp)[0]
                files[stem] = os.path.join(d, fp)
            per_mod[m] = files

        # Union of all stems = all sample IDs
        all_ids: set[str] = set()
        for files in per_mod.values():
            all_ids.update(files.keys())

        for sample_id in sorted(all_ids):
            sample = {"id": sample_id, "label": ""}
            for m in self.modalities:
                sample[m] = per_mod.get(m, {}).get(sample_id)
            self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        loaded: dict = {}
        for m in self.modalities:
            path = s.get(m)
            if path is None:
                loaded[m] = None
            else:
                loader = _LOADERS.get(m, _LOADERS["sensor"])
                loaded[m] = loader(path)
        # Build modality mask
        try:
            import torch
            mask = torch.tensor(
                [1.0 if loaded[m] is not None else 0.0 for m in self.modalities]
            )
        except ImportError:
            mask = [1 if loaded[m] is not None else 0 for m in self.modalities]
        return {
            "data":  loaded,
            "mask":  mask,
            "label": s.get("label", ""),
            "id":    s.get("id", ""),
        }


class FolderMultimodalDatasetNode(BaseNode):
    type_name   = "pt_folder_multimodal_dataset"
    label       = "Folder Multimodal Dataset"
    category    = "Datasets"
    subcategory = "Sources"
    description = (
        "Load a multimodal dataset from a folder layout (audio/, text/, images/, sensor/). "
        "Optional samples.csv manifest with empty cells for missing modalities. "
        "Yields {data, mask, label, id} per sample."
    )

    def _setup_ports(self):
        self.add_input("root_path",  PortType.STRING, default="./data/my_multimodal",
                       description="Folder containing modality subdirs and optional samples.csv")
        self.add_input("modalities", PortType.STRING, default="audio,text,image",
                       description="Comma-separated list of modalities to load")
        self.add_input("drop_missing", PortType.BOOL, default=False,
                       description="If true, drop samples that lack any required modality")
        self.add_output("dataset", PortType.DATASET)
        self.add_output("info",    PortType.STRING)
        self.add_output("__terminal__", PortType.STRING)

    def execute(self, inputs):
        root  = inputs.get("root_path", "") or ""
        mods  = [m.strip() for m in (inputs.get("modalities") or "").split(",") if m.strip()]
        drop  = bool(inputs.get("drop_missing", False))
        if not root or not mods:
            return {"dataset": None, "info": "no root or modalities", "__terminal__": "[MM] Need root_path and modalities."}
        try:
            ds = _MultimodalFolderDS(root, mods, drop_missing=drop)
            n = len(ds)
            n_present = {m: sum(1 for s in ds.samples if s.get(m) is not None) for m in mods}
            info = f"{n} samples, " + ", ".join(f"{m}:{n_present[m]}" for m in mods)
            return {"dataset": ds, "info": info, "__terminal__": f"[MM] {root} -> {info}"}
        except Exception as exc:
            return {"dataset": None, "info": f"error: {exc}", "__terminal__": f"[MM] Error: {exc}"}

    def export(self, iv, ov):
        root = self._val(iv, "root_path")
        mods = self._val(iv, "modalities")
        return [], [
            f"# {self.label}: requires custom dataset class — see nodes/pytorch/folder_multimodal_dataset.py",
            f"{ov['dataset']} = None  # implement loading from {root} with modalities {mods}",
            f"{ov['info']} = ''",
        ]
