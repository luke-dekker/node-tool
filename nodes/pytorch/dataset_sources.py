"""Dataset source nodes — load raw data from files, folders, and hubs."""
from __future__ import annotations
from core.node import BaseNode
from core.node import PortType

CATEGORY = "Datasets"


class CSVDatasetNode(BaseNode):
    type_name = "pt_csv_dataset"
    label = "CSV Dataset"
    category = CATEGORY
    description = "Load a CSV file into a TensorDataset. feature_cols: comma-separated column names (blank = all except target). target_col: label column name."

    def _setup_ports(self):
        self.add_input("file_path",    PortType.STRING, default="data.csv")
        self.add_input("target_col",   PortType.STRING, default="label")
        self.add_input("feature_cols", PortType.STRING, default="")
        self.add_input("normalize",    PortType.BOOL,   default=False)
        self.add_output("dataset", PortType.DATASET)
        self.add_output("info",    PortType.STRING)

    def execute(self, inputs):
        try:
            import pandas as pd
            import torch
            from torch.utils.data import TensorDataset

            path = str(inputs.get("file_path") or "")
            target_col = str(inputs.get("target_col") or "label")
            feature_cols_raw = str(inputs.get("feature_cols") or "").strip()
            normalize = bool(inputs.get("normalize", False))

            df = pd.read_csv(path)

            if feature_cols_raw:
                feat_cols = [c.strip() for c in feature_cols_raw.split(",") if c.strip()]
            else:
                feat_cols = [c for c in df.columns if c != target_col]

            X = df[feat_cols].values.astype("float32")
            y = df[target_col].values

            if normalize:
                mean = X.mean(axis=0)
                std = X.std(axis=0) + 1e-8
                X = (X - mean) / std

            # Try int labels first, fallback to float
            try:
                y = y.astype("int64")
                y_tensor = torch.tensor(y, dtype=torch.long)
            except (ValueError, TypeError):
                y = y.astype("float32")
                y_tensor = torch.tensor(y, dtype=torch.float32)

            x_tensor = torch.tensor(X, dtype=torch.float32)
            dataset = TensorDataset(x_tensor, y_tensor)
            info = f"CSVDataset: {len(dataset)} samples, {len(feat_cols)} features"
            return {"dataset": dataset, "info": info}
        except Exception:
            import traceback
            return {"dataset": None, "info": traceback.format_exc().split("\n")[-2]}


class NumpyDatasetNode(BaseNode):
    type_name = "pt_numpy_dataset"
    label = "Numpy Dataset"
    category = CATEGORY
    description = "Wrap X (ndarray) and y (ndarray) into a TensorDataset."

    def _setup_ports(self):
        self.add_input("X", PortType.NDARRAY, default=None)
        self.add_input("y", PortType.NDARRAY, default=None)
        self.add_output("dataset", PortType.DATASET)
        self.add_output("info",    PortType.STRING)

    def execute(self, inputs):
        try:
            import torch
            from torch.utils.data import TensorDataset
            X = inputs.get("X")
            y = inputs.get("y")
            if X is None or y is None:
                return {"dataset": None, "info": "X and y required"}
            x_t = torch.tensor(X.astype("float32"))
            try:
                y_t = torch.tensor(y.astype("int64"), dtype=torch.long)
            except (ValueError, TypeError):
                y_t = torch.tensor(y.astype("float32"), dtype=torch.float32)
            dataset = TensorDataset(x_t, y_t)
            return {"dataset": dataset, "info": f"NumpyDataset: {len(dataset)} samples, shape {list(x_t.shape)}"}
        except Exception:
            import traceback
            return {"dataset": None, "info": traceback.format_exc().split("\n")[-2]}


class ImageFolderDatasetNode(BaseNode):
    type_name = "pt_image_folder_dataset"
    label = "Image Folder Dataset"
    category = CATEGORY
    description = "Load images from root/class_name/image.jpg folder structure (torchvision ImageFolder)."

    def _setup_ports(self):
        self.add_input("root_path",  PortType.STRING,    default="./data/images")
        self.add_input("transform",  PortType.TRANSFORM, default=None)
        self.add_output("dataset",     PortType.DATASET)
        self.add_output("class_names", PortType.STRING)
        self.add_output("info",        PortType.STRING)

    def execute(self, inputs):
        try:
            from torchvision.datasets import ImageFolder
            root = str(inputs.get("root_path") or "./data/images")
            transform = inputs.get("transform")
            dataset = ImageFolder(root=root, transform=transform)
            names = ", ".join(dataset.classes)
            info = f"ImageFolder: {len(dataset)} images, {len(dataset.classes)} classes"
            return {"dataset": dataset, "class_names": names, "info": info}
        except Exception:
            import traceback
            return {"dataset": None, "class_names": "", "info": traceback.format_exc().split("\n")[-2]}


class HuggingFaceDatasetNode(BaseNode):
    type_name = "pt_hf_dataset"
    label = "HuggingFace Dataset"
    category = CATEGORY
    description = "Load a dataset from HuggingFace Hub via datasets.load_dataset(). Outputs a TensorDataset for text classification."

    def _setup_ports(self):
        self.add_input("dataset_name", PortType.STRING, default="imdb")
        self.add_input("split",        PortType.STRING, default="train")
        self.add_input("text_col",     PortType.STRING, default="text")
        self.add_input("label_col",    PortType.STRING, default="label")
        self.add_input("max_samples",  PortType.INT,    default=0)
        self.add_output("dataset", PortType.DATASET)
        self.add_output("info",    PortType.STRING)

    def execute(self, inputs):
        try:
            from datasets import load_dataset
            import torch
            from torch.utils.data import Dataset as TorchDataset

            name   = str(inputs.get("dataset_name") or "imdb")
            split  = str(inputs.get("split") or "train")
            text_col  = str(inputs.get("text_col") or "text")
            label_col = str(inputs.get("label_col") or "label")
            max_n = int(inputs.get("max_samples") or 0)

            hf_ds = load_dataset(name, split=split)
            if max_n > 0:
                hf_ds = hf_ds.select(range(min(max_n, len(hf_ds))))

            class HFWrapper(TorchDataset):
                def __init__(self, ds, tc, lc):
                    self.ds = ds; self.tc = tc; self.lc = lc
                def __len__(self): return len(self.ds)
                def __getitem__(self, i):
                    item = self.ds[i]
                    return item[self.tc], item[self.lc]

            dataset = HFWrapper(hf_ds, text_col, label_col)
            info = f"HFDataset '{name}' [{split}]: {len(dataset)} samples"
            return {"dataset": dataset, "info": info}
        except Exception:
            import traceback
            return {"dataset": None, "info": traceback.format_exc().split("\n")[-2]}


class AudioFolderDatasetNode(BaseNode):
    type_name = "pt_audio_folder_dataset"
    label = "Audio Folder Dataset"
    category = CATEGORY
    description = "Load audio files from root/class_name/*.wav. Returns mel-spectrogram tensors via torchaudio."

    def _setup_ports(self):
        self.add_input("root_path",   PortType.STRING, default="./data/audio")
        self.add_input("sample_rate", PortType.INT,    default=16000)
        self.add_input("n_mels",      PortType.INT,    default=64)
        self.add_input("max_samples", PortType.INT,    default=0)
        self.add_output("dataset", PortType.DATASET)
        self.add_output("info",    PortType.STRING)

    def execute(self, inputs):
        try:
            import os, torch, torchaudio
            from torch.utils.data import Dataset as TorchDataset
            import torchaudio.transforms as T

            root = str(inputs.get("root_path") or "./data/audio")
            sr   = int(inputs.get("sample_rate") or 16000)
            n_mels = int(inputs.get("n_mels") or 64)
            max_n  = int(inputs.get("max_samples") or 0)

            classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
            class_to_idx = {c: i for i, c in enumerate(classes)}

            samples = []
            for cls in classes:
                cls_dir = os.path.join(root, cls)
                for fname in os.listdir(cls_dir):
                    if fname.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
                        samples.append((os.path.join(cls_dir, fname), class_to_idx[cls]))

            if max_n > 0:
                samples = samples[:max_n]

            mel_transform = T.MelSpectrogram(sample_rate=sr, n_mels=n_mels)

            class AudioDataset(TorchDataset):
                def __init__(self, s, mt, target_sr):
                    self.samples = s; self.mel = mt; self.target_sr = target_sr
                def __len__(self): return len(self.samples)
                def __getitem__(self, i):
                    path, label = self.samples[i]
                    waveform, orig_sr = torchaudio.load(path)
                    if orig_sr != self.target_sr:
                        waveform = torchaudio.functional.resample(waveform, orig_sr, self.target_sr)
                    spec = self.mel(waveform)
                    return spec, label

            dataset = AudioDataset(samples, mel_transform, sr)
            info = f"AudioDataset: {len(dataset)} files, {len(classes)} classes"
            return {"dataset": dataset, "info": info}
        except Exception:
            import traceback
            return {"dataset": None, "info": traceback.format_exc().split("\n")[-2]}


# Subcategory stamp
_SC = "Sources"
from core.node import BaseNode as _BN
for _n, _c in list(globals().items()):
    if isinstance(_c, type) and issubclass(_c, _BN) and _c is not _BN:
        _c.subcategory = _SC
