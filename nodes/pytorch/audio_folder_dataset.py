"""Audio Folder Dataset node."""
from __future__ import annotations
from core.node import BaseNode, PortType


class AudioFolderDatasetNode(BaseNode):
    type_name   = "pt_audio_folder_dataset"
    label       = "Audio Folder Dataset"
    category    = "Datasets"
    subcategory = "Sources"
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

    def export(self, iv, ov):
        return [], [f"# [{self.label}]: load dataset manually — audio loading requires runtime context"]
